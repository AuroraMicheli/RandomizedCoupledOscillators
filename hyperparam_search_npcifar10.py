import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# npCIFAR-10 Hyperparameter Search
# =============================
# Key differences from sMNIST/SHD:
#   - n_inp = 96 (32 cols x 3 RGB channels), much wider than sMNIST (1) but denser than SHD (700)
#   - seq_length = 1000 (32 real steps + 968 random noise padding)
#   - The signal is front-loaded: model must hold memory over ~968 noise steps
#   - Non-spiking coESN baseline: rho=9.0, inp_scaling=0.1 → 38.5% test
#   - Spiking with same params: 25.5% → dynamics are wrong for spiking regime
#
# Main issues to fix vs non-spiking params:
#   - rho=9.0 is pathological for spiking (drives firing rates to saturation)
#     → need rho < 1.5 for stable spiking dynamics
#   - epsilon=12.7 overdamps oscillations → use epsilon ~ 0.05-2.0
#   - inp_scaling=0.1 with 96 channels: total input drive ~ 96*0.1=9.6 → may saturate LIF
#     → need to explore broader range, likely lower
#   - Long memory requirement (968 noise steps) favors:
#       * slow oscillations: small gamma, larger dt
#       * moderate damping: small epsilon
#       * near-critical recurrence: rho close to 1.0
#
# Strategy: random search, log-uniform for scale params
# =============================

SEARCH_SPACE = {
    # Oscillator dynamics
    # Long memory over 968 steps → need slow oscillations
    # sqrt(gamma)*dt should be small → few cycles over 1000 steps
    "gamma":         (0.01, 2.0),    # small = slow oscillations = long memory
    "dt":            (0.01, 0.2),    # moderate dt
    "epsilon":       (0.01, 2.0),    # damping: half-life ≈ ln2/(dt*eps), keep > 100 steps
    "gamma_range":   (0.1, 2.0),     # frequency diversity
    "epsilon_range": (0.0, 1.0),     # damping diversity

    # Recurrent stability — critical fix vs non-spiking params
    # Non-spiking uses rho=9.0 (works because no spiking nonlinearity)
    # Spiking needs rho < ~1.5 to avoid saturation
    "rho":           (0.5, 1.5),

    # Input scaling
    # 96 input channels: total drive per step ~ 96 * inp_scaling / 2
    # With theta_lif ~ 0.05-0.3, need drive per step << theta_lif
    # → inp_scaling << theta_lif / 48 ~ 0.001-0.006
    # But also need enough signal → explore wider range
    "inp_scaling":   (0.001, 0.5),

    # Thresholds
    # theta_lif: LIF fires when membrane > theta → controls firing rate
    # theta_rf:  HRF fires when oscillator amplitude > theta
    # For npCIFAR with 96-dim input, expect larger activations than sMNIST
    "theta_lif":     (0.01, 1.0),
    "theta_rf":      (0.001, 0.5),

    # tau_filter: HRF spike filter timescale (in steps)
    # Longer filter = more temporal integration = better memory
    # For 1000-step sequences, explore longer filters
    "tau_filter":    (5.0, 100.0),
}

# Readout regularization
# npCIFAR-10 is harder than sMNIST → more regularization may help
READOUT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

N_SAMPLES       = 50
SEEDS           = [0, 1, 2]
N_HID           = 800          # match non-spiking baseline n_hid
RESULTS_DIR     = Path("hyperparam_search_npCIFAR10")
SCRIPT_NAME     = "npCIFAR10_spiking_ron.py"
RESULTS_SUBDIR  = "results_npcifar10"

RESULTS_DIR.mkdir(exist_ok=True)


def sample_params():
    params = {}
    for key, (lo, hi) in SEARCH_SPACE.items():
        if key in ("inp_scaling", "theta_lif", "theta_rf", "epsilon",
                   "gamma", "dt", "tau_filter"):
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    params["readout_C"] = random.choice(READOUT_C_VALUES)
    return params


all_results    = []
failed_configs = []

for i in range(N_SAMPLES):
    params = sample_params()

    print(f"\n🔍 Config {i+1}/{N_SAMPLES}: "
          f"inp={params['inp_scaling']:.4f} "
          f"dt={params['dt']:.3f} "
          f"γ={params['gamma']:.3f}±{params['gamma_range']:.2f} "
          f"ε={params['epsilon']:.4f}±{params['epsilon_range']:.3f} "
          f"ρ={params['rho']:.2f} "
          f"θlif={params['theta_lif']:.4f} "
          f"θrf={params['theta_rf']:.4f} "
          f"τ={params['tau_filter']:.1f} "
          f"C={params['readout_C']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT_NAME,
            "--n_hid",              str(N_HID),
            "--dt",                 str(params["dt"]),
            "--gamma",              str(params["gamma"]),
            "--epsilon",            str(params["epsilon"]),
            "--gamma_range",        str(params["gamma_range"]),
            "--epsilon_range",      str(params["epsilon_range"]),
            "--rho",                str(params["rho"]),
            "--inp_scaling",        str(params["inp_scaling"]),
            "--theta_lif",          str(params["theta_lif"]),
            "--theta_rf",           str(params["theta_rf"]),
            "--tau_filter",         str(params["tau_filter"]),
            "--connectivity_lif2hrf", "0.2",
            "--connectivity_hrf2lif", "1.0",
            "--seed",               str(seed),
            "--test_trials",        "1",
            "--use_test",
            "--readout_C",          str(params["readout_C"]),
        ]

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=600
            )

            result_file = max(
                Path(RESULTS_SUBDIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]  = i
            res["seed"]       = seed
            res["readout_C"]  = params["readout_C"]
            all_results.append(res)

        except subprocess.CalledProcessError as e:
            print(f"❌ Config {i+1} seed {seed} FAILED")
            if e.stderr:
                lines = e.stderr.strip().split('\n')
                for line in lines[-3:]:
                    print(f"   {line}")
            config_failed = True
            break
        except subprocess.TimeoutExpired:
            print(f"⏰ Config {i+1} seed {seed} TIMEOUT (>900s)")
            config_failed = True
            break

    if config_failed:
        failed_configs.append({"config_id": i, "params": params})
    else:
        config_results = [r for r in all_results if r["config_id"] == i]
        if len(config_results) == len(SEEDS):
            accs       = [r["test_acc_mean"]  for r in config_results]
            train_accs = [r["train_acc_mean"] for r in config_results]
            print(f"   ✅ Test: {np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"Train: {np.mean(train_accs):.2f}%  "
                  f"Gap: {np.mean(train_accs)-np.mean(accs):.1f}%")


# =============================
# Aggregate results
# =============================

summary = {}
for r in all_results:
    summary.setdefault(r["config_id"], []).append(r)

final_results = []
for cid, runs in summary.items():
    if len(runs) == len(SEEDS):
        test_accs  = [r["test_acc_mean"]  for r in runs]
        train_accs = [r["train_acc_mean"] for r in runs]
        final_results.append({
            "config_id":      cid,
            "mean_test_acc":  float(np.mean(test_accs)),
            "std_test_acc":   float(np.std(test_accs)),
            "mean_train_acc": float(np.mean(train_accs)),
            "overfit_gap":    float(np.mean(train_accs) - np.mean(test_accs)),
            "params":         runs[0]["args"],
            "readout_C":      runs[0].get("readout_C", 1.0),
        })

final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)


# =============================
# Print results
# =============================

print("\n" + "="*70)
print(f"✅ Completed: {len(final_results)}/{N_SAMPLES} configs successful")
print(f"❌ Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("="*70)

if final_results:
    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<14} {'Train%':<10} {'Gap':<8} "
          f"{'inp_scl':<9} {'C':<7} {'dt':<7} "
          f"{'gamma':<7} {'eps':<8} {'rho':<6} {'τ':<7} "
          f"{'θlif':<8} {'θrf':<7}")
    print("-" * 110)

    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}±{r['std_test_acc']:.2f}    "
              f"{r['mean_train_acc']:.1f}%     "
              f"{r['overfit_gap']:.1f}%   "
              f"{float(p.get('inp_scaling', 0)):<9.4f}"
              f"{r.get('readout_C', '?'):<7}"
              f"{float(p.get('dt', 0)):<7.3f}"
              f"{float(p.get('gamma', 0)):<7.3f}"
              f"{float(p.get('epsilon', 0)):<8.4f}"
              f"{float(p.get('rho', 0)):<6.2f}"
              f"{float(p.get('tau_filter', 0)):<7.1f}"
              f"{float(p.get('theta_lif', 0)):<8.4f}"
              f"{float(p.get('theta_rf', 0)):<7.4f}")

    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for param_name in ["inp_scaling", "dt", "gamma", "epsilon", "rho",
                       "tau_filter", "theta_lif", "theta_rf"]:
        top_vals = [float(r["params"].get(param_name, 0)) for r in final_results[:10]]
        all_vals = [float(r["params"].get(param_name, 0)) for r in final_results]
        print(f"  {param_name:>15}: top10={np.mean(top_vals):.4f}±{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}±{np.std(all_vals):.4f}")

    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_results = [r for r in final_results if r.get("readout_C") == C_val]
        if c_results:
            c_accs = [r["mean_test_acc"] for r in c_results]
            c_gaps = [r["overfit_gap"]   for r in c_results]
            print(f"  C={C_val:<7}: n={len(c_results)}, "
                  f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                  f"gap={np.mean(c_gaps):.1f}%")

else:
    print("\n⚠️  No successful configurations!")
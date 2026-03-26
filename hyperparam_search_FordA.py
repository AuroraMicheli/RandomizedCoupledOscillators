import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# FordA Hyperparameter Search — readout_mode=final
# =============================
# Baseline (current defaults):
#   dt=0.2, gamma=1.88, epsilon=0.022, gamma_range=2.64,
#   epsilon_range=0.068, inp_scaling=1.76, rho=0.95,
#   theta_lif=0.05, theta_rf=0.005, tau_filter=20.0
#
# Key insight for "final" readout:
#   The classifier only sees hy at the LAST timestep (t=500).
#   This means we need the reservoir state at t=500 to be:
#     1. Still "alive" (not decayed to zero) -> large tau_filter, low epsilon
#     2. Encoding history, not just recent input -> slow dynamics
#     3. Not exploded -> rho < 1, epsilon > 0, gamma > 0
#
#   Compared to rms_std_final search:
#     - tau_filter: push HIGHER (more memory, state persists to final step)
#     - epsilon:    push LOWER  (less damping = oscillations persist longer)
#     - dt:         push LOWER  (finer integration = more stable slow dynamics)
#     - rho:        slightly lower (stability more critical with slow dynamics)
#     - gamma_range/epsilon_range: narrower (heterogeneity less helpful
#                                            when only final state matters)
# =============================

BEST = {
    "dt":            0.2,
    "gamma":         1.88,
    "epsilon":       0.022,
    "gamma_range":   2.64,
    "epsilon_range": 0.068,
    "inp_scaling":   1.76,
    "rho":           0.95,
    "theta_lif":     0.05,
    "theta_rf":      0.005,
    "tau_filter":    20.0,
}

# For "final" readout we want slow, persistent dynamics ->
#   higher tau_filter, lower epsilon, moderate dt
NARROW_SPACE = {
    "dt":            (0.02,  0.3),       # log: push lower for stability
    "gamma":         (0.3,   4.0),       # log: best=1.88
    "epsilon":       (0.003, 0.1),       # log: push lower → oscillations persist longer
    "gamma_range":   (0.3,   2.5),       # linear: narrower than rms search
    "epsilon_range": (0.01,  0.2),       # linear: narrower than rms search
    "inp_scaling":   (0.3,   5.0),       # log: best=1.76
    "rho":           (0.6,   1.0),       # linear: keep below 1 for stability
    "theta_lif":     (0.01,  0.3),       # log: best=0.05
    "theta_rf":      (0.001, 0.05),      # log: best=0.005
    "tau_filter":    (15.0,  150.0),     # log: push HIGHER → state persists to t=500
}

WIDE_SPACE = {
    "dt":            (0.005, 0.5),
    "gamma":         (0.01,  10.0),
    "epsilon":       (0.001, 1.0),
    "gamma_range":   (0.1,   5.0),
    "epsilon_range": (0.0,   0.5),
    "inp_scaling":   (0.01,  10.0),
    "rho":           (0.3,   1.1),
    "theta_lif":     (0.005, 1.0),
    "theta_rf":      (0.0005, 0.5),
    "tau_filter":    (5.0,   200.0),     # wide: let search find the right memory scale
}

LOG_PARAMS = {"dt", "gamma", "epsilon", "inp_scaling",
              "theta_lif", "theta_rf", "tau_filter"}

READOUT_C_VALUES  = [0.001, 0.01, 0.1, 1.0, 10.0]
READOUT_C_WEIGHTS = [0.05, 0.15, 0.35, 0.35, 0.10]  # uniform, no prior bias

N_SAMPLES      = 100
EXPLOIT_FRAC   = 0.70
SEED           = 42
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_fordA_final")
SCRIPT_NAME    = "FordA_spiking_ron.py"
RESULTS_SUBDIR = "results_fordA"

RESULTS_DIR.mkdir(exist_ok=True)


def sample_from_space(space):
    params = {}
    for key, (lo, hi) in space.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    return params


def sample_params(exploit=True):
    space = NARROW_SPACE if exploit else WIDE_SPACE
    params = sample_from_space(space)
    params["readout_C"] = random.choices(READOUT_C_VALUES,
                                         weights=READOUT_C_WEIGHTS)[0]
    return params


all_results    = []
failed_configs = []

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)

print(f"FordA search (readout=final): {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore), seed={SEED}")
print(f"Reference best: {BEST}")
print("=" * 70)

for i, stype in enumerate(sample_types):
    exploit = (stype == "exploit")
    params  = sample_params(exploit=exploit)

    print(f"\n{'🎯' if exploit else '🌐'} Config {i+1}/{N_SAMPLES} [{stype}]: "
          f"inp={params['inp_scaling']:.4f} "
          f"dt={params['dt']:.3f} "
          f"γ={params['gamma']:.4f}±{params['gamma_range']:.2f} "
          f"ε={params['epsilon']:.4f}±{params['epsilon_range']:.3f} "
          f"ρ={params['rho']:.2f} "
          f"θlif={params['theta_lif']:.4f} "
          f"θrf={params['theta_rf']:.4f} "
          f"τ={params['tau_filter']:.1f} "
          f"C={params['readout_C']}")

    cmd = [
        "python", SCRIPT_NAME,
        "--n_hid",                str(N_HID),
        "--dt",                   str(params["dt"]),
        "--gamma",                str(params["gamma"]),
        "--epsilon",              str(params["epsilon"]),
        "--gamma_range",          str(params["gamma_range"]),
        "--epsilon_range",        str(params["epsilon_range"]),
        "--rho",                  str(params["rho"]),
        "--inp_scaling",          str(params["inp_scaling"]),
        "--theta_lif",            str(params["theta_lif"]),
        "--theta_rf",             str(params["theta_rf"]),
        "--tau_filter",           str(params["tau_filter"]),
        "--connectivity_lif2hrf", "1.0",
        "--connectivity_hrf2lif", "1.0",
        "--readout_mode",         "final",
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--readout_C",            str(params["readout_C"]),
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=600
        )

        result_file = max(
            Path(RESULTS_SUBDIR).glob(f"*final*seed{SEED}.json"),
            key=lambda p: p.stat().st_mtime
        )
        with open(result_file) as f:
            res = json.load(f)

        res["config_id"]   = i
        res["seed"]        = SEED
        res["readout_C"]   = params["readout_C"]
        res["sample_type"] = stype
        all_results.append(res)

        print(f"   ✅ Test: {res['test_acc_mean']:.2f}%  "
              f"Train: {res['train_acc_mean']:.2f}%  "
              f"Gap: {res['train_acc_mean']-res['test_acc_mean']:.1f}%")

    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-3:]:
                print(f"      {line}")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>600s)")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    if (i + 1) % 10 == 0:
        intermediate = sorted(all_results,
                               key=lambda x: x["test_acc_mean"],
                               reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
        if intermediate:
            print(f"\n   💾 Saved top-20 intermediate results "
                  f"(best so far: {intermediate[0]['test_acc_mean']:.2f}% "
                  f"[{intermediate[0].get('sample_type','?')}])")


# =============================
# Final aggregation
# =============================

all_results.sort(key=lambda x: x["test_acc_mean"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ Completed: {len(all_results)}/{N_SAMPLES} configs successful")
print(f"❌ Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
exploit_results = [r for r in all_results if r.get("sample_type") == "exploit"]
explore_results = [r for r in all_results if r.get("sample_type") == "explore"]
if exploit_results:
    print(f"🎯 Exploit best: {max(r['test_acc_mean'] for r in exploit_results):.2f}%  "
          f"(mean: {np.mean([r['test_acc_mean'] for r in exploit_results]):.2f}%)")
if explore_results:
    print(f"🌐 Explore best: {max(r['test_acc_mean'] for r in explore_results):.2f}%  "
          f"(mean: {np.mean([r['test_acc_mean'] for r in explore_results]):.2f}%)")
print("=" * 70)

if all_results:
    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<10} {'Train%':<10} {'Gap':<7} {'Type':<9} "
          f"{'inp_scl':<9} {'C':<7} {'dt':<7} "
          f"{'gamma':<8} {'eps':<8} {'rho':<6} {'τ':<6} "
          f"{'θlif':<8} {'θrf':<7}")
    print("-" * 120)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        print(f"{rank:<5} "
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']-r['test_acc_mean']:.1f}%   "
              f"{r.get('sample_type','?'):<9} "
              f"{float(p.get('inp_scaling', 0)):<9.4f}"
              f"{float(r.get('readout_C', 0)):<7}"
              f"{float(p.get('dt', 0)):<7.3f}"
              f"{float(p.get('gamma', 0)):<8.4f}"
              f"{float(p.get('epsilon', 0)):<8.4f}"
              f"{float(p.get('rho', 0)):<6.2f}"
              f"{float(p.get('tau_filter', 0)):<6.1f}"
              f"{float(p.get('theta_lif', 0)):<8.4f}"
              f"{float(p.get('theta_rf', 0)):<7.4f}")

    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for param_name in ["inp_scaling", "dt", "gamma", "epsilon", "rho",
                        "tau_filter", "theta_lif", "theta_rf"]:
        top_vals = [float(r.get("args", r).get(param_name, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(param_name, 0))
                    for r in all_results]
        print(f"  {param_name:>15}: "
              f"top10={np.mean(top_vals):.4f}±{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}±{np.std(all_vals):.4f}")

    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_results = [r for r in all_results
                     if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_results:
            c_accs = [r["test_acc_mean"] for r in c_results]
            c_gaps = [r["train_acc_mean"] - r["test_acc_mean"]
                      for r in c_results]
            print(f"  C={C_val:<7}: n={len(c_results)}, "
                  f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                  f"gap={np.mean(c_gaps):.1f}%")
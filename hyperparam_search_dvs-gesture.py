"""
DVS Gesture hyperparam search — n_hid=3000 (fixed).

Why n_hid=3000 needs its own search:
  - Recurrent drive per neuron scales as sqrt(512/3000) ~ 0.41x vs n_hid=512
    -> theta_lif must come DOWN to maintain the same sparse firing regime
    -> Expected optimal theta_lif: 0.4-0.8 (vs 0.926 at n_hid=512)
  - Input drive per neuron is UNCHANGED (depends only on inp_scaling,
    density, n_inp — not on n_hid), so inp_scaling/density ranges stay the same
  - Readout: 3000 features / 1077 samples = 2.79x overdetermined
    -> C must be lower than the C=0.01 that won at n_hid=512
    -> Suggested C range: 0.001-0.01
  - rho: super-critical (>1.0) confirmed from v1; keep same range
  - T=200, sf=4 fixed as clear winners from v1 search

Starting point: the best v1 config at n_hid=3000 gives 74.24%.
Target: find the operating regime where 3000 neurons give meaningfully better
separation than 512, which should push well past 80%.
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fixed settings
# ==============================

N_HID      = 3000
NUM_STEPS  = 200
SCRIPT     = "dvs-gesture_spiking_ron.py"   # check your filename!
RESULTS_DIR            = Path("hyperparam_search_DVSGesture_3000")
DVSGESTURE_RESULTS_DIR = "results_dvsgesture_3000"

RESULTS_DIR.mkdir(exist_ok=True)

# ==============================
# Search space — rescaled for n_hid=3000
# ==============================

SEARCH_SPACE = {
    # Oscillator dynamics — independent of n_hid, keep from v1
    "gamma":         (0.003, 0.08),   # slow oscillations confirmed
    "dt":            (0.1,   0.4),
    "epsilon":       (0.003, 0.12),
    "gamma_range":   (0.003, 0.15),
    "epsilon_range": (0.0,   0.08),
    "rho":           (0.9,   1.6),    # super-critical bias confirmed

    # Input drive — independent of n_hid
    "input_scaling": (0.01, 0.2),     # low range confirmed from v1
    "input_density": (0.01, 0.1),     # low-sparse regime confirmed

    # Thresholds — theta_lif MUST come down for n_hid=3000
    # Recurrent drive per neuron is 0.41x of n_hid=512
    # v1 winner: theta_lif=0.926 -> rescaled: 0.926 * 0.41 ~ 0.38
    # Search around this: [0.05, 1.5] with log-uniform (center ~0.4)
    "theta_lif":     (0.05, 1.5),
    "theta_rf":      (0.001, 0.04),
}

# Spatial factor: sf=4 (2048ch) and sf=8 (512ch)
# sf=4 won in v1 (config 35 used sf=8 but config 12 at 63% used sf=4)
# Keep both in search
SPATIAL_FACTOR_OPTIONS = [4, 8]

# C: with 3000 features / 1077 samples = 2.79x overdetermined
# Scale down from winning C=0.01 at n_hid=512
# C=0.01 * (512/3000) ~ 0.0017, so search around 0.001-0.02
READOUT_C_VALUES = [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]

# readout mode: mean won in v1, final was runner-up. Drop rms_std_final.
READOUT_MODE_OPTIONS = ['mean', 'final']

# ==============================
# Search settings
# ==============================

N_SAMPLES = 60
SEEDS     = [0, 1, 2, 3, 4]   # 5 seeds for reliable mean±std

# ==============================
# Sampling — 50% forced super-critical rho (strong prior from v1)
# ==============================

def sample_params():
    params = {}
    for key, (lo, hi) in SEARCH_SPACE.items():
        if key in ("input_scaling", "input_density", "theta_lif", "theta_rf",
                   "epsilon", "gamma", "dt"):
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)

    params["readout_C"]      = random.choice(READOUT_C_VALUES)
    params["spatial_factor"] = random.choice(SPATIAL_FACTOR_OPTIONS)
    params["readout_mode"]   = random.choice(READOUT_MODE_OPTIONS)

    # 60% chance to force super-critical rho (strong prior from v1)
    if random.random() < 0.6:
        params["rho"] = random.uniform(1.0, 1.6)

    return params

# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

for i in range(N_SAMPLES):
    params = sample_params()

    sf    = params["spatial_factor"]
    n_ch  = 2 * (128 // sf) ** 2
    n_feat = N_HID  # mean or final both give n_hid features

    # Oscillation coverage diagnostic
    osc = np.sqrt(params["gamma"]) * params["dt"] * NUM_STEPS
    osc_flag = "" if 1.0 <= osc <= 6.0 else f" [osc={osc:.2f} outside [1,6]!]"

    print(f"\nConfig {i+1}/{N_SAMPLES}: "
          f"sf={sf}({n_ch}ch) mode={params['readout_mode']} C={params['readout_C']} "
          f"inp={params['input_scaling']:.4f} dens={params['input_density']:.4f} "
          f"dt={params['dt']:.3f} "
          f"g={params['gamma']:.4f}+/-{params['gamma_range']:.4f} "
          f"e={params['epsilon']:.4f}+/-{params['epsilon_range']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} th_rf={params['theta_rf']:.4f}"
          f"{osc_flag}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid",                str(N_HID),
            "--spatial_factor",       str(params["spatial_factor"]),
            "--num_steps",            str(NUM_STEPS),
            "--dt",                   str(params["dt"]),
            "--gamma",                str(params["gamma"]),
            "--gamma_range",          str(params["gamma_range"]),
            "--epsilon",              str(params["epsilon"]),
            "--epsilon_range",        str(params["epsilon_range"]),
            "--rho",                  str(params["rho"]),
            "--inp_scaling",          str(params["input_scaling"]),
            "--input_density",        str(params["input_density"]),
            "--theta_lif",            str(params["theta_lif"]),
            "--theta_rf",             str(params["theta_rf"]),
            "--connectivity_lif2hrf", "1.0",
            "--connectivity_hrf2lif", "1.0",
            "--seed",                 str(seed),
            "--test_trials",          "1",
            "--use_test",
            "--readout_C",            str(params["readout_C"]),
            "--readout_mode",         params["readout_mode"],
            "--results_dir",          DVSGESTURE_RESULTS_DIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=900)

            result_file = max(
                Path(DVSGESTURE_RESULTS_DIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]      = i
            res["search_seed"]    = seed
            res["readout_C"]      = params["readout_C"]
            res["spatial_factor"] = params["spatial_factor"]
            res["readout_mode"]   = params["readout_mode"]
            all_results.append(res)

        except subprocess.CalledProcessError as e:
            print(f"  FAILED (seed {seed})")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"    {line}")
            config_failed = True
            break
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (seed {seed}, >900s)")
            config_failed = True
            break

    if config_failed:
        failed_configs.append({"config_id": i, "params": params})
    else:
        config_results = [r for r in all_results if r["config_id"] == i]
        if len(config_results) == len(SEEDS):
            accs       = [r["test_acc_mean"]  for r in config_results]
            train_accs = [r["train_acc_mean"] for r in config_results]
            r_hrf      = np.mean([r["r_hrf_mean"] for r in config_results])
            r_lif      = np.mean([r["r_lif_mean"] for r in config_results])
            print(f"  -> Test: {np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"Train: {np.mean(train_accs):.2f}%  "
                  f"Gap: {np.mean(train_accs)-np.mean(accs):.1f}%  "
                  f"r_hrf={r_hrf:.4f} r_lif={r_lif:.4f}")

# ==============================
# Aggregate
# ==============================

summary = {}
for r in all_results:
    summary.setdefault(r["config_id"], []).append(r)

final_results = []
for cid, runs in summary.items():
    if len(runs) == len(SEEDS):
        test_accs  = [r["test_acc_mean"]  for r in runs]
        train_accs = [r["train_acc_mean"] for r in runs]
        r_hrf_vals = [r["r_hrf_mean"]     for r in runs]
        r_lif_vals = [r["r_lif_mean"]     for r in runs]
        final_results.append({
            "config_id":      cid,
            "mean_test_acc":  float(np.mean(test_accs)),
            "std_test_acc":   float(np.std(test_accs)),
            "mean_train_acc": float(np.mean(train_accs)),
            "overfit_gap":    float(np.mean(train_accs) - np.mean(test_accs)),
            "spatial_factor": runs[0]["spatial_factor"],
            "readout_mode":   runs[0]["readout_mode"],
            "readout_C":      runs[0].get("readout_C", 1.0),
            "r_hrf_mean":     float(np.mean(r_hrf_vals)),
            "r_lif_mean":     float(np.mean(r_lif_vals)),
            "params":         runs[0]["args"],
        })

final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

# ==============================
# Print results
# ==============================

print("\n" + "="*70)
print(f"Completed: {len(final_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("="*70)

if final_results:
    print(f"\nTOP 10 CONFIGURATIONS (n_hid=3000, T=200):")
    print(f"{'Rank':<5} {'Test%':<14} {'Train%':<10} {'Gap':<7} "
          f"{'sf':<4} {'mode':<7} {'C':<8} "
          f"{'inp':<8} {'dens':<7} {'dt':<6} "
          f"{'gamma':<8} {'g_rng':<7} {'eps':<7} {'e_rng':<7} "
          f"{'rho':<6} {'th_lif':<8} {'th_rf':<8} "
          f"{'r_hrf':<7} {'r_lif'}")
    print("-" * 155)

    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}+/-{r['std_test_acc']:.2f}  "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{r['spatial_factor']:<4} "
              f"{r['readout_mode']:<7} "
              f"{r['readout_C']:<8} "
              f"{p.get('inp_scaling',0):<8.4f}"
              f"{p.get('input_density',0):<7.4f}"
              f"{p.get('dt',0):<6.3f}"
              f"{p.get('gamma',0):<8.4f}"
              f"{p.get('gamma_range',0):<7.4f}"
              f"{p.get('epsilon',0):<7.4f}"
              f"{p.get('epsilon_range',0):<7.4f}"
              f"{p.get('rho',0):<6.3f}"
              f"{p.get('theta_lif',0):<8.4f}"
              f"{p.get('theta_rf',0):<8.4f}"
              f"{r['r_hrf_mean']:<7.4f}"
              f"{r['r_lif_mean']:.4f}")

    # Parameter trends
    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "dt", "gamma", "gamma_range",
                  "epsilon", "epsilon_range", "rho", "theta_lif", "theta_rf"]:
        top_vals = [r["params"].get(pname, 0) for r in final_results[:10]]
        all_vals = [r["params"].get(pname, 0) for r in final_results]
        print(f"  {pname:>15}: top10={np.mean(top_vals):.5f}+/-{np.std(top_vals):.5f}  "
              f"all={np.mean(all_vals):.5f}+/-{np.std(all_vals):.5f}")

    # Firing rate analysis — critical for diagnosing regime
    print(f"\nFIRING RATE ANALYSIS (top 10):")
    print(f"  {'Rank':<5} {'r_hrf':<8} {'r_lif':<8} {'test%'}")
    for rank, r in enumerate(final_results[:10], 1):
        print(f"  {rank:<5} {r['r_hrf_mean']:<8.4f} {r['r_lif_mean']:<8.4f} "
              f"{r['mean_test_acc']:.2f}%")
    print("  (target: r_lif ~0.05-0.3, r_hrf ~0.01-0.1 for sparse selective coding)")

    # spatial_factor breakdown
    print(f"\nSPATIAL_FACTOR BREAKDOWN:")
    for sf in SPATIAL_FACTOR_OPTIONS:
        sf_res = [r for r in final_results if r.get("spatial_factor") == sf]
        n_ch = 2 * (128 // sf) ** 2
        if sf_res:
            accs = [r["mean_test_acc"] for r in sf_res]
            print(f"  sf={sf} ({n_ch} ch): n={len(sf_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  best={max(accs):.2f}%")

    # readout mode breakdown
    print(f"\nREADOUT MODE BREAKDOWN:")
    for mode in READOUT_MODE_OPTIONS:
        m_res = [r for r in final_results if r["readout_mode"] == mode]
        if m_res:
            accs = [r["mean_test_acc"] for r in m_res]
            gaps = [r["overfit_gap"]   for r in m_res]
            print(f"  {mode:<7}: n={len(m_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  avg_gap={np.mean(gaps):.1f}%")

    # C breakdown
    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final_results if r.get("readout_C") == C_val]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            gaps = [r["overfit_gap"]   for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  avg_gap={np.mean(gaps):.1f}%")

else:
    print("\nNo successful configurations.")
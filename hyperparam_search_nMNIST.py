"""
Hyperparameter search for Spiking RON on N-MNIST.

Key differences from SHD search:
  - T=20 (or ~30-50) is VERY SHORT. Oscillators must complete meaningful dynamics
    within T steps: need sqrt(gamma)*dt*T ~ 1-5 (i.e. 0.3-2.5 cycles in T steps).
    With T=20: sqrt(gamma)*dt must be in [0.015, 0.125] per step.
  - 578 input channels (spatial_factor=2), sparse binary events (~10% active/bin).
    Avg drive per neuron = input_density * 578 * activity * inp_scaling * 0.5
    → need inp_scaling tuned so LIF fires at moderate rate.
  - theta_lif must be low enough that LIF neurons actually fire given sparse input.
  - More time bins (num_steps) generally helps — search includes this too.
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Search space
# ==============================

SEARCH_SPACE = {
    # Oscillator dynamics
    # Target: sqrt(gamma)*dt*T ~ 1-5 cycles in T steps
    # With T=30: sqrt(gamma)*dt in [0.03, 0.17]
    # → gamma in [0.01, 0.5] combined with dt in [0.1, 0.5] covers this
    "gamma":         (0.01, 0.8),
    "dt":            (0.1,  0.6),
    "epsilon":       (0.01, 0.15),
    "gamma_range":   (0.01, 0.8),   # heterogeneity should roughly match gamma range
    "epsilon_range": (0.0,  0.10),
    "rho":           (0.5,  1.4),

    # Input
    # 578 channels (sf=2), ~10% active/bin → ~58 active/bin
    # Per-neuron drive = input_density * 58 * inp_scaling * 0.5
    # Want drive ~ 0.1-1.0 per step so LIF integrates to threshold in ~5-15 steps
    "input_scaling": (0.05, 2.0),
    "input_density": (0.05, 0.5),   # wider range than SHD (578 ch vs 700)

    # Thresholds
    # theta_lif: LIF fires when membrane > threshold.
    # With small inp_scaling/density → need low threshold. Allow 0.01-0.5.
    "theta_lif":     (0.01, 0.5),
    "theta_rf":      (0.001, 0.05),
}

# num_steps: short sequences hurt dynamics; test a few values
NUM_STEPS_OPTIONS    = [20, 30, 50]
SPATIAL_FACTOR_OPTIONS = [1, 2]

# Readout regularization
READOUT_C_VALUES = [0.01, 0.1, 1.0, 10.0]

# Readout mode: rms_std_final has 3x features and suits short sequences well
READOUT_MODE_OPTIONS = ['rms_std_final', 'final', 'mean']

# ==============================
# Search settings
# ==============================

N_SAMPLES    = 60           # random configs
SEEDS        = [0, 1, 2]    # 3 seeds per config → mean ± std
N_HID        = 512          # fixed (fast enough to search with)
SCRIPT       = "nMNIST_spiking_ron.py"
RESULTS_DIR  = Path("hyperparam_search_NMNIST")
NMNIST_RESULTS_DIR = "results_nmnist_search"

RESULTS_DIR.mkdir(exist_ok=True)

# ==============================
# Sampling
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
    params["num_steps"]      = random.choice(NUM_STEPS_OPTIONS)
    params["spatial_factor"] = random.choice(SPATIAL_FACTOR_OPTIONS)
    params["readout_mode"]   = random.choice(READOUT_MODE_OPTIONS)
    return params

# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

for i in range(N_SAMPLES):
    params = sample_params()

    sf = params["spatial_factor"]
    n_ch = 2 * (34 // sf) ** 2
    print(f"\nConfig {i+1}/{N_SAMPLES}: "
          f"T={params['num_steps']} sf={params['spatial_factor']}({n_ch}ch) "
          f"inp={params['input_scaling']:.3f} "
          f"dens={params['input_density']:.3f} "
          f"dt={params['dt']:.3f} "
          f"g={params['gamma']:.3f}+/-{params['gamma_range']:.3f} "
          f"e={params['epsilon']:.4f}+/-{params['epsilon_range']:.3f} "
          f"rho={params['rho']:.2f} "
          f"th_lif={params['theta_lif']:.4f} "
          f"th_rf={params['theta_rf']:.4f} "
          f"C={params['readout_C']} "
          f"mode={params['readout_mode']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid",           str(N_HID),
            "--spatial_factor",  str(params["spatial_factor"]),
            "--num_steps",       str(params["num_steps"]),
            "--dt",              str(params["dt"]),
            "--gamma",           str(params["gamma"]),
            "--epsilon",         str(params["epsilon"]),
            "--gamma_range",     str(params["gamma_range"]),
            "--epsilon_range",   str(params["epsilon_range"]),
            "--rho",             str(params["rho"]),
            "--inp_scaling",     str(params["input_scaling"]),
            "--input_density",   str(params["input_density"]),
            "--theta_lif",       str(params["theta_lif"]),
            "--theta_rf",        str(params["theta_rf"]),
            "--connectivity_lif2hrf", "1.0",
            "--connectivity_hrf2lif", "1.0",
            "--seed",            str(seed),
            "--test_trials",     "1",
            "--use_test",
            "--readout_C",       str(params["readout_C"]),
            "--readout_mode",    params["readout_mode"],
            "--results_dir",     NMNIST_RESULTS_DIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)

            result_file = max(
                Path(NMNIST_RESULTS_DIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]      = i
            res["search_seed"]    = seed
            res["readout_C"]      = params["readout_C"]
            res["num_steps"]      = params["num_steps"]
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
            print(f"  TIMEOUT (seed {seed}, >600s)")
            config_failed = True
            break

    if config_failed:
        failed_configs.append({"config_id": i, "params": params})
    else:
        config_results = [r for r in all_results if r["config_id"] == i]
        if len(config_results) == len(SEEDS):
            accs       = [r["test_acc_mean"]  for r in config_results]
            train_accs = [r["train_acc_mean"] for r in config_results]
            print(f"  -> Test: {np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"Train: {np.mean(train_accs):.2f}%  "
                  f"Gap: {np.mean(train_accs)-np.mean(accs):.1f}%")

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
        final_results.append({
            "config_id":      cid,
            "mean_test_acc":  float(np.mean(test_accs)),
            "std_test_acc":   float(np.std(test_accs)),
            "mean_train_acc": float(np.mean(train_accs)),
            "overfit_gap":    float(np.mean(train_accs) - np.mean(test_accs)),
            "num_steps":      runs[0]["num_steps"],
            "spatial_factor": runs[0]["spatial_factor"],
            "readout_mode":   runs[0]["readout_mode"],
            "readout_C":      runs[0].get("readout_C", 1.0),
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
    print(f"\nTOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<14} {'Train%':<10} {'Gap':<7} "
          f"{'T':<4} {'sf':<3} {'mode':<14} {'C':<6} "
          f"{'inp':<7} {'dens':<7} {'dt':<6} "
          f"{'gamma':<7} {'eps':<7} {'rho':<6} "
          f"{'th_lif':<8} {'th_rf'}")
    print("-" * 120)

    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}+/-{r['std_test_acc']:.2f}  "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{r['num_steps']:<4} "
              f"{r.get('spatial_factor','-'):<3} "
              f"{r['readout_mode']:<14} "
              f"{r['readout_C']:<6} "
              f"{p.get('inp_scaling',0):<7.3f}"
              f"{p.get('input_density',0):<7.3f}"
              f"{p.get('dt',0):<6.3f}"
              f"{p.get('gamma',0):<7.3f}"
              f"{p.get('epsilon',0):<7.4f}"
              f"{p.get('rho',0):<6.2f}"
              f"{p.get('theta_lif',0):<8.4f}"
              f"{p.get('theta_rf',0):.4f}")

    # Parameter trends: top 10 vs all
    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "dt", "gamma", "epsilon", "rho",
                  "theta_lif", "theta_rf"]:
        top_vals = [r["params"].get(pname, 0) for r in final_results[:10]]
        all_vals = [r["params"].get(pname, 0) for r in final_results]
        print(f"  {pname:>15}: top10={np.mean(top_vals):.4f}+/-{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}+/-{np.std(all_vals):.4f}")

    # num_steps breakdown
    print(f"\nNUM_STEPS BREAKDOWN:")
    for T in NUM_STEPS_OPTIONS:
        t_res = [r for r in final_results if r["num_steps"] == T]
        if t_res:
            accs = [r["mean_test_acc"] for r in t_res]
            print(f"  T={T:<4}: n={len(t_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    # spatial_factor breakdown
    print(f"\nSPATIAL_FACTOR BREAKDOWN:")
    for sf in SPATIAL_FACTOR_OPTIONS:
        sf_res = [r for r in final_results if r.get("spatial_factor") == sf]
        n_ch = 2 * (34 // sf) ** 2
        if sf_res:
            accs = [r["mean_test_acc"] for r in sf_res]
            print(f"  sf={sf} ({n_ch:>4} ch): n={len(sf_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    # readout_mode breakdown
    print(f"\nREADOUT MODE BREAKDOWN:")
    for mode in READOUT_MODE_OPTIONS:
        m_res = [r for r in final_results if r["readout_mode"] == mode]
        if m_res:
            accs = [r["mean_test_acc"] for r in m_res]
            print(f"  {mode:<16}: n={len(m_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    # Regularization breakdown
    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final_results if r.get("readout_C") == C_val]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            gaps = [r["overfit_gap"]   for r in c_res]
            print(f"  C={C_val:<6}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"gap={np.mean(gaps):.1f}%")

else:
    print("\nNo successful configurations.")
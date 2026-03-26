"""
DVS Gesture hyperparam search — Phase 2 (Dual Regime)
n_hid=3000, sf=4, num_steps=200, readout_mode=mean fixed.

Two promising regimes identified:
  A) SLOW + SUPERCRITICAL (Phase 1 best, config_id=53, 74.24%):
       gamma~0.051, rho~1.27, theta_lif~1.50, dt~0.30
       Slow oscillations, amplifying recurrence
  B) FAST + SUBCRITICAL (Manual experiment, 75.38%):
       gamma~1.0, rho~0.99
       Fast oscillations match DVS temporal structure,
       contractive dynamics naturally fight overfitting

Sampling strategy:
  - 40% exploit regime A (around phase 1 best)
  - 40% exploit regime B (around new manual finding)
  - 20% wide exploration
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fixed settings
# ==============================

N_HID          = 3000
NUM_STEPS      = 200
SPATIAL_FACTOR = 4
READOUT_MODE   = "mean"
SEED           = 0
N_SAMPLES      = 150

SCRIPT                 = "dvs-gesture_spiking_ron.py"
RESULTS_DIR            = Path("hyperparam_search_DVSGesture_phase2")
DVSGESTURE_RESULTS_DIR = "results_dvsgesture_phase2"

RESULTS_DIR.mkdir(exist_ok=True)

# ==============================
# Search spaces
# ==============================

LOG_PARAMS = {"inp_scaling", "input_density", "theta_lif", "theta_rf",
              "epsilon", "gamma", "dt"}

# Regime A: SLOW + SUPERCRITICAL (phase 1 best)
# theta_rf pushed UP from 0.00108 to reduce HRF saturation (r_hrf was ~0.497)
REGIME_A = {
    "gamma":         (0.01,  0.15),
    "dt":            (0.15,  0.4),
    "epsilon":       (0.003, 0.08),
    "gamma_range":   (0.01,  0.15),
    "epsilon_range": (0.01,  0.10),
    "inp_scaling":   (0.01,  0.12),
    "input_density": (0.03,  0.15),
    "rho":           (1.0,   1.6),
    "theta_lif":     (0.3,   3.0),
    "theta_rf":      (0.005, 0.05),   # pushed up from 0.00108
}

# Regime B: FAST + SUBCRITICAL (manual experiment: gamma=1.0, rho=0.99 -> 75.38%)
REGIME_B = {
    "gamma":         (0.3,   3.0),
    "dt":            (0.05,  0.3),
    "epsilon":       (0.01,  0.5),
    "gamma_range":   (0.1,   1.5),
    "epsilon_range": (0.0,   0.3),
    "inp_scaling":   (0.01,  0.15),
    "input_density": (0.02,  0.15),
    "rho":           (0.85,  1.05),
    "theta_lif":     (0.1,   2.0),
    "theta_rf":      (0.001, 0.05),
}

# Wide exploration
WIDE_SPACE = {
    "gamma":         (0.003, 3.0),
    "dt":            (0.05,  0.4),
    "epsilon":       (0.003, 0.5),
    "gamma_range":   (0.003, 1.5),
    "epsilon_range": (0.0,   0.3),
    "inp_scaling":   (0.005, 0.2),
    "input_density": (0.01,  0.15),
    "rho":           (0.85,  1.6),
    "theta_lif":     (0.05,  3.0),
    "theta_rf":      (0.001, 0.1),
}

# C: push lower to fight overfitting (1077 train, 3000 features)
READOUT_C_VALUES  = [0.0001, 0.0003, 0.001, 0.003, 0.01]
READOUT_C_WEIGHTS = [0.10,   0.20,   0.35,  0.25,  0.10]

# ==============================
# Sampling
# ==============================

def sample_from_space(space):
    params = {}
    for key, (lo, hi) in space.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    return params


def sample_params(regime):
    space = {"A": REGIME_A, "B": REGIME_B, "wide": WIDE_SPACE}[regime]
    params = sample_from_space(space)
    params["readout_C"] = random.choices(READOUT_C_VALUES,
                                         weights=READOUT_C_WEIGHTS)[0]
    return params


n_A    = int(N_SAMPLES * 0.40)
n_B    = int(N_SAMPLES * 0.40)
n_wide = N_SAMPLES - n_A - n_B
regimes = (["A"] * n_A) + (["B"] * n_B) + (["wide"] * n_wide)
random.shuffle(regimes)

# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

ICONS = {"A": "🔵", "B": "🟠", "wide": "🌐"}

print(f"DVS Gesture Phase 2 (Dual Regime): {N_SAMPLES} configs — "
      f"{n_A}x Regime A / {n_B}x Regime B / {n_wide}x wide")
print(f"Fixed: n_hid={N_HID}, sf={SPATIAL_FACTOR}, "
      f"T={NUM_STEPS}, mode={READOUT_MODE}, seed={SEED}")
print("=" * 70)

for i, regime in enumerate(regimes):
    params = sample_params(regime)

    osc = np.sqrt(params["gamma"]) * params["dt"] * NUM_STEPS
    osc_flag = "" if 0.5 <= osc <= 15.0 else f" [osc={osc:.2f}!]"

    print(f"\n{ICONS[regime]} Config {i+1}/{N_SAMPLES} [Regime {regime}]: "
          f"C={params['readout_C']} "
          f"inp={params['inp_scaling']:.4f} dens={params['input_density']:.4f} "
          f"dt={params['dt']:.3f} "
          f"g={params['gamma']:.4f}+/-{params['gamma_range']:.4f} "
          f"e={params['epsilon']:.4f}+/-{params['epsilon_range']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} th_rf={params['theta_rf']:.5f}"
          f"{osc_flag}")

    cmd = [
        "python", SCRIPT,
        "--n_hid",                str(N_HID),
        "--spatial_factor",       str(SPATIAL_FACTOR),
        "--num_steps",            str(NUM_STEPS),
        "--dt",                   str(params["dt"]),
        "--gamma",                str(params["gamma"]),
        "--gamma_range",          str(params["gamma_range"]),
        "--epsilon",              str(params["epsilon"]),
        "--epsilon_range",        str(params["epsilon_range"]),
        "--rho",                  str(params["rho"]),
        "--inp_scaling",          str(params["inp_scaling"]),
        "--input_density",        str(params["input_density"]),
        "--theta_lif",            str(params["theta_lif"]),
        "--theta_rf",             str(params["theta_rf"]),
        "--connectivity_lif2hrf", "1.0",
        "--connectivity_hrf2lif", "1.0",
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--readout_C",            str(params["readout_C"]),
        "--readout_mode",         READOUT_MODE,
        "--results_dir",          DVSGESTURE_RESULTS_DIR,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=900)

        result_file = max(
            Path(DVSGESTURE_RESULTS_DIR).glob(f"*seed{SEED}.json"),
            key=lambda p: p.stat().st_mtime
        )
        with open(result_file) as f:
            res = json.load(f)

        res["config_id"]   = i
        res["search_seed"] = SEED
        res["readout_C"]   = params["readout_C"]
        res["regime"]      = regime
        all_results.append(res)

        gap = res["train_acc_mean"] - res["test_acc_mean"]
        sat = " ⚠️sat" if res["r_hrf_mean"] > 0.4 else ""
        print(f"   -> Test: {res['test_acc_mean']:.2f}%  "
              f"Train: {res['train_acc_mean']:.2f}%  "
              f"Gap: {gap:.1f}%  "
              f"r_hrf={res['r_hrf_mean']:.4f}{sat}  "
              f"r_lif={res['r_lif_mean']:.4f}")

    except subprocess.CalledProcessError as e:
        print(f"   FAILED")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-3:]:
                print(f"      {line}")
        failed_configs.append({"config_id": i, "params": params, "regime": regime})

    except subprocess.TimeoutExpired:
        print(f"   TIMEOUT (>900s)")
        failed_configs.append({"config_id": i, "params": params, "regime": regime})

    # Intermediate save every 10 configs
    if (i + 1) % 10 == 0 and all_results:
        top = sorted(all_results, key=lambda x: x["test_acc_mean"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(top[:20], f, indent=2)
        best = top[0]
        print(f"\n   [saved] best so far: {best['test_acc_mean']:.2f}% "
              f"[Regime {best.get('regime','?')}] "
              f"r_hrf={best['r_hrf_mean']:.4f}")

# ==============================
# Aggregate & save
# ==============================

all_results.sort(key=lambda x: x["test_acc_mean"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

# ==============================
# Print results
# ==============================

print("\n" + "=" * 70)
print(f"Completed: {len(all_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
for reg in ["A", "B", "wide"]:
    reg_res = [r for r in all_results if r.get("regime") == reg]
    if reg_res:
        accs = [r["test_acc_mean"] for r in reg_res]
        label = {"A": "Slow+Supercrit", "B": "Fast+Subcrit", "wide": "Exploration"}[reg]
        print(f"{ICONS[reg]} Regime {reg} ({label}): "
              f"best={max(accs):.2f}%  mean={np.mean(accs):.2f}%  n={len(reg_res)}")
print("=" * 70)

if all_results:
    print(f"\nTOP 10 (n_hid={N_HID}, sf={SPATIAL_FACTOR}, "
          f"mode={READOUT_MODE}, T={NUM_STEPS}):")
    print(f"{'Rk':<4} {'Test%':<14} {'Train%':<10} {'Gap':<7} {'Reg':<5} "
          f"{'C':<8} {'inp':<8} {'dens':<7} {'dt':<6} "
          f"{'gamma':<8} {'g_rng':<7} {'eps':<7} {'e_rng':<7} "
          f"{'rho':<6} {'th_lif':<8} {'th_rf':<8} "
          f"{'r_hrf':<7} {'r_lif'}")
    print("-" * 165)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        gap = r["train_acc_mean"] - r["test_acc_mean"]
        sat = "(*)" if r["r_hrf_mean"] > 0.4 else "   "
        print(f"{rank:<4} "
              f"{r['test_acc_mean']:.2f}+/-{r['std_test_acc']:.2f}    "
              f"{r['train_acc_mean']:.1f}%    "
              f"{gap:.1f}%   "
              f"{r.get('regime','?'):<5} "
              f"{float(r.get('readout_C',0)):<8}"
              f"{float(p.get('inp_scaling',0)):<8.4f}"
              f"{float(p.get('input_density',0)):<7.4f}"
              f"{float(p.get('dt',0)):<6.3f}"
              f"{float(p.get('gamma',0)):<8.4f}"
              f"{float(p.get('gamma_range',0)):<7.4f}"
              f"{float(p.get('epsilon',0)):<7.4f}"
              f"{float(p.get('epsilon_range',0)):<7.4f}"
              f"{float(p.get('rho',0)):<6.3f}"
              f"{float(p.get('theta_lif',0)):<8.4f}"
              f"{float(p.get('theta_rf',0)):<8.5f}"
              f"{sat}{r['r_hrf_mean']:<6.4f}  "
              f"{r['r_lif_mean']:.4f}")

    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "dt", "gamma", "gamma_range",
                  "epsilon", "epsilon_range", "rho", "theta_lif", "theta_rf"]:
        top_vals = [float(r.get("args", r).get(pname, 0)) for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(pname, 0)) for r in all_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.5f}+/-{np.std(top_vals):.5f}  "
              f"all={np.mean(all_vals):.5f}+/-{np.std(all_vals):.5f}")

    print(f"\nREGIME BREAKDOWN:")
    for reg in ["A", "B", "wide"]:
        reg_res = [r for r in all_results if r.get("regime") == reg]
        if reg_res:
            accs   = [r["test_acc_mean"] for r in reg_res]
            gaps   = [r["train_acc_mean"] - r["test_acc_mean"] for r in reg_res]
            r_hrfs = [r["r_hrf_mean"] for r in reg_res]
            label  = {"A": "Slow+Supercrit", "B": "Fast+Subcrit",
                      "wide": "Exploration"}[reg]
            print(f"  {ICONS[reg]} {label}: n={len(reg_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%  "
                  f"avg_r_hrf={np.mean(r_hrfs):.4f}")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in all_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < C_val * 0.01]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  avg_gap={np.mean(gaps):.1f}%")

    print(f"\nHRF SATURATION CHECK — top 10 (target r_hrf ~0.05-0.3):")
    for rank, r in enumerate(all_results[:10], 1):
        flag = " (*) SATURATED" if r["r_hrf_mean"] > 0.4 else ""
        print(f"  {rank}: r_hrf={r['r_hrf_mean']:.4f} r_lif={r['r_lif_mean']:.4f} "
              f"[{r.get('regime','?')}] test={r['test_acc_mean']:.2f}%{flag}")

else:
    print("\nNo successful configurations!")
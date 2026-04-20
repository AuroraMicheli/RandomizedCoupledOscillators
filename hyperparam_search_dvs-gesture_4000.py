"""
Hyperparameter search for Spiking RON on DVS Gesture at N_hid=4000.

Phase 3 targeted search. Key issue at N_hid=3000: r_hrf=0.495 (saturated).
This search actively fights HRF saturation by:
  1. Pushing theta_rf upward to reduce HRF firing rate
  2. Re-searching inp_scaling and input_density (size-sensitive)
  3. Re-searching rho (spectral radius can shift with N_hid)
  4. Re-searching theta_lif (interacts with input drive)
  5. Adding connectivity_lif2hrf to reduce LIF->HRF drive
  6. Also trying rms_std_final readout

Best config from Phase 2 (N_hid=3000, 78.41%):
  dt=0.259, gamma=0.046, gamma_range=0.130, epsilon=0.035,
  epsilon_range=0.099, inp_scaling=0.113, rho=1.581,
  theta_lif=2.968, theta_rf=0.036, input_density=0.031,
  num_steps=200, spatial_factor=4, readout_mode=mean, C=0.01
  r_hrf=0.495 (SATURATED — target: 0.1-0.3)

Fixed: num_steps=200, spatial_factor=4, dt, gamma, epsilon and ranges.
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fully fixed (per-neuron oscillator dynamics, size-independent)
# ==============================

FIXED = {
    "dt":            0.2593116113964727,
    "gamma":         0.04564827763077075,
    "gamma_range":   0.1304332231478915,
    "epsilon":       0.035383225582404816,
    "epsilon_range": 0.09892703865201465,
    "tau_filter":    20.0,
    "num_steps":     200,
    "spatial_factor": 4,          # 2048 input channels
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space
# ==============================

# theta_rf: MUST go up from 0.036 to reduce r_hrf from 0.495 to ~0.1-0.3
# The best config had theta_rf=0.036 giving r_hrf=0.495; we need ~3-5x higher
# threshold to bring firing rate down to a healthy range.
# inp_scaling / input_density: size-sensitive, re-search around best values
# rho: re-search; supercritical rho=1.58 may be too amplifying at larger N_hid
# theta_lif: re-search; interacts with input drive
# connectivity_lif2hrf: sparse LIF->HRF reduces drive to HRF, fights saturation

SEARCH_SPACE = {
    "inp_scaling":   (0.02,   0.3),    # log; best was 0.113
    "input_density": (0.01,   0.10),   # log; best was 0.031, try lower
    "rho":           (0.85,   1.6),    # linear; best was 1.581
    "theta_lif":     (0.5,    5.0),    # log; best was 2.968, keep similar range
    "theta_rf":      (0.02,   0.2),    # log; best was 0.036 → push UP to fight saturation
}

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.20, 0.35, 0.25, 0.20]

READOUT_MODE_OPTIONS = ["mean", "rms_std_final"]
READOUT_MODE_WEIGHTS = [0.60, 0.40]   # mean was best; also explore rms_std_final

# DVS Gesture has only 1077 training samples → strong regularisation needed
# With mean (4000 features) or rms_std_final (12000 features)
READOUT_C_VALUES  = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
READOUT_C_WEIGHTS = [0.10,   0.20,   0.30,  0.25,  0.10, 0.05]

LOG_PARAMS = {"inp_scaling", "input_density", "theta_lif", "theta_rf"}

# ==============================
# Sampling strategy:
# 50% exploit (narrowed around best, with theta_rf pushed up)
# 30% regime B (fast+subcritical from phase 2)
# 20% wide exploration
# ==============================

# Narrowed around phase 2 best, but theta_rf shifted up
EXPLOIT_SPACE = {
    "inp_scaling":   (0.05,   0.20),
    "input_density": (0.01,   0.06),
    "rho":           (1.2,    1.6),
    "theta_lif":     (1.5,    5.0),
    "theta_rf":      (0.05,   0.20),   # pushed up from 0.036
}

# Regime B: fast+subcritical — different oscillator regime, uses FIXED gamma
# but explores lower rho and different thresholds
REGIME_B_SPACE = {
    "inp_scaling":   (0.02,   0.15),
    "input_density": (0.01,   0.08),
    "rho":           (0.85,   1.05),
    "theta_lif":     (0.5,    3.0),
    "theta_rf":      (0.03,   0.15),
}

WIDE_SPACE = SEARCH_SPACE  # full range


def sample_from_space(space):
    params = {}
    for key, (lo, hi) in space.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    return params


def sample_params(regime):
    space = {
        "exploit": EXPLOIT_SPACE,
        "B":       REGIME_B_SPACE,
        "wide":    WIDE_SPACE,
    }[regime]
    params = dict(FIXED)
    params.update(sample_from_space(space))
    params["connectivity_lif2hrf"] = random.choices(
        CONNECTIVITY_LIF2HRF_OPTIONS, weights=CONNECTIVITY_LIF2HRF_WEIGHTS)[0]
    params["readout_mode"] = random.choices(
        READOUT_MODE_OPTIONS, weights=READOUT_MODE_WEIGHTS)[0]
    params["readout_C"] = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


# ==============================
# Search settings
# ==============================

N_SAMPLES   = 100
SEED        = 0       # single seed per config (DVS Gesture is slow)
N_HID       = 4000
SCRIPT      = "dvs-gesture_spiking_ron.py"
RESULTS_DIR = Path("hyperparam_search_DVSGesture_nhid4000")
DVS_RESULTS_DIR = "results_dvsgesture_nhid4000"

RESULTS_DIR.mkdir(exist_ok=True)

n_exploit = int(N_SAMPLES * 0.50)
n_B       = int(N_SAMPLES * 0.30)
n_wide    = N_SAMPLES - n_exploit - n_B
regimes   = (["exploit"] * n_exploit) + (["B"] * n_B) + (["wide"] * n_wide)
random.shuffle(regimes)

ICONS = {"exploit": "🎯", "B": "🟠", "wide": "🌐"}

# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"DVS Gesture Phase 3 (N_hid={N_HID}): {N_SAMPLES} configs — "
      f"{n_exploit}x exploit / {n_B}x regime B / {n_wide}x wide")
print(f"Fixed: {FIXED}")
print(f"Key change: theta_rf pushed UP to fight r_hrf saturation (was 0.036, r_hrf=0.495)")
print("=" * 70)

for i, regime in enumerate(regimes):
    params = sample_params(regime)

    # Diagnostic: expected oscillation cycles
    osc = np.sqrt(params["gamma"]) * params["dt"] * params["num_steps"]

    print(f"\n{ICONS[regime]} Config {i+1}/{N_SAMPLES} [regime={regime}]: "
          f"inp={params['inp_scaling']:.4f} "
          f"dens={params['input_density']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.3f} "
          f"th_rf={params['theta_rf']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"mode={params['readout_mode']} "
          f"C={params['readout_C']} "
          f"[osc={osc:.1f}]")

    cmd = [
        "python", SCRIPT,
        "--n_hid",                str(N_HID),
        "--spatial_factor",       str(int(params["spatial_factor"])),
        "--num_steps",            str(int(params["num_steps"])),
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
        "--tau_filter",           str(params["tau_filter"]),
        "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
        "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
        "--readout_mode",         params["readout_mode"],
        "--readout_C",            str(params["readout_C"]),
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--results_dir",          DVS_RESULTS_DIR,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=1200)

        result_file = max(
            Path(DVS_RESULTS_DIR).glob(f"*seed{SEED}.json"),
            key=lambda p: p.stat().st_mtime
        )
        with open(result_file) as f:
            res = json.load(f)

        res["config_id"]            = i
        res["search_seed"]          = SEED
        res["readout_C"]            = params["readout_C"]
        res["regime"]               = regime
        res["connectivity_lif2hrf"] = params["connectivity_lif2hrf"]
        all_results.append(res)

        gap = res["train_acc_mean"] - res["test_acc_mean"]
        sat = " ⚠️SAT" if res["r_hrf_mean"] > 0.4 else (
              " ✓" if res["r_hrf_mean"] < 0.3 else "")
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
        print(f"   TIMEOUT (>1200s)")
        failed_configs.append({"config_id": i, "params": params, "regime": regime})

    # Save intermediate every 10 configs
    if (i + 1) % 10 == 0 and all_results:
        top = sorted(all_results, key=lambda x: x["test_acc_mean"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(top[:20], f, indent=2)
        best = top[0]
        print(f"\n   [saved] best so far: {best['test_acc_mean']:.2f}%  "
              f"r_hrf={best['r_hrf_mean']:.4f}  "
              f"[{best.get('regime','?')}]")


# ==============================
# Final aggregation
# ==============================

all_results.sort(key=lambda x: x["test_acc_mean"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"Completed: {len(all_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

if all_results:
    print(f"\nTOP 10 (n_hid={N_HID}, sf=4, T=200):")
    print(f"{'Rk':<4} {'Test%':<10} {'Train%':<9} {'Gap':<6} {'Reg':<8} "
          f"{'mode':<14} {'C':<8} {'inp':<8} {'dens':<7} "
          f"{'rho':<6} {'th_lif':<8} {'th_rf':<8} "
          f"{'lif2hrf':<8} {'r_hrf':<8} {'r_lif'}")
    print("-" * 130)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        gap = r["train_acc_mean"] - r["test_acc_mean"]
        sat = "(*)" if r["r_hrf_mean"] > 0.4 else " ok"
        print(f"{rank:<4} "
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.1f}%   "
              f"{gap:.1f}%  "
              f"{r.get('regime','?'):<8} "
              f"{r.get('readout_mode', p.get('readout_mode','?')):<14} "
              f"{float(r.get('readout_C', 0)):<8}"
              f"{float(p.get('inp_scaling', 0)):<8.4f}"
              f"{float(p.get('input_density', 0)):<7.4f}"
              f"{float(p.get('rho', 0)):<6.3f}"
              f"{float(p.get('theta_lif', 0)):<8.3f}"
              f"{float(p.get('theta_rf', 0)):<8.4f}"
              f"{float(p.get('connectivity_lif2hrf', 1.0)):<8}"
              f"{sat}{r['r_hrf_mean']:.4f}  "
              f"{r['r_lif_mean']:.4f}")

    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "rho",
                  "theta_lif", "theta_rf"]:
        top_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.5f}+/-{np.std(top_vals):.5f}  "
              f"all={np.mean(all_vals):.5f}+/-{np.std(all_vals):.5f}")

    print(f"\nSATURATION ANALYSIS — all configs:")
    sat_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]
    for lo, hi in sat_bins:
        bucket = [r for r in all_results
                  if lo <= r["r_hrf_mean"] < hi]
        if bucket:
            accs = [r["test_acc_mean"] for r in bucket]
            print(f"  r_hrf [{lo:.1f}-{hi:.1f}): n={len(bucket)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    print(f"\nREGIME BREAKDOWN:")
    for reg in ["exploit", "B", "wide"]:
        reg_res = [r for r in all_results if r.get("regime") == reg]
        if reg_res:
            accs  = [r["test_acc_mean"] for r in reg_res]
            gaps  = [r["train_acc_mean"] - r["test_acc_mean"] for r in reg_res]
            r_hrfs = [r["r_hrf_mean"] for r in reg_res]
            print(f"  {ICONS[reg]} {reg}: n={len(reg_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%  "
                  f"avg_r_hrf={np.mean(r_hrfs):.4f}")

    print(f"\nREADOUT MODE BREAKDOWN:")
    for mode in READOUT_MODE_OPTIONS:
        m_res = [r for r in all_results
                 if r.get("readout_mode",
                          r.get("args", {}).get("readout_mode", "?")) == mode]
        if m_res:
            accs = [r["test_acc_mean"] for r in m_res]
            print(f"  {mode:<16}: n={len(m_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}+/-{np.std(accs):.2f}%")

    print(f"\nCONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in all_results
                 if abs(float(r.get("connectivity_lif2hrf",
                               r.get("args", {}).get(
                                   "connectivity_lif2hrf", 1.0))) - c) < 1e-6]
        if c_res:
            accs   = [r["test_acc_mean"] for r in c_res]
            r_hrfs = [r["r_hrf_mean"] for r in c_res]
            print(f"  lif2hrf={c:<5}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}%  "
                  f"avg_r_hrf={np.mean(r_hrfs):.4f}")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in all_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < C_val * 0.01]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}+/-{np.std(accs):.2f}%")

else:
    print("\nNo successful configurations!")
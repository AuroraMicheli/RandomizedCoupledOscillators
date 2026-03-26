import random
import json
import subprocess
import os
import time
from pathlib import Path
import numpy as np

# =============================
# npCIFAR-10 Hyperparameter Search — rms_std_final readout
# =============================
# This search is for the FIXED model with range-clipped gamma/epsilon.
# Key differences from the final readout search:
#
# 1. readout_mode is fixed to rms_std_final
# 2. gamma and epsilon centers are shifted higher so that after range
#    clipping to positive, the distribution still has meaningful spread.
#    e.g. gamma=0.5, gamma_range=0.8 → effective range [0.1, 0.9] (good)
#         gamma=0.02, gamma_range=0.8 → effective range [1e-6, 0.42] (bad,
#         half the intended range is lost)
# 3. C weights are uniform — no prior from a previous rms_std_final search
# 4. Wide space allows gamma/epsilon up to higher values since positive-only
#    sampling means we need higher centers to get heterogeneity
# =============================

BEST = {
    "gamma":         0.5,
    "dt":            0.0377,
    "epsilon":       0.4,
    "gamma_range":   0.8,
    "epsilon_range": 0.3,
    "inp_scaling":   0.0138,
    "rho":           0.518,
    "theta_lif":     0.030,
    "theta_rf":      0.209,
    "tau_filter":    5.61,
}

NARROW_SPACE = {
    "gamma":         (0.3,    1.5),
    "dt":            (0.01,   0.15),
    "epsilon":       (0.2,    1.5),
    "gamma_range":   (0.2,    1.2),
    "epsilon_range": (0.05,   0.5),
    "inp_scaling":   (0.003,  0.06),
    "rho":           (0.4,    0.95),
    "theta_lif":     (0.008,  0.1),
    "theta_rf":      (0.05,   0.5),
    "tau_filter":    (3.0,    30.0),
}

WIDE_SPACE = {
    "gamma":         (0.1,    3.0),
    "dt":            (0.005,  0.2),
    "epsilon":       (0.05,   2.0),
    "gamma_range":   (0.1,    2.0),
    "epsilon_range": (0.0,    1.0),
    "inp_scaling":   (0.001,  0.5),
    "rho":           (0.3,    1.0),
    "theta_lif":     (0.005,  1.0),
    "theta_rf":      (0.001,  0.5),
    "tau_filter":    (1.0,    50.0),
}

LOG_PARAMS = {"inp_scaling", "theta_lif", "theta_rf", "epsilon",
              "gamma", "dt", "tau_filter"}

READOUT_C_VALUES  = [0.001, 0.01, 0.1, 1.0, 10.0]
# No prior — we're seeing consistent overfitting so weight toward smaller C
READOUT_C_WEIGHTS = [0.15, 0.40, 0.30, 0.10, 0.05]

N_SAMPLES      = 150
EXPLOIT_FRAC   = 0.80
SEED           = 42
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_npCIFAR10_rms_std_final")
SCRIPT_NAME    = "npCIFAR10_spiking_ron.py"
RESULTS_SUBDIR = "results_npcifar10"

RESULTS_DIR.mkdir(exist_ok=True)


def cleanup_shm():
    """Remove leaked joblib/loky shared memory segments between runs."""
    subprocess.run(
        ["bash", "-c",
         "ls /dev/shm/joblib_memmapping_folder_* 2>/dev/null | xargs rm -rf; "
         "ls /dev/shm/loky-* 2>/dev/null | xargs rm -rf"],
        check=False
    )


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


# Build a clean env that redirects joblib's temp folder away from /dev/shm
# and caps the number of loky workers to avoid over-spawning.
child_env = os.environ.copy()
child_env["JOBLIB_TEMP_FOLDER"] = "/tmp"       # avoid /dev/shm exhaustion
child_env["LOKY_MAX_CPU_COUNT"] = "2"          # cap worker count per subprocess


all_results    = []
failed_configs = []

random.seed(SEED)

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)

print(f"npCIFAR-10 rms_std_final search: {N_SAMPLES} configs "
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
        "--connectivity_lif2hrf", "0.2",
        "--connectivity_hrf2lif", "1.0",
        "--readout_mode",         "rms_std_final",
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--readout_C",            str(params["readout_C"]),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env=child_env,          # <-- redirects joblib temp + caps workers
        )

        result_file = max(
            Path(RESULTS_SUBDIR).glob(f"*rms_std_final*seed{SEED}.json"),
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

    finally:
        # Always clean up leaked shm segments, whether the run succeeded or not
        cleanup_shm()
        time.sleep(2)   # give the OS a moment to reclaim resources

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
exploit_results  = [r for r in all_results if r.get("sample_type") == "exploit"]
explore_results  = [r for r in all_results if r.get("sample_type") == "explore"]
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
            print(
                f"  C={C_val:<7}: n={len(c_results)}, "
                f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                f"gap={np.mean(c_gaps):.1f}%"
            )

'''
#NPCIFAR10 SCRIPT FOR READOUT_MODE = FINAL

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# npCIFAR-10 Hyperparameter Search — Phase 2
# =============================
# Phase 1 best config (config_id=39, 31.3% test):
#   dt=0.0377, gamma=0.0221, epsilon=0.244, gamma_range=0.806,
#   epsilon_range=0.302, inp_scaling=0.0138, rho=0.518,
#   theta_lif=0.0301, theta_rf=0.209, tau_filter=5.61, C=0.01
#
# Strategy:
#   - 80% of samples: narrowed search around best config (exploit)
#   - 20% of samples: wider exploration in case global optimum is elsewhere
#   - 1 seed only (seed=0), 150 configs → 3x more coverage than phase 1
# =============================

# --- Phase 1 best config (reference point for narrowed search) ---
BEST = {
    "gamma":         0.022124825018135126,
    "dt":            0.03767457087824196,
    "epsilon":       0.24371246704641245,
    "gamma_range":   0.8057386186915503,
    "epsilon_range": 0.30215568102010293,
    "inp_scaling":   0.013843835724517757,
    "rho":           0.5180381501925403,
    "theta_lif":     0.03014544609189747,
    "theta_rf":      0.2085508689716086,
    "tau_filter":    5.6099550220268295,
}

# --- Narrowed search space (±1.5 decades for log params, ±40% for linear) ---
NARROW_SPACE = {
    "gamma":         (0.005,  0.15),      # log: best=0.022, explore 0.005-0.15
    "dt":            (0.01,   0.15),      # log: best=0.038
    "epsilon":       (0.05,   1.5),       # log: best=0.244
    "gamma_range":   (0.2,    1.8),       # linear: best=0.806
    "epsilon_range": (0.05,   0.7),       # linear: best=0.302
    "inp_scaling":   (0.003,  0.06),      # log: best=0.014
    "rho":           (0.4,    0.9),       # linear: best=0.518
    "theta_lif":     (0.008,  0.1),       # log: best=0.030
    "theta_rf":      (0.05,   0.5),       # log: best=0.209
    "tau_filter":    (3.0,    30.0),      # log: best=5.61
}

# --- Wide exploration space (same as phase 1) ---
WIDE_SPACE = {
    "gamma":         (0.01,  2.0),
    "dt":            (0.01,  0.2),
    "epsilon":       (0.01,  2.0),
    "gamma_range":   (0.1,   2.0),
    "epsilon_range": (0.0,   1.0),
    "inp_scaling":   (0.001, 0.5),
    "rho":           (0.5,   1.5),
    "theta_lif":     (0.01,  1.0),
    "theta_rf":      (0.001, 0.5),
    "tau_filter":    (5.0,   100.0),
}

LOG_PARAMS = {"inp_scaling", "theta_lif", "theta_rf", "epsilon",
              "gamma", "dt", "tau_filter"}

READOUT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
# Phase 1 shows C=0.01 wins → weight it more heavily
READOUT_C_WEIGHTS = [0.05, 0.50, 0.30, 0.10, 0.05]

N_SAMPLES      = 150
EXPLOIT_FRAC   = 0.80          # 80% narrowed, 20% wide exploration
SEED           = 0             # single seed for more config coverage
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_npCIFAR10_phase2")
SCRIPT_NAME    = "npCIFAR10_spiking_ron.py"
RESULTS_SUBDIR = "results_npcifar10"

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
    # Weighted C sampling — phase 1 shows C=0.01 dominates
    params["readout_C"] = random.choices(READOUT_C_VALUES,
                                         weights=READOUT_C_WEIGHTS)[0]
    return params


all_results    = []
failed_configs = []

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)

print(f"Phase 2 search: {N_SAMPLES} configs "
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
        "--connectivity_lif2hrf", "0.2",
        "--connectivity_hrf2lif", "1.0",
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
            Path(RESULTS_SUBDIR).glob(f"*seed{SEED}.json"),
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

    # Save intermediate results every 10 configs
    if (i + 1) % 10 == 0:
        intermediate = sorted(all_results,
                               key=lambda x: x["test_acc_mean"],
                               reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
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
exploit_results  = [r for r in all_results if r.get("sample_type") == "exploit"]
explore_results  = [r for r in all_results if r.get("sample_type") == "explore"]
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

'''
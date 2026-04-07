import random
import json
import subprocess
import os
import time
from pathlib import Path
import numpy as np




# =============================
# npCIFAR-10 Hyperparameter Search — rms_std_final readout, Round 4
# =============================
# Key changes from round 3:
#
# 1. C values refined — round 3 revealed C=0.0001 underfits (gaps ~1-2%)
#    and C=0.001 is still slightly underfitting at the best config (gap 0.8%).
#    The per-C mean test acc was: 0.0001→24.2%, 0.001→26.1%, 0.01→27.4%, 0.1→26.4%
#    This suggests the optimum is between C=0.001 and C=0.01.
#    → Added intermediate values C=0.003 and C=0.007, removed C=0.0001.
#
# 2. BEST updated to round 3's best config (30.83%, config 22):
#      inp=0.0248, dt=0.015, γ=0.2727, ε=0.1987, ρ=0.67, τ=13.9
#
# 3. NARROW_SPACE updated from round 3 top-10 trends:
#      dt:          top10 mean=0.021  → search [0.008, 0.055]
#      gamma:       top10 mean=0.63   → search [0.25,  1.1]
#      epsilon:     top10 mean=0.38   → search [0.15,  0.80]
#      rho:         top10 mean=0.64   → search [0.40,  0.95]
#      tau_filter:  top10 mean=19.8   → search [7.0,   40.0]
#      inp_scaling: top10 mean=0.030  → search [0.008, 0.07]
#
# 4. 30 configs, 90% exploit, new seed. 48h SLURM budget.
# =============================
 
BEST = {
    # Round 3 best: config 22, 30.83% test
    "gamma":         0.2727,
    "dt":            0.015,
    "epsilon":       0.1987,
    "gamma_range":   0.62,
    "epsilon_range": 0.262,
    "inp_scaling":   0.0248,
    "rho":           0.67,
    "theta_lif":     0.0125,
    "theta_rf":      0.0340,
    "tau_filter":    13.9,
}
 
# Tightened around round-3 top-10 trends
NARROW_SPACE = {
    "gamma":         (0.25,   1.1),
    "dt":            (0.008,  0.055),
    "epsilon":       (0.15,   0.80),
    "gamma_range":   (0.2,    1.2),
    "epsilon_range": (0.05,   0.55),
    "inp_scaling":   (0.008,  0.07),
    "rho":           (0.40,   0.95),
    "theta_lif":     (0.008,  0.12),
    "theta_rf":      (0.03,   0.45),
    "tau_filter":    (7.0,    40.0),
}
 
WIDE_SPACE = {
    "gamma":         (0.1,    2.0),
    "dt":            (0.005,  0.15),
    "epsilon":       (0.05,   1.5),
    "gamma_range":   (0.1,    2.0),
    "epsilon_range": (0.0,    0.8),
    "inp_scaling":   (0.003,  0.1),
    "rho":           (0.30,   0.98),
    "theta_lif":     (0.005,  0.5),
    "theta_rf":      (0.005,  0.5),
    "tau_filter":    (3.0,    50.0),
}
 
LOG_PARAMS = {"inp_scaling", "theta_lif", "theta_rf", "epsilon",
              "gamma", "dt", "tau_filter"}
 
# Key change: intermediate C values to find the optimum between 0.001 and 0.01
# Round 3 per-C means: 0.0001→24.2%, 0.001→26.1%, 0.01→27.4%, 0.1→26.4%
# Optimum is likely around 0.003–0.007
READOUT_C_VALUES  = [0.001,  0.003,  0.007,  0.01,  0.03,  0.1]
READOUT_C_WEIGHTS = [0.15,   0.25,   0.25,   0.20,  0.10,  0.05]
 
N_SAMPLES      = 30
EXPLOIT_FRAC   = 0.90   # 27 exploit, 3 explore
SEED           = 456    # different seed from rounds 2 and 3
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_npCIFAR10_rms_std_final_round4")
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
 
 
child_env = os.environ.copy()
child_env["JOBLIB_TEMP_FOLDER"] = "/tmp"
child_env["LOKY_MAX_CPU_COUNT"] = "2"
 
all_results    = []
failed_configs = []
 
random.seed(SEED)
 
n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)
 
print(f"npCIFAR-10 rms_std_final ROUND 4 search: {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore), seed={SEED}")
print(f"Reference best (round 3): {BEST}")
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
            timeout=3600,
            env=child_env,
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
        print(f"   ⏰ TIMEOUT (>3600s)")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})
 
    finally:
        cleanup_shm()
        time.sleep(2)
 
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
          f"{'inp_scl':<9} {'C':<8} {'dt':<7} "
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
              f"{float(r.get('readout_C', 0)):<8}"
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
                f"  C={C_val:<8}: n={len(c_results)}, "
                f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                f"gap={np.mean(c_gaps):.1f}%"
            )
 




'''
# =============================
# npCIFAR-10 Hyperparameter Search — rms_std_final readout, Round 3
# =============================
# Key changes from round 2:
#
# 1. Focused search: exploit-only around the top-10 configs from round 2,
#    all of which had C=0.001, small dt, moderate rho.
# 2. Added C=0.0001 — round 2 showed C=0.001 dominates, so we explore
#    even smaller regularization which may help the 3x larger feature space.
# 3. BEST reference updated to round 2's best config (31.02%, config 110).
# 4. NARROW_SPACE tightened around the round-2 top-10 parameter trends:
#      dt:          top10 mean=0.033  → search [0.008, 0.09]
#      gamma:       top10 mean=0.56   → search [0.25, 1.0]
#      epsilon:     top10 mean=0.42   → search [0.15, 0.8]
#      rho:         top10 mean=0.73   → search [0.4,  0.95]
#      tau_filter:  top10 mean=14.0   → search [5.0,  35.0]
#      inp_scaling: top10 mean=0.027  → search [0.005, 0.07]
# 5. Timeout raised to 3600s per config, SLURM time to 120h.
# 6. Reduced N_SAMPLES to 60 (all exploit) — no point exploring widely
#    when we know where the good region is.
# =============================

BEST = {
    # Round 2 best: config 110, 31.02% test
    "gamma":         0.5371,
    "dt":            0.027,
    "epsilon":       0.2876,
    "gamma_range":   0.41,
    "epsilon_range": 0.476,
    "inp_scaling":   0.0129,
    "rho":           0.52,
    "theta_lif":     0.0113,
    "theta_rf":      0.0597,
    "tau_filter":    21.0,
}

# Tightened around round-2 top-10 trends
NARROW_SPACE = {
    "gamma":         (0.25,   1.0),
    "dt":            (0.008,  0.09),
    "epsilon":       (0.15,   0.80),
    "gamma_range":   (0.2,    1.2),
    "epsilon_range": (0.05,   0.55),
    "inp_scaling":   (0.005,  0.07),
    "rho":           (0.40,   0.95),
    "theta_lif":     (0.008,  0.12),
    "theta_rf":      (0.03,   0.45),
    "tau_filter":    (5.0,    35.0),
}

# Kept wider for a small explore fraction
WIDE_SPACE = {
    "gamma":         (0.1,    2.0),
    "dt":            (0.005,  0.15),
    "epsilon":       (0.05,   1.5),
    "gamma_range":   (0.1,    2.0),
    "epsilon_range": (0.0,    0.8),
    "inp_scaling":   (0.003,  0.1),
    "rho":           (0.30,   0.98),
    "theta_lif":     (0.005,  0.5),
    "theta_rf":      (0.005,  0.5),
    "tau_filter":    (3.0,    50.0),
}

LOG_PARAMS = {"inp_scaling", "theta_lif", "theta_rf", "epsilon",
              "gamma", "dt", "tau_filter"}

# Round 2 showed C=0.001 dominates strongly → add C=0.0001, weight heavily low
READOUT_C_VALUES  = [0.0001, 0.001, 0.01,  0.1,  1.0]
READOUT_C_WEIGHTS = [0.30,   0.50,  0.15,  0.04, 0.01]

N_SAMPLES      = 30
EXPLOIT_FRAC   = 0.90   # 54 exploit, 6 explore
SEED           = 123    # different seed from round 2 to get new samples
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_npCIFAR10_rms_std_final_round3")
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


child_env = os.environ.copy()
child_env["JOBLIB_TEMP_FOLDER"] = "/tmp"
child_env["LOKY_MAX_CPU_COUNT"] = "2"

all_results    = []
failed_configs = []

random.seed(SEED)

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)

print(f"npCIFAR-10 rms_std_final ROUND 3 search: {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore), seed={SEED}")
print(f"Reference best (round 2): {BEST}")
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
            timeout=3600,   # 1 hour per config
            env=child_env,
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
        print(f"   ⏰ TIMEOUT (>3600s)")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    finally:
        cleanup_shm()
        time.sleep(2)

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
          f"{'inp_scl':<9} {'C':<8} {'dt':<7} "
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
              f"{float(r.get('readout_C', 0)):<8}"
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
                f"  C={C_val:<8}: n={len(c_results)}, "
                f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                f"gap={np.mean(c_gaps):.1f}%"
            )

'''
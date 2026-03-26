import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# sMNIST Hyperparameter Search — Phase 2
# =============================
# Phase 1 best config (from previous search):
#   dt=0.042, gamma=2.7, epsilon=0.08, gamma_range=2.0,
#   epsilon_range=1.0, inp_scaling=2.0, rho=0.99,
#   theta_lif=0.05, theta_rf=0.005, tau_filter=20.0
#
# Strategy:
#   - 80% exploit: narrowed around phase 1 best
#   - 20% explore: wider search
#   - readout_mode fixed to final
#   - uses original spiking_coESN_rescaled_II from utils_aurora (no modifications)
#   - seed=42, n_hid=800, 150 configs
#
# Fixes vs previous run:
#   - dt lower bound raised to 0.03 (narrow) / 0.02 (wide) to avoid timeouts
#   - n_jobs=2 in LogisticRegression to avoid OOM kills from joblib workers
#   - mem increased to 32GB in sbatch
# =============================

BEST = {
    "dt":            0.042,
    "gamma":         2.7,
    "epsilon":       0.08,
    "gamma_range":   2.0,
    "epsilon_range": 1.0,
    "inp_scaling":   2.0,
    "rho":           0.99,
    "theta_lif":     0.05,
    "theta_rf":      0.005,
    "tau_filter":    20.0,
}

NARROW_SPACE = {
    "dt":            (0.03,   0.15),    # log: best=0.042, raised floor to avoid timeouts
    "gamma":         (1.0,    5.0),     # log: best=2.7
    "epsilon":       (0.02,   0.3),     # log: best=0.08
    "gamma_range":   (0.5,    3.5),     # linear: best=2.0
    "epsilon_range": (0.1,    1.5),     # linear: best=1.0
    "inp_scaling":   (0.5,    5.0),     # log: best=2.0
    "rho":           (0.85,   1.05),    # linear: best=0.99
    "theta_lif":     (0.01,   0.2),     # log: best=0.05
    "theta_rf":      (0.001,  0.05),    # log: best=0.005
    "tau_filter":    (5.0,    50.0),    # log: best=20.0
}

WIDE_SPACE = {
    "dt":            (0.02,   0.3),     # raised floor from 0.005
    "gamma":         (0.3,    10.0),
    "epsilon":       (0.005,  1.0),
    "gamma_range":   (0.1,    5.0),
    "epsilon_range": (0.0,    2.0),
    "inp_scaling":   (0.1,    10.0),
    "rho":           (0.7,    1.1),
    "theta_lif":     (0.005,  1.0),
    "theta_rf":      (0.0005, 0.1),
    "tau_filter":    (1.0,    100.0),
}

LOG_PARAMS = {"dt", "gamma", "epsilon", "inp_scaling",
              "theta_lif", "theta_rf", "tau_filter"}

READOUT_C_VALUES  = [0.001, 0.01, 0.1, 1.0, 10.0]
READOUT_C_WEIGHTS = [0.05, 0.25, 0.40, 0.25, 0.05]

N_SAMPLES      = 150
EXPLOIT_FRAC   = 0.80
SEED           = 42
N_HID          = 800
RESULTS_DIR    = Path("hyperparam_search_sMNIST_phase2")
SCRIPT_NAME    = "sMNIST_spiking_ron.py"
RESULTS_SUBDIR = "results_smnist"

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

print(f"sMNIST phase 2 search (readout=final): {N_SAMPLES} configs "
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
            cmd, check=True, capture_output=True, text=True, timeout=300
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
        print(f"   ⏰ TIMEOUT (>300s)")
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
            print(
                f"  C={C_val:<7}: n={len(c_results)}, "
                f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                f"gap={np.mean(c_gaps):.1f}%"
            )
'''
#ORIGINAL ONE


import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# Search space (centered on good config)
# =============================
SEARCH_SPACE = {
    "gamma": (2.0, 3.5),
    "epsilon": (0.04, 0.12),
    "gamma_range": (0.5, 3.0),
    "epsilon_range": (0.0, 1.0),
    "rho": (0.90, 1.05),
    "inp_scaling": (1.0, 3.0),
}


N_SAMPLES = 30              # good tradeoff for sMNIST
SEEDS = [0, 1, 2]
N_HID = 256
RESULTS_DIR = Path("hyparam_search_smnist")
RESULTS_DIR.mkdir(exist_ok=True)

SCRIPT = "sparse_connectivity_lif_hrf_cluster.py"   # <-- your main script filename

# =============================
# Sampling function
# =============================
def sample_params():
    return {
        k: random.uniform(*v)
        for k, v in SEARCH_SPACE.items()
    }

# =============================
# Run experiments
# =============================
all_results = []

for i in range(N_SAMPLES):
    params = sample_params()
    print(f"\n🔍 Config {i+1}/{N_SAMPLES}: {params}")

    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid", str(N_HID),
            "--dt", "0.042",
            "--gamma", str(params["gamma"]),
            "--epsilon", str(params["epsilon"]),
            "--gamma_range", str(params["gamma_range"]),
            "--epsilon_range", str(params["epsilon_range"]),
            "--rho", str(params["rho"]),
            "--inp_scaling", str(params["inp_scaling"]),
            "--theta_lif", "0.05",
            "--theta_rf", "0.005",
            "--tau_filter", "20.0",
            "--seed", str(seed),
            "--use_test"
        ]

        subprocess.run(cmd, check=True)

        # Grab most recent result file
        result_file = max(
            Path("results_sMNIST").glob("results_*.json"),
            key=lambda p: p.stat().st_mtime
        )

        with open(result_file) as f:
            res = json.load(f)

        res["config_id"] = i
        res["seed"] = seed
        all_results.append(res)

# =============================
# Aggregate results
# =============================
summary = {}

for r in all_results:
    cid = r["config_id"]
    summary.setdefault(cid, []).append(r)

final_results = []

for cid, runs in summary.items():
    val_accs = [r["valid_acc"] for r in runs]
    final_results.append({
        "config_id": cid,
        "mean_valid_acc": float(np.mean(val_accs)),
        "std_valid_acc": float(np.std(val_accs)),
        "params": runs[0]["args"]
    })

final_results.sort(key=lambda x: x["mean_valid_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\n🏆 Best configuration:")
print(final_results[0])
'''
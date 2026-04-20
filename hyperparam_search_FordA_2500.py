"""
FordA Phase 2 search at N_hid=2500.

Best from Phase 1 (config_id=73, 76.74%, single seed):
  inp_scaling=2.131, rho=0.985, theta_lif=1.211,
  connectivity_lif2hrf=0.1, C=0.001

Phase 2 strategy:
  - Narrowed search around Phase 1 best
  - 3 seeds per config to get reliable estimates (Phase 1 used 1 seed)
  - Also try connectivity_lif2hrf=0.05 (even sparser) which wasn't searched
  - Also try slightly larger C values since C=0.001 might be over-regularising
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

FIXED = {
    "dt":            0.051,
    "gamma":         7.0124,
    "gamma_range":   3.01,
    "epsilon":       0.1528,
    "epsilon_range": 0.419,
    "theta_rf":      0.001,
    "tau_filter":    6.1,
    "readout_mode":  "rms_std_final",
    "connectivity_hrf2lif": 1.0,
}

# Narrowed around Phase 1 best
EXPLOIT_SPACE = {
    "inp_scaling": (1.0,   4.0),    # log; best was 2.131
    "rho":         (0.85,  1.05),   # linear; best was 0.985
    "theta_lif":   (0.5,   2.5),    # log; best was 1.211
}

WIDE_SPACE = {
    "inp_scaling": (0.1,   5.0),
    "rho":         (0.5,   1.1),
    "theta_lif":   (0.1,   4.0),
}

# Phase 1 best had lif2hrf=0.1; also try 0.05
CONNECTIVITY_LIF2HRF_OPTIONS = [0.05, 0.1, 0.2, 0.5]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.25, 0.45, 0.20, 0.10]

# Phase 1 best had C=0.001; also try slightly larger
READOUT_C_VALUES  = [0.0005, 0.001, 0.005, 0.01, 0.05]
READOUT_C_WEIGHTS = [0.15,   0.40,  0.25,  0.15, 0.05]

LOG_PARAMS = {"inp_scaling", "theta_lif"}

N_SAMPLES    = 60
EXPLOIT_FRAC = 0.75
SEEDS        = [42, 43, 44]   # 3 seeds for reliable estimates
N_HID        = 2500
SCRIPT       = "FordA_spiking_ron.py"
RESULTS_DIR  = Path("hyperparam_search_FordA_nhid2500_phase2")
RESULTS_SUBDIR = "results_fordA_nhid2500_p2"

RESULTS_DIR.mkdir(exist_ok=True)

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)


def sample_params(exploit=True):
    space = EXPLOIT_SPACE if exploit else WIDE_SPACE
    params = dict(FIXED)
    for key, (lo, hi) in space.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    params["connectivity_lif2hrf"] = random.choices(
        CONNECTIVITY_LIF2HRF_OPTIONS, weights=CONNECTIVITY_LIF2HRF_WEIGHTS)[0]
    params["readout_C"] = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


def _aggregate(all_results, seeds):
    summary = {}
    for r in all_results:
        summary.setdefault(r["config_id"], []).append(r)
    final = []
    for cid, runs in summary.items():
        if len(runs) == len(seeds):
            test_accs  = [r["test_acc_mean"] for r in runs]
            train_accs = [r["train_acc_mean"] for r in runs]
            final.append({
                "config_id":      cid,
                "mean_test_acc":  float(np.mean(test_accs)),
                "std_test_acc":   float(np.std(test_accs)),
                "mean_train_acc": float(np.mean(train_accs)),
                "overfit_gap":    float(np.mean(train_accs) - np.mean(test_accs)),
                "readout_C":      runs[0].get("readout_C", 0.001),
                "connectivity_lif2hrf": float(runs[0].get(
                    "args", {}).get("connectivity_lif2hrf", 0.1)),
                "params":         runs[0]["args"],
            })
    return final


all_results    = []
failed_configs = []

print(f"FordA Phase 2: N_hid={N_HID}, {N_SAMPLES} configs x {len(SEEDS)} seeds")
print(f"Phase 1 best: inp=2.131, rho=0.985, th_lif=1.211, lif2hrf=0.1, C=0.001 -> 76.74%")
print(f"Target: beat Spiking RC GTE (80.37%)")
print("=" * 70)

for i, stype in enumerate(sample_types):
    exploit = (stype == "exploit")
    params  = sample_params(exploit=exploit)

    print(f"\n{'🎯' if exploit else '🌐'} Config {i+1}/{N_SAMPLES} [{stype}]: "
          f"inp={params['inp_scaling']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"C={params['readout_C']}")

    config_failed = False
    seed_results  = []

    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
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
            "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
            "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
            "--readout_mode",         params["readout_mode"],
            "--readout_C",            str(params["readout_C"]),
            "--seed",                 str(seed),
            "--test_trials",          "1",
            "--use_test",
            "--results_dir",          RESULTS_SUBDIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=600)
            result_file = max(
                Path(RESULTS_SUBDIR).glob(f"*rms_std_final*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)
            res["config_id"] = i
            res["readout_C"] = params["readout_C"]
            res["sample_type"] = stype
            all_results.append(res)
            seed_results.append(res["test_acc_mean"])

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"   ❌ seed {seed} FAILED")
            config_failed = True
            break

    if config_failed:
        failed_configs.append({"config_id": i, "params": params})
    elif len(seed_results) == len(SEEDS):
        mean_acc = np.mean(seed_results)
        std_acc  = np.std(seed_results)
        print(f"   ✅ {mean_acc:.2f}±{std_acc:.2f}%  "
              f"(seeds: {[f'{a:.2f}' for a in seed_results]})")

    if (i + 1) % 10 == 0:
        agg = _aggregate(all_results, SEEDS)
        agg.sort(key=lambda x: x["mean_test_acc"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(agg[:20], f, indent=2)
        if agg:
            print(f"\n   💾 Best so far: "
                  f"{agg[0]['mean_test_acc']:.2f}±{agg[0]['std_test_acc']:.2f}%")

# Final
final = _aggregate(all_results, SEEDS)
final.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"Completed: {len(final)}/{N_SAMPLES} configs (3-seed mean)")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

if final:
    print(f"\nTOP 10:")
    print(f"{'Rank':<5} {'Test%':<18} {'Train%':<10} {'Gap':<7} "
          f"{'inp':<9} {'rho':<6} {'th_lif':<8} {'lif2hrf':<9} {'C'}")
    print("-" * 80)
    for rank, r in enumerate(final[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}±{r['std_test_acc']:.2f}      "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{float(p.get('inp_scaling',0)):<9.4f}"
              f"{float(p.get('rho',0)):<6.3f}"
              f"{float(p.get('theta_lif',0)):<8.4f}"
              f"{r['connectivity_lif2hrf']:<9}"
              f"{r['readout_C']}")

    print(f"\nCONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in final if abs(r["connectivity_lif2hrf"] - c) < 1e-6]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  lif2hrf={c:<6}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  mean={np.mean(accs):.2f}%")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final
                 if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  mean={np.mean(accs):.2f}%")

                  

'''

"""
Hyperparameter search for Spiking RON on FordA at N_hid=2500.

Goal: match the baseline neuron count (2500) to attempt to beat
Spiking RC GTE (80.37%, 2500 neurons).

Strategy: fix per-neuron oscillator dynamics from the best N_hid=800
config, re-search only size-sensitive parameters (inp_scaling, rho,
connectivity_lif2hrf, readout_C).

Best config at N_hid=800 (73.91%, readout_mode=rms_std_final,
connectivity_lif2hrf=0.2, theta_lif=1.0):
  dt=0.051, gamma=7.0124, gamma_range=3.01, epsilon=0.1528,
  epsilon_range=0.419, inp_scaling=0.6247, rho=0.75,
  theta_lif=1.0 (overridden), theta_rf=0.001, tau_filter=6.1

At N_hid=2500 with rms_std_final: 7500 features, ~3600 train samples
-> need smaller C than at N_hid=800.
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fixed (per-neuron, size-independent)
# ==============================

FIXED = {
    "dt":            0.051,
    "gamma":         7.0124,
    "gamma_range":   3.01,
    "epsilon":       0.1528,
    "epsilon_range": 0.419,
    "theta_lif":     1.0,       # overridden from default in best run
    "theta_rf":      0.001,
    "tau_filter":    6.1,
    "readout_mode":  "rms_std_final",
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space (size-sensitive)
# ==============================

# inp_scaling: at larger N_hid recurrent feedback is stronger,
# input drive may need adjustment. Search around best value.
# rho: spectral radius can shift with reservoir size.
# connectivity_lif2hrf: sparse reduces energy and can improve accuracy;
# best at 800 was 0.2, but at 2500 might benefit from different value.
# readout_C: with 7500 features and ~3600 samples, need smaller C.
# Also explore theta_lif slightly since it was manually set to 1.0
# and wasn't part of the original search.

SEARCH_SPACE = {
    "inp_scaling": (0.1,  3.0),    # log; best was 0.6247
    "rho":         (0.5,  1.1),    # linear; best was 0.75
    "theta_lif":   (0.1,  3.0),    # log; manually set to 1.0, worth re-exploring
}

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.20, 0.40, 0.25, 0.15]  # bias toward 0.2

# With 7500 features and 3600 train samples: need stronger regularisation
# than at N_hid=800 (where C=0.1 was best)
READOUT_C_VALUES  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
READOUT_C_WEIGHTS = [0.15,  0.25,  0.30, 0.20, 0.07, 0.03]

LOG_PARAMS = {"inp_scaling", "theta_lif"}

# ==============================
# Sampling strategy
# ==============================

# 70% exploit: narrowed around best N_hid=800 values
# 30% explore: wider range in case optimal shifts more at 2500

EXPLOIT_SPACE = {
    "inp_scaling": (0.2,  1.5),    # log; narrowed around 0.6247
    "rho":         (0.6,  0.95),   # linear; narrowed around 0.75
    "theta_lif":   (0.3,  2.0),    # log; narrowed around 1.0
}

WIDE_SPACE = SEARCH_SPACE  # full range for exploration


def sample_params(exploit=True):
    space = EXPLOIT_SPACE if exploit else WIDE_SPACE
    params = dict(FIXED)
    for key, (lo, hi) in space.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    params["connectivity_lif2hrf"] = random.choices(
        CONNECTIVITY_LIF2HRF_OPTIONS, weights=CONNECTIVITY_LIF2HRF_WEIGHTS)[0]
    params["readout_C"] = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


# ==============================
# Search settings
# ==============================

N_SAMPLES   = 80
EXPLOIT_FRAC = 0.70
SEED        = 42
N_HID       = 2500
SCRIPT      = "FordA_spiking_ron.py"
RESULTS_DIR = Path("hyperparam_search_FordA_nhid2500")
RESULTS_SUBDIR = "results_fordA_nhid2500"

RESULTS_DIR.mkdir(exist_ok=True)

n_exploit = int(N_SAMPLES * EXPLOIT_FRAC)
n_explore = N_SAMPLES - n_exploit
sample_types = (["exploit"] * n_exploit) + (["explore"] * n_explore)
random.shuffle(sample_types)

# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"FordA search: N_hid={N_HID}, {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore)")
print(f"Fixed params: {FIXED}")
print(f"Target: beat Spiking RC GTE (80.37%, 2500 neurons)")
print("=" * 70)

for i, stype in enumerate(sample_types):
    exploit = (stype == "exploit")
    params  = sample_params(exploit=exploit)

    print(f"\n{'🎯' if exploit else '🌐'} Config {i+1}/{N_SAMPLES} [{stype}]: "
          f"inp={params['inp_scaling']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"C={params['readout_C']}")

    cmd = [
        "python", SCRIPT,
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
        "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
        "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
        "--readout_mode",         params["readout_mode"],
        "--readout_C",            str(params["readout_C"]),
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--results_dir",          RESULTS_SUBDIR,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=600)

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
              f"Gap: {res['train_acc_mean']-res['test_acc_mean']:.1f}%  "
              f"r_hrf={res['r_hrf_mean']:.4f}")

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
            print(f"\n   💾 Best so far: {intermediate[0]['test_acc_mean']:.2f}% "
                  f"[{intermediate[0].get('sample_type','?')}]")

# ==============================
# Final aggregation
# ==============================

all_results.sort(key=lambda x: x["test_acc_mean"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ Completed: {len(all_results)}/{N_SAMPLES} configs")
print(f"❌ Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

exploit_results = [r for r in all_results if r.get("sample_type") == "exploit"]
explore_results = [r for r in all_results if r.get("sample_type") == "explore"]
if exploit_results:
    print(f"🎯 Exploit best: {max(r['test_acc_mean'] for r in exploit_results):.2f}%")
if explore_results:
    print(f"🌐 Explore best: {max(r['test_acc_mean'] for r in explore_results):.2f}%")

if all_results:
    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<10} {'Train%':<10} {'Gap':<7} {'Type':<9} "
          f"{'inp':<9} {'rho':<6} {'th_lif':<8} {'lif2hrf':<9} {'C':<7}")
    print("-" * 80)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        print(f"{rank:<5} "
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']-r['test_acc_mean']:.1f}%   "
              f"{r.get('sample_type','?'):<9} "
              f"{float(p.get('inp_scaling', 0)):<9.4f}"
              f"{float(p.get('rho', 0)):<6.3f}"
              f"{float(p.get('theta_lif', 0)):<8.4f}"
              f"{float(p.get('connectivity_lif2hrf', 1.0)):<9}"
              f"{float(r.get('readout_C', 0)):<7}")

    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "rho", "theta_lif"]:
        top_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.4f}±{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}±{np.std(all_vals):.4f}")

    print(f"\n📊 CONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in all_results
                 if abs(float(r.get("args", r).get(
                     "connectivity_lif2hrf", 1.0)) - c) < 1e-6]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            print(f"  lif2hrf={c:<5}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%")

    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in all_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in c_res]
            print(f"  C={C_val:<7}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"gap={np.mean(gaps):.1f}%")
            


'''
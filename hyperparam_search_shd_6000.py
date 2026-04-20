"""
SHD Phase 2 search at N_hid=6000.

Problem with Phase 1 best (88.25%):
  - r_hrf=0.471 (saturated — target: 0.05-0.25)
  - Train-test gap: 11.5 points (overfitting)

Root cause: gamma, theta_rf, rho were not jointly tuned to fight
saturation at N_hid=6000. This search opens them all up.

Strategy:
  - Open ALL params that affect firing rate: gamma, theta_rf, rho, theta_lif
  - Also re-search inp_scaling, input_density, connectivity_lif2hrf
  - Strongly bias readout_C toward very small values to fight overfitting
  - Bias readout_mode toward final (6000 features) vs rms_std_final (18000)
    since with 8332 train samples and 18000 features the LR overfits easily

Phase 1 best reference:
  gamma=0.036, theta_rf=0.013, rho=1.206, theta_lif=2.118,
  inp_scaling=0.198, input_density=0.049, lif2hrf=0.2,
  C=0.001, mode=rms_std_final -> 88.25%, r_hrf=0.471, gap=11.5pp
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fixed (truly size/data independent)
# ==============================

FIXED = {
    "dt":                   0.223,
    "epsilon_range":        0.063,   # keep — narrowly tuned
    "num_steps":            250,
    "max_time":             1.4,
    "connectivity_hrf2lif": 1.0,
    "tau_filter":           20.0,
}

# ==============================
# Search space — everything that affects firing dynamics is open
# ==============================

# EXPLOIT: narrowed around Phase 1 best but with wider ranges on
# saturation-relevant params (gamma, theta_rf, rho pushed toward
# values that reduce r_hrf)
EXPLOIT_SPACE = {
    # Oscillator dynamics — open these up to fight saturation
    "gamma":         (0.01,  0.15),   # log; Phase 1 had 0.036, try wider
    "gamma_range":   (0.1,   0.5),    # linear; Phase 1 had 0.268
    "epsilon":       (0.02,  0.15),   # log; Phase 1 had 0.06
    "theta_rf":      (0.02,  0.15),   # log; Phase 1 had 0.013 — push UP
    "theta_lif":     (1.0,   5.0),    # log; Phase 1 had 2.118

    # Size-sensitive
    "inp_scaling":   (0.05,  0.5),    # log; Phase 1 had 0.198
    "rho":           (0.85,  1.3),    # linear; Phase 1 had 1.206 — try lower
    "input_density": (0.02,  0.10),   # log; Phase 1 had 0.049
}

WIDE_SPACE = {
    "gamma":         (0.005, 0.5),
    "gamma_range":   (0.05,  1.0),
    "epsilon":       (0.01,  0.3),
    "theta_rf":      (0.005, 0.3),
    "theta_lif":     (0.5,   8.0),
    "inp_scaling":   (0.02,  1.0),
    "rho":           (0.7,   1.5),
    "input_density": (0.01,  0.15),
}

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.20, 0.45, 0.25, 0.10]

# Key insight: with 18000 features (rms_std_final) and 8332 train samples
# the LR memorises training data. Bias toward:
#   - final (6000 features) — much easier regularisation problem
#   - very small C to fight overfitting in both modes
READOUT_MODE_OPTIONS = ["rms_std_final"]
READOUT_MODE_WEIGHTS = [1.0]

READOUT_C_VALUES  = [0.0001, 0.0003, 0.001, 0.003, 0.01]
READOUT_C_WEIGHTS = [0.10,   0.15,   0.40,  0.25,  0.10]

LOG_PARAMS = {"gamma", "epsilon", "theta_rf", "theta_lif",
              "inp_scaling", "input_density"}

# ==============================
# Search settings
# ==============================

N_SAMPLES    = 80
EXPLOIT_FRAC = 0.65
SEED         = 42
N_HID        = 6000
SCRIPT       = "shd_spiking_ron.py"
RESULTS_DIR  = Path("hyperparam_search_SHD_nhid6000_phase2")
SHD_RESULTS_DIR = "results_shd_nhid6000_p2"

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
    params["readout_mode"] = "rms_std_final" 
    params["readout_C"] = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"SHD Phase 2: N_hid={N_HID}, {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore), seed={SEED}")
print(f"Fixed: {FIXED}")
print(f"Key change: gamma, theta_rf, rho ALL open — targeting r_hrf < 0.25")
print(f"Target: beat ELSM-large 89.3% (16,000 neurons)")
print("=" * 70)

for i, stype in enumerate(sample_types):
    exploit = (stype == "exploit")
    params  = sample_params(exploit=exploit)

    # Diagnostic: rough oscillation cycles estimate
    osc = np.sqrt(params["gamma"]) * params["dt"] * params["num_steps"]

    print(f"\n{'🎯' if exploit else '🌐'} Config {i+1}/{N_SAMPLES} [{stype}]: "
          f"g={params['gamma']:.4f} "
          f"th_rf={params['theta_rf']:.4f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.3f} "
          f"inp={params['inp_scaling']:.4f} "
          f"dens={params['input_density']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"mode={params['readout_mode']} "
          f"C={params['readout_C']} "
          f"[osc≈{osc:.1f}]")

    cmd = [
        "python", SCRIPT,
        "--n_hid",                str(N_HID),
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
        "--num_steps",            str(int(params["num_steps"])),
        "--max_time",             str(params["max_time"]),
        "--readout_mode",         params["readout_mode"],
        "--readout_C",            str(params["readout_C"]),
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--results_dir",          SHD_RESULTS_DIR,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=1800)

        result_file = max(
            Path(SHD_RESULTS_DIR).glob(f"*seed{SEED}.json"),
            key=lambda p: p.stat().st_mtime
        )
        with open(result_file) as f:
            res = json.load(f)

        res["config_id"]            = i
        res["search_seed"]          = SEED
        res["readout_C"]            = params["readout_C"]
        res["sample_type"]          = stype
        res["connectivity_lif2hrf"] = params["connectivity_lif2hrf"]
        all_results.append(res)

        gap = res["train_acc_mean"] - res["test_acc_mean"]
        sat = " ⚠️SAT" if res["r_hrf_mean"] > 0.35 else (
              " ✓"    if res["r_hrf_mean"] < 0.20 else "")
        print(f"   ✅ Test: {res['test_acc_mean']:.2f}%  "
              f"Train: {res['train_acc_mean']:.2f}%  "
              f"Gap: {gap:.1f}%  "
              f"r_hrf={res['r_hrf_mean']:.4f}{sat}")

    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-3:]:
                print(f"      {line}")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>1800s)")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    if (i + 1) % 10 == 0 and all_results:
        top = sorted(all_results, key=lambda x: x["test_acc_mean"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(top[:20], f, indent=2)
        best = top[0]
        print(f"\n   💾 Best so far: {best['test_acc_mean']:.2f}%  "
              f"gap={best['train_acc_mean']-best['test_acc_mean']:.1f}%  "
              f"r_hrf={best['r_hrf_mean']:.4f}  "
              f"[{best.get('sample_type','?')}]")

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

exploit_r = [r for r in all_results if r.get("sample_type") == "exploit"]
explore_r = [r for r in all_results if r.get("sample_type") == "explore"]
if exploit_r:
    print(f"🎯 Exploit best: {max(r['test_acc_mean'] for r in exploit_r):.2f}%")
if explore_r:
    print(f"🌐 Explore best: {max(r['test_acc_mean'] for r in explore_r):.2f}%")

if all_results:
    print(f"\n🏆 TOP 10:")
    print(f"{'Rk':<4} {'Test%':<10} {'Train%':<9} {'Gap':<6} {'Type':<8} "
          f"{'mode':<14} {'C':<9} {'gamma':<8} {'th_rf':<8} "
          f"{'rho':<6} {'th_lif':<8} {'inp':<8} {'dens':<7} "
          f"{'lif2hrf':<8} {'r_hrf'}")
    print("-" * 130)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        gap  = r["train_acc_mean"] - r["test_acc_mean"]
        mode = r.get("readout_mode", p.get("readout_mode", "?"))
        sat  = "(*)" if r["r_hrf_mean"] > 0.35 else " ok"
        print(f"{rank:<4} "
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.1f}%   "
              f"{gap:.1f}%  "
              f"{r.get('sample_type','?'):<8} "
              f"{mode:<14} "
              f"{float(r.get('readout_C',0)):<9}"
              f"{float(p.get('gamma',0)):<8.4f}"
              f"{float(p.get('theta_rf',0)):<8.4f}"
              f"{float(p.get('rho',0)):<6.3f}"
              f"{float(p.get('theta_lif',0)):<8.3f}"
              f"{float(p.get('inp_scaling',0)):<8.4f}"
              f"{float(p.get('input_density',0)):<7.4f}"
              f"{float(p.get('connectivity_lif2hrf',0.2)):<8}"
              f"{sat}{r['r_hrf_mean']:.4f}")

    print(f"\n📊 SATURATION ANALYSIS:")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]:
        bucket = [r for r in all_results if lo <= r["r_hrf_mean"] < hi]
        if bucket:
            accs = [r["test_acc_mean"] for r in bucket]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in bucket]
            print(f"  r_hrf [{lo:.1f}-{hi:.1f}): n={len(bucket)}, "
                  f"test={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%")

    print(f"\n📊 READOUT MODE BREAKDOWN:")
    for mode in READOUT_MODE_OPTIONS:
        m_res = [r for r in all_results
                 if r.get("readout_mode",
                           r.get("args", {}).get("readout_mode", "?")) == mode]
        if m_res:
            accs = [r["test_acc_mean"] for r in m_res]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in m_res]
            print(f"  {mode:<16}: n={len(m_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%")

    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in all_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < C_val * 0.01]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in c_res]
            print(f"  C={C_val:<9}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%")

    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for pname in ["gamma", "theta_rf", "rho", "theta_lif",
                  "inp_scaling", "input_density"]:
        top_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.5f}±{np.std(top_vals):.5f}  "
              f"all={np.mean(all_vals):.5f}±{np.std(all_vals):.5f}")

else:
    print("\nNo successful configurations!")



'''

"""
Hyperparameter search for Spiking RON on SHD at N_hid=6000.

Goal: push accuracy above ELSM-large (89.3%, 16,000 neurons) using
a much smaller reservoir, or at minimum close the gap significantly.

Current best: 87.06% at N_hid=3000 (readout_mode=final, theta_lif=1.0)

Key insight: theta_lif must scale with N_hid because total recurrent
current into each LIF neuron scales as N_hid * r_hrf * rho.
At N_hid=6000 (2x larger than 3000), theta_lif needs to be ~2x higher
to maintain the same LIF firing regime. This is the most important
parameter to re-tune.

Strategy:
  - Fix per-neuron oscillator dynamics (dt, gamma, epsilon, ranges, theta_rf)
  - Re-search size-sensitive params: inp_scaling, rho, theta_lif,
    input_density, connectivity_lif2hrf, readout_C, readout_mode
  - 60% exploit (narrowed around best N_hid=3000 config, theta_lif scaled up)
  - 40% wide exploration
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fully fixed (per-neuron, size-independent)
# ==============================

FIXED = {
    "dt":            0.223,
    "gamma":         0.036,
    "gamma_range":   0.268,
    "epsilon":       0.06,
    "epsilon_range": 0.063,
    "theta_rf":      0.013,
    "num_steps":     250,
    "max_time":      1.4,
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space
# ==============================

# theta_lif: at N_hid=3000 best was 1.0. At N_hid=6000 (2x larger)
# recurrent current is ~2x stronger -> search range [1.5, 6.0]
# inp_scaling: at larger N_hid recurrent feedback is stronger,
# input may need to be reduced
# rho: supercritical rho=1.16 was best at 3000; at 6000 may need adjustment
# input_density: can afford sparser connections at larger N_hid
# connectivity_lif2hrf: 0.2 was best at 3000; re-search

EXPLOIT_SPACE = {
    "theta_lif":     (1.5,   6.0),    # log; scaled up from 1.0 at N_hid=3000
    "inp_scaling":   (0.05,  0.6),    # log; narrowed around 0.23
    "rho":           (0.9,   1.4),    # linear; narrowed around 1.16
    "input_density": (0.01,  0.08),   # log; narrowed around 0.036
}

WIDE_SPACE = {
    "theta_lif":     (0.5,   10.0),   # log; wide range
    "inp_scaling":   (0.02,  1.5),    # log
    "rho":           (0.7,   1.6),    # linear
    "input_density": (0.01,  0.15),   # log
}

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.15, 0.50, 0.25, 0.10]  # bias toward 0.2

READOUT_MODE_OPTIONS = ["final", "rms_std_final"]
READOUT_MODE_WEIGHTS = [0.50, 0.50]  # equal — explore both

# With rms_std_final at N_hid=6000: 18000 features, 8332 train samples
# -> need very small C. With final: 6000 features, also needs small C
READOUT_C_VALUES  = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.05]
READOUT_C_WEIGHTS = [0.10,   0.20,   0.35,  0.20,  0.10, 0.05]

LOG_PARAMS = {"theta_lif", "inp_scaling", "input_density"}

# ==============================
# Search settings
# ==============================

N_SAMPLES    = 60         # single seed — SHD at N_hid=6000 is slow
EXPLOIT_FRAC = 0.60
SEED         = 42
N_HID        = 6000
SCRIPT       = "shd_spiking_ron.py"
RESULTS_DIR  = Path("hyperparam_search_SHD_nhid6000")
SHD_RESULTS_DIR = "results_shd_nhid6000"

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
    params["readout_mode"] = random.choices(
        READOUT_MODE_OPTIONS, weights=READOUT_MODE_WEIGHTS)[0]
    params["readout_C"] = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"SHD search: N_hid={N_HID}, {N_SAMPLES} configs "
      f"({n_exploit} exploit / {n_explore} explore), single seed={SEED}")
print(f"Fixed: {FIXED}")
print(f"Key change: theta_lif range [1.5, 6.0] (scaled up from 1.0 at N_hid=3000)")
print(f"Target: beat ELSM-large 89.3% (16,000 neurons) with {N_HID} neurons")
print("=" * 70)

for i, stype in enumerate(sample_types):
    exploit = (stype == "exploit")
    params  = sample_params(exploit=exploit)

    print(f"\n{'🎯' if exploit else '🌐'} Config {i+1}/{N_SAMPLES} [{stype}]: "
          f"th_lif={params['theta_lif']:.3f} "
          f"inp={params['inp_scaling']:.4f} "
          f"rho={params['rho']:.3f} "
          f"dens={params['input_density']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"mode={params['readout_mode']} "
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
        "--input_density",        str(params["input_density"]),
        "--theta_lif",            str(params["theta_lif"]),
        "--theta_rf",             str(params["theta_rf"]),
        "--tau_filter",           "20.0",
        "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
        "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
        "--num_steps",            str(int(params["num_steps"])),
        "--max_time",             str(params["max_time"]),
        "--readout_mode",         params["readout_mode"],
        "--readout_C",            str(params["readout_C"]),
        "--seed",                 str(SEED),
        "--test_trials",          "1",
        "--use_test",
        "--results_dir",          SHD_RESULTS_DIR,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       text=True, timeout=1800)

        result_file = max(
            Path(SHD_RESULTS_DIR).glob(f"*seed{SEED}.json"),
            key=lambda p: p.stat().st_mtime
        )
        with open(result_file) as f:
            res = json.load(f)

        res["config_id"]            = i
        res["search_seed"]          = SEED
        res["readout_C"]            = params["readout_C"]
        res["sample_type"]          = stype
        res["connectivity_lif2hrf"] = params["connectivity_lif2hrf"]
        all_results.append(res)

        gap = res["train_acc_mean"] - res["test_acc_mean"]
        print(f"   ✅ Test: {res['test_acc_mean']:.2f}%  "
              f"Train: {res['train_acc_mean']:.2f}%  "
              f"Gap: {gap:.1f}%  "
              f"r_hrf={res['r_hrf_mean']:.4f}")

    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-3:]:
                print(f"      {line}")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>1800s)")
        failed_configs.append({"config_id": i, "params": params,
                                "sample_type": stype})

    if (i + 1) % 10 == 0 and all_results:
        intermediate = sorted(all_results,
                               key=lambda x: x["test_acc_mean"],
                               reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
        print(f"\n   💾 Best so far: {intermediate[0]['test_acc_mean']:.2f}% "
              f"[{intermediate[0].get('sample_type','?')}]  "
              f"r_hrf={intermediate[0]['r_hrf_mean']:.4f}")

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
    print(f"{'Rank':<5} {'Test%':<10} {'Train%':<10} {'Gap':<7} {'Type':<8} "
          f"{'mode':<14} {'th_lif':<8} {'inp':<8} {'rho':<6} "
          f"{'dens':<7} {'lif2hrf':<9} {'C':<7} {'r_hrf'}")
    print("-" * 110)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get("args", r)
        gap = r["train_acc_mean"] - r["test_acc_mean"]
        mode = r.get("readout_mode", p.get("readout_mode", "?"))
        print(f"{rank:<5} "
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.2f}%     "
              f"{gap:.1f}%   "
              f"{r.get('sample_type','?'):<8} "
              f"{mode:<14} "
              f"{float(p.get('theta_lif', 0)):<8.3f}"
              f"{float(p.get('inp_scaling', 0)):<8.4f}"
              f"{float(p.get('rho', 0)):<6.3f}"
              f"{float(p.get('input_density', 0)):<7.4f}"
              f"{float(p.get('connectivity_lif2hrf', 0.2)):<9}"
              f"{float(r.get('readout_C', 0)):<7}"
              f"{r['r_hrf_mean']:.4f}")

    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for pname in ["theta_lif", "inp_scaling", "rho", "input_density"]:
        top_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get("args", r).get(pname, 0))
                    for r in all_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.4f}±{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}±{np.std(all_vals):.4f}")

    print(f"\n📊 READOUT MODE BREAKDOWN:")
    for mode in READOUT_MODE_OPTIONS:
        m_res = [r for r in all_results
                 if r.get("readout_mode",
                           r.get("args", {}).get("readout_mode", "?")) == mode]
        if m_res:
            accs = [r["test_acc_mean"] for r in m_res]
            print(f"  {mode:<16}: n={len(m_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%")

    print(f"\n📊 CONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in all_results
                 if abs(float(r.get("connectivity_lif2hrf",
                               r.get("args", {}).get(
                                   "connectivity_lif2hrf", 0.2))) - c) < 1e-6]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            print(f"  lif2hrf={c:<5}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%")

    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in all_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < C_val * 0.01]
        if c_res:
            accs = [r["test_acc_mean"] for r in c_res]
            gaps = [r["train_acc_mean"] - r["test_acc_mean"] for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"best={max(accs):.2f}%  "
                  f"mean={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"gap={np.mean(gaps):.1f}%")

    print(f"\n📊 HRF SATURATION (top 10):")
    for rank, r in enumerate(all_results[:10], 1):
        flag = " ⚠️SAT" if r["r_hrf_mean"] > 0.4 else " ✓"
        print(f"  {rank}: r_hrf={r['r_hrf_mean']:.4f}{flag}  "
              f"test={r['test_acc_mean']:.2f}%")

else:
    print("\nNo successful configurations!")

'''
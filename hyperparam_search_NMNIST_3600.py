"""
Hyperparameter search for Spiking RON on N-MNIST at N_hid=3600.

Targeted search fixing the best per-neuron oscillator dynamics from
the N_hid=512 search, while re-optimising parameters sensitive to
reservoir size (inp_scaling, input_density, rho, theta_lif, theta_rf,
connectivity_lif2hrf).

num_steps is fixed to 30 (best from original search, size-independent).
readout_mode is fixed to 'final' (3600 features) to keep the LR tractable
in memory. Once the best config is found, re-run with rms_std_final on a
high-memory node to get the final reported result.

Best config from N_hid=512 search (config_id=47, 95.31%):
  dt=0.109, gamma=0.109, gamma_range=0.362, epsilon=0.021,
  epsilon_range=0.083, inp_scaling=0.218, rho=1.207,
  theta_lif=0.189, theta_rf=0.045, input_density=0.063,
  num_steps=30, spatial_factor=2, readout_mode=rms_std_final, C=0.1

Target: N_hid=3600, connectivity_lif2hrf searched
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fully fixed parameters
# ==============================

FIXED = {
    "dt":                   0.1092124883046145,
    "gamma":                0.10895240475386166,
    "gamma_range":          0.36156780786060716,
    "epsilon":              0.02076624190152689,
    "epsilon_range":        0.08275192805570303,
    "tau_filter":           20.0,
    "spatial_factor":       2,           # 578 input channels
    "num_steps":            30,          # fixed — best from original search
    "readout_mode":         "final",     # 3600 features — tractable for LR
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space (size-sensitive params only)
# ==============================

SEARCH_SPACE = {
    "inp_scaling":   (0.02,  0.5),    # log
    "input_density": (0.01,  0.2),    # log
    "rho":           (0.8,   1.6),    # linear
    "theta_lif":     (0.05,  0.5),    # log
    "theta_rf":      (0.01,  0.15),   # log
}

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.15, 0.40, 0.25, 0.20]

# With final readout: 3600 features, 60000 samples — ratio is fine.
# Can afford larger C than with rms_std_final (10800 features).
# Bias toward 0.01-0.1 range; keep small values for exploration.
READOUT_C_VALUES  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
READOUT_C_WEIGHTS = [0.05,  0.10,  0.25, 0.30, 0.20, 0.10]

LOG_PARAMS = {"inp_scaling", "input_density", "theta_lif", "theta_rf"}

# ==============================
# Search settings
# ==============================

N_SAMPLES          = 40
SEEDS              = [0, 1, 2]
N_HID              = 3600
SCRIPT             = "nMNIST_spiking_ron.py"
RESULTS_DIR        = Path("hyperparam_search_NMNIST_nhid3600")
NMNIST_RESULTS_DIR = "results_nmnist_search_3600"

RESULTS_DIR.mkdir(exist_ok=True)

# ==============================
# Aggregation helper — defined before the loop so it can be called inside it
# ==============================

def _aggregate(all_results, seeds):
    summary = {}
    for r in all_results:
        summary.setdefault(r["config_id"], []).append(r)
    final = []
    for cid, runs in summary.items():
        if len(runs) == len(seeds):
            test_accs  = [r["test_acc_mean"]  for r in runs]
            train_accs = [r["train_acc_mean"] for r in runs]
            final.append({
                "config_id":            cid,
                "mean_test_acc":        float(np.mean(test_accs)),
                "std_test_acc":         float(np.std(test_accs)),
                "mean_train_acc":       float(np.mean(train_accs)),
                "overfit_gap":          float(np.mean(train_accs) -
                                              np.mean(test_accs)),
                "connectivity_lif2hrf": runs[0].get(
                    "connectivity_lif2hrf",
                    runs[0]["args"].get("connectivity_lif2hrf", 0.2)),
                "readout_C":            runs[0].get("readout_C", 0.01),
                "params":               runs[0]["args"],
            })
    return final


def sample_params():
    params = dict(FIXED)
    for key, (lo, hi) in SEARCH_SPACE.items():
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
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"N-MNIST search: N_hid={N_HID}, {N_SAMPLES} configs, "
      f"{len(SEEDS)} seeds each")
print(f"Fixed params: {FIXED}")
print(f"Note: using readout_mode=final for search (3600 features).")
print(f"      Re-run best config with rms_std_final on high-mem node.")
print("=" * 70)

for i in range(N_SAMPLES):
    params = sample_params()

    print(f"\nConfig {i+1}/{N_SAMPLES}: "
          f"inp={params['inp_scaling']:.3f} "
          f"dens={params['input_density']:.3f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} "
          f"th_rf={params['theta_rf']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"C={params['readout_C']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid",                str(N_HID),
            "--batch",                "256",
            "--spatial_factor",       str(int(params["spatial_factor"])),
            "--num_steps",            str(int(params["num_steps"])),
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
            "--tau_filter",           str(params["tau_filter"]),
            "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
            "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
            "--readout_mode",         params["readout_mode"],
            "--readout_C",            str(params["readout_C"]),
            "--seed",                 str(seed),
            "--test_trials",          "1",
            "--use_test",
            "--results_dir",          NMNIST_RESULTS_DIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=1800)

            result_file = max(
                Path(NMNIST_RESULTS_DIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]            = i
            res["search_seed"]          = seed
            res["readout_C"]            = params["readout_C"]
            res["connectivity_lif2hrf"] = params["connectivity_lif2hrf"]
            all_results.append(res)

        except subprocess.CalledProcessError as e:
            print(f"  FAILED (seed {seed})")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"    {line}")
            config_failed = True
            break
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (seed {seed}, >1800s)")
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

    if (i + 1) % 10 == 0:
        intermediate = _aggregate(all_results, SEEDS)
        intermediate.sort(key=lambda x: x["mean_test_acc"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
        if intermediate:
            print(f"\n  Saved top-20 intermediate "
                  f"(best so far: {intermediate[0]['mean_test_acc']:.2f}%)")


# ==============================
# Final aggregation
# ==============================

final_results = _aggregate(all_results, SEEDS)
final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"Completed: {len(final_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

if final_results:
    print(f"\nTOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<18} {'Train%':<10} {'Gap':<7} "
          f"{'lif2hrf':<8} {'C':<7} "
          f"{'inp':<8} {'dens':<8} {'rho':<6} "
          f"{'th_lif':<8} {'th_rf'}")
    print("-" * 90)
    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}+/-{r['std_test_acc']:.2f}    "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{r.get('connectivity_lif2hrf', 0.2):<8}"
              f"{r['readout_C']:<7}"
              f"{p.get('inp_scaling', 0):<8.3f}"
              f"{p.get('input_density', 0):<8.3f}"
              f"{p.get('rho', 0):<6.3f}"
              f"{p.get('theta_lif', 0):<8.4f}"
              f"{p.get('theta_rf', 0):.4f}")

    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "rho", "theta_lif", "theta_rf"]:
        top_vals = [r["params"].get(pname, 0) for r in final_results[:10]]
        all_vals = [r["params"].get(pname, 0) for r in final_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.4f}+/-{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}+/-{np.std(all_vals):.4f}")

    print(f"\nCONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in final_results
                 if abs(float(r.get("connectivity_lif2hrf", 0.2)) - c) < 1e-6]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  lif2hrf={c:<5}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  C={C_val:<7}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    print(f"\nNOTE: These results use readout_mode=final (3600 features).")
    print(f"      Re-run best config with --readout_mode rms_std_final")
    print(f"      on a high-memory node (--mem=96000) for final result.")







'''
"""
Hyperparameter search for Spiking RON on N-MNIST at N_hid=3600.

Phase 2 targeted search: fix the best per-neuron parameters from the
N_hid=512 search (dt, gamma, epsilon, theta_lif, theta_rf, num_steps,
spatial_factor) and re-optimise only the parameters that are sensitive
to reservoir size: inp_scaling, input_density, rho, readout_C.

Best config from N_hid=512 search (config_id=47, 95.31%):
  dt=0.109, gamma=0.109, gamma_range=0.362, epsilon=0.021,
  epsilon_range=0.083, inp_scaling=0.218, rho=1.207,
  theta_lif=0.189, theta_rf=0.045, input_density=0.063,
  num_steps=30, spatial_factor=2, readout_mode=rms_std_final, C=0.1

Target: N_hid=3600, connectivity_lif2hrf=0.2
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fixed parameters (from best N_hid=512 config)
# ==============================

FIXED = {
    "dt":            0.1092124883046145,
    "gamma":         0.10895240475386166,
    "gamma_range":   0.36156780786060716,
    "epsilon":       0.02076624190152689,
    "epsilon_range": 0.08275192805570303,
    "theta_lif":     0.18938194585216667,
    "theta_rf":      0.04496743965658108,
    "tau_filter":    20.0,
    "num_steps":     30,
    "spatial_factor": 2,
    "readout_mode":  "rms_std_final",
    "connectivity_lif2hrf": 0.2,
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space (size-sensitive params only)
# ==============================

# inp_scaling: at larger N_hid the recurrent feedback is stronger,
# so the input drive may need to be reduced. Search around the best
# value with a wider range toward smaller values.
# input_density: at N_hid=3600 each neuron can connect to fewer
# inputs proportionally; explore a wider range downward.
# rho: spectral radius optimal value can shift with reservoir size;
# search a wide range.
# readout_C: with 3*3600=10800 features the LR needs more
# regularisation than with 3*512=1536; bias toward smaller C.

SEARCH_SPACE = {
    "inp_scaling":   (0.02,  0.5),    # log; best was 0.218, explore wider
    "input_density": (0.01,  0.2),    # log; best was 0.063, explore lower end
    "rho":           (0.8,   1.6),    # linear; best was 1.207
}

READOUT_C_VALUES  = [0.01, 0.05, 0.1, 0.5, 1.0]
# With 10800 features we expect smaller C to be better — weight accordingly
READOUT_C_WEIGHTS = [0.30, 0.30, 0.25, 0.10, 0.05]

LOG_PARAMS = {"inp_scaling", "input_density"}

# ==============================
# Search settings
# ==============================

N_SAMPLES   = 60
SEEDS       = [0, 1, 2]      # 3 seeds per config
N_HID       = 3600
SCRIPT      = "nMNIST_spiking_ron.py"
RESULTS_DIR = Path("hyperparam_search_NMNIST_nhid3600")
NMNIST_RESULTS_DIR = "results_nmnist_search_3600"

RESULTS_DIR.mkdir(exist_ok=True)


def sample_params():
    params = dict(FIXED)  # start from fixed best values
    for key, (lo, hi) in SEARCH_SPACE.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    params["readout_C"] = random.choices(READOUT_C_VALUES,
                                         weights=READOUT_C_WEIGHTS)[0]
    return params


# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"N-MNIST Phase 2 search: N_hid={N_HID}, {N_SAMPLES} configs, "
      f"{len(SEEDS)} seeds each")
print(f"Fixed params: {FIXED}")
print("=" * 70)

for i in range(N_SAMPLES):
    params = sample_params()

    print(f"\nConfig {i+1}/{N_SAMPLES}: "
          f"inp={params['inp_scaling']:.3f} "
          f"dens={params['input_density']:.3f} "
          f"rho={params['rho']:.3f} "
          f"C={params['readout_C']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid",                str(N_HID),
            "--spatial_factor",       str(int(params["spatial_factor"])),
            "--num_steps",            str(int(params["num_steps"])),
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
            "--tau_filter",           str(params["tau_filter"]),
            "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
            "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
            "--readout_mode",         params["readout_mode"],
            "--readout_C",            str(params["readout_C"]),
            "--seed",                 str(seed),
            "--test_trials",          "1",
            "--use_test",
            "--results_dir",          NMNIST_RESULTS_DIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=900)

            result_file = max(
                Path(NMNIST_RESULTS_DIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]   = i
            res["search_seed"] = seed
            res["readout_C"]   = params["readout_C"]
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
            print(f"  -> Test: {np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"Train: {np.mean(train_accs):.2f}%  "
                  f"Gap: {np.mean(train_accs)-np.mean(accs):.1f}%")

    # Save intermediate results every 10 configs
    if (i + 1) % 10 == 0:
        intermediate = _aggregate(all_results, SEEDS)
        intermediate.sort(key=lambda x: x["mean_test_acc"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
        if intermediate:
            print(f"\n  Saved top-20 intermediate "
                  f"(best so far: {intermediate[0]['mean_test_acc']:.2f}%)")


def _aggregate(all_results, seeds):
    summary = {}
    for r in all_results:
        summary.setdefault(r["config_id"], []).append(r)
    final = []
    for cid, runs in summary.items():
        if len(runs) == len(seeds):
            test_accs  = [r["test_acc_mean"]  for r in runs]
            train_accs = [r["train_acc_mean"] for r in runs]
            final.append({
                "config_id":      cid,
                "mean_test_acc":  float(np.mean(test_accs)),
                "std_test_acc":   float(np.std(test_accs)),
                "mean_train_acc": float(np.mean(train_accs)),
                "overfit_gap":    float(np.mean(train_accs) - np.mean(test_accs)),
                "readout_C":      runs[0].get("readout_C", 0.1),
                "params":         runs[0]["args"],
            })
    return final


# ==============================
# Final aggregation
# ==============================

final_results = _aggregate(all_results, SEEDS)
final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"Completed: {len(final_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

if final_results:
    print(f"\nTOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<18} {'Train%':<10} {'Gap':<7} "
          f"{'C':<6} {'inp':<8} {'dens':<8} {'rho':<6}")
    print("-" * 70)
    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}+/-{r['std_test_acc']:.2f}    "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{r['readout_C']:<6}"
              f"{p.get('inp_scaling', 0):<8.3f}"
              f"{p.get('input_density', 0):<8.3f}"
              f"{p.get('rho', 0):<6.3f}")

    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "rho"]:
        top_vals = [r["params"].get(pname, 0) for r in final_results[:10]]
        all_vals = [r["params"].get(pname, 0) for r in final_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.4f}+/-{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}+/-{np.std(all_vals):.4f}")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  C={C_val:<6}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")






"""
Hyperparameter search for Spiking RON on N-MNIST at N_hid=3600.

Targeted search fixing the best per-neuron oscillator dynamics from
the N_hid=512 search, while re-optimising parameters sensitive to
reservoir size (inp_scaling, input_density, rho) and adding back
theta_lif, theta_rf, num_steps, and connectivity_lif2hrf which can
all interact with reservoir size.

Best config from N_hid=512 search (config_id=47, 95.31%):
  dt=0.109, gamma=0.109, gamma_range=0.362, epsilon=0.021,
  epsilon_range=0.083, inp_scaling=0.218, rho=1.207,
  theta_lif=0.189, theta_rf=0.045, input_density=0.063,
  num_steps=30, spatial_factor=2, readout_mode=rms_std_final, C=0.1

Target: N_hid=3600, connectivity_lif2hrf searched
"""

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# ==============================
# Fully fixed parameters
# (oscillator dynamics — independent of reservoir size)
# ==============================

FIXED = {
    "dt":            0.1092124883046145,
    "gamma":         0.10895240475386166,
    "gamma_range":   0.36156780786060716,
    "epsilon":       0.02076624190152689,
    "epsilon_range": 0.08275192805570303,
    "tau_filter":    20.0,
    "spatial_factor": 2,          # 578 input channels
    "readout_mode":  "rms_std_final",
    "connectivity_hrf2lif": 1.0,
}

# ==============================
# Search space
# ==============================

SEARCH_SPACE = {
    # Size-sensitive: scale with N_hid
    "inp_scaling":   (0.02,  0.5),    # log
    "input_density": (0.01,  0.2),    # log
    "rho":           (0.8,   1.6),    # linear

    # Threshold params: interact with inp_scaling and input_density
    # Narrowed around best values (theta_lif=0.189, theta_rf=0.045)
    "theta_lif":     (0.05,  0.5),    # log
    "theta_rf":      (0.01,  0.15),   # log

    # More time bins can help capture richer dynamics
    # num_steps=30 was best before; also try 50
}

NUM_STEPS_OPTIONS           = [30, 50]
NUM_STEPS_WEIGHTS           = [0.5, 0.5]   # equal weight — explore both

CONNECTIVITY_LIF2HRF_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_LIF2HRF_WEIGHTS = [0.15, 0.40, 0.25, 0.20]  # bias toward 0.2

# At N_hid=3600, rms_std_final gives 10800 features → need smaller C
READOUT_C_VALUES  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
READOUT_C_WEIGHTS = [0.15,  0.20,  0.25, 0.20, 0.15, 0.05]

LOG_PARAMS = {"inp_scaling", "input_density", "theta_lif", "theta_rf"}

# ==============================
# Search settings
# ==============================

N_SAMPLES   = 80          # more configs given wider search space
SEEDS       = [0, 1, 2]   # 3 seeds per config
N_HID       = 3600
SCRIPT      = "nMNIST_spiking_ron.py"
RESULTS_DIR = Path("hyperparam_search_NMNIST_nhid3600")
NMNIST_RESULTS_DIR = "results_nmnist_search_3600"

RESULTS_DIR.mkdir(exist_ok=True)


def sample_params():
    params = dict(FIXED)
    for key, (lo, hi) in SEARCH_SPACE.items():
        if key in LOG_PARAMS:
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    params["num_steps"]            = random.choices(
        NUM_STEPS_OPTIONS, weights=NUM_STEPS_WEIGHTS)[0]
    params["connectivity_lif2hrf"] = random.choices(
        CONNECTIVITY_LIF2HRF_OPTIONS, weights=CONNECTIVITY_LIF2HRF_WEIGHTS)[0]
    params["readout_C"]            = random.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


def _aggregate(all_results, seeds):
    summary = {}
    for r in all_results:
        summary.setdefault(r["config_id"], []).append(r)
    final = []
    for cid, runs in summary.items():
        if len(runs) == len(seeds):
            test_accs  = [r["test_acc_mean"]  for r in runs]
            train_accs = [r["train_acc_mean"] for r in runs]
            final.append({
                "config_id":             cid,
                "mean_test_acc":         float(np.mean(test_accs)),
                "std_test_acc":          float(np.std(test_accs)),
                "mean_train_acc":        float(np.mean(train_accs)),
                "overfit_gap":           float(np.mean(train_accs) -
                                               np.mean(test_accs)),
                "num_steps":             runs[0].get("num_steps", 30),
                "connectivity_lif2hrf":  runs[0].get(
                    "connectivity_lif2hrf",
                    runs[0]["args"].get("connectivity_lif2hrf", 0.2)),
                "readout_C":             runs[0].get("readout_C", 0.01),
                "params":                runs[0]["args"],
            })
    return final


# ==============================
# Run experiments
# ==============================

all_results    = []
failed_configs = []

print(f"N-MNIST search: N_hid={N_HID}, {N_SAMPLES} configs, "
      f"{len(SEEDS)} seeds each")
print(f"Fixed params: {FIXED}")
print("=" * 70)

for i in range(N_SAMPLES):
    params = sample_params()

    print(f"\nConfig {i+1}/{N_SAMPLES}: "
          f"T={params['num_steps']} "
          f"inp={params['inp_scaling']:.3f} "
          f"dens={params['input_density']:.3f} "
          f"rho={params['rho']:.3f} "
          f"th_lif={params['theta_lif']:.4f} "
          f"th_rf={params['theta_rf']:.4f} "
          f"lif2hrf={params['connectivity_lif2hrf']} "
          f"C={params['readout_C']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", SCRIPT,
            "--n_hid",                str(N_HID),
            "--spatial_factor",       str(int(params["spatial_factor"])),
            "--num_steps",            str(int(params["num_steps"])),
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
            "--tau_filter",           str(params["tau_filter"]),
            "--connectivity_lif2hrf", str(params["connectivity_lif2hrf"]),
            "--connectivity_hrf2lif", str(params["connectivity_hrf2lif"]),
            "--readout_mode",         params["readout_mode"],
            "--readout_C",            str(params["readout_C"]),
            "--seed",                 str(seed),
            "--test_trials",          "1",
            "--use_test",
            "--results_dir",          NMNIST_RESULTS_DIR,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=1200)

            result_file = max(
                Path(NMNIST_RESULTS_DIR).glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )
            with open(result_file) as f:
                res = json.load(f)

            res["config_id"]            = i
            res["search_seed"]          = seed
            res["readout_C"]            = params["readout_C"]
            res["num_steps"]            = params["num_steps"]
            res["connectivity_lif2hrf"] = params["connectivity_lif2hrf"]
            all_results.append(res)

        except subprocess.CalledProcessError as e:
            print(f"  FAILED (seed {seed})")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"    {line}")
            config_failed = True
            break
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (seed {seed}, >1200s)")
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

    # Save intermediate results every 10 configs
    if (i + 1) % 10 == 0:
        intermediate = _aggregate(all_results, SEEDS)
        intermediate.sort(key=lambda x: x["mean_test_acc"], reverse=True)
        with open(RESULTS_DIR / "summary_intermediate.json", "w") as f:
            json.dump(intermediate[:20], f, indent=2)
        if intermediate:
            print(f"\n  Saved top-20 intermediate "
                  f"(best so far: {intermediate[0]['mean_test_acc']:.2f}%)")


# ==============================
# Final aggregation
# ==============================

final_results = _aggregate(all_results, SEEDS)
final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)
with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "=" * 70)
print(f"Completed: {len(final_results)}/{N_SAMPLES} configs")
print(f"Failed:    {len(failed_configs)}/{N_SAMPLES} configs")
print("=" * 70)

if final_results:
    print(f"\nTOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<18} {'Train%':<10} {'Gap':<7} "
          f"{'T':<4} {'lif2hrf':<8} {'C':<7} "
          f"{'inp':<8} {'dens':<8} {'rho':<6} "
          f"{'th_lif':<8} {'th_rf'}")
    print("-" * 90)
    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}+/-{r['std_test_acc']:.2f}    "
              f"{r['mean_train_acc']:.1f}%    "
              f"{r['overfit_gap']:.1f}%  "
              f"{r.get('num_steps', 30):<4} "
              f"{r.get('connectivity_lif2hrf', 0.2):<8}"
              f"{r['readout_C']:<7}"
              f"{p.get('inp_scaling', 0):<8.3f}"
              f"{p.get('input_density', 0):<8.3f}"
              f"{p.get('rho', 0):<6.3f}"
              f"{p.get('theta_lif', 0):<8.4f}"
              f"{p.get('theta_rf', 0):.4f}")

    print(f"\nPARAMETER TRENDS (top 10 vs all):")
    for pname in ["inp_scaling", "input_density", "rho", "theta_lif", "theta_rf"]:
        top_vals = [r["params"].get(pname, 0) for r in final_results[:10]]
        all_vals = [r["params"].get(pname, 0) for r in final_results]
        print(f"  {pname:>15}: "
              f"top10={np.mean(top_vals):.4f}+/-{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}+/-{np.std(all_vals):.4f}")

    print(f"\nNUM_STEPS BREAKDOWN:")
    for T in NUM_STEPS_OPTIONS:
        t_res = [r for r in final_results if r.get("num_steps") == T]
        if t_res:
            accs = [r["mean_test_acc"] for r in t_res]
            print(f"  T={T:<4}: n={len(t_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    print(f"\nCONNECTIVITY LIF2HRF BREAKDOWN:")
    for c in CONNECTIVITY_LIF2HRF_OPTIONS:
        c_res = [r for r in final_results
                 if abs(float(r.get("connectivity_lif2hrf", 0.2)) - c) < 1e-6]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  lif2hrf={c:<5}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

    print(f"\nREGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_res = [r for r in final_results
                 if abs(float(r.get("readout_C", 0)) - C_val) < 1e-9]
        if c_res:
            accs = [r["mean_test_acc"] for r in c_res]
            print(f"  C={C_val:<7}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%")

'''
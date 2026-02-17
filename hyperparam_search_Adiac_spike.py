import random
import json
import subprocess
from itertools import product
from pathlib import Path
import numpy as np

# =============================
# Search space (paper-worthy)
# =============================

SEARCH_SPACE = {
    "gamma": (1.5, 4.0),
    "dt": (0.02, 0.5),
    "epsilon": (0.02, 0.15),  #(0.02, 0.15)
    "gamma_range": (0.5, 4.0),
    "epsilon_range": (0.0, 0.2), #(0.0, 0.1)
    "rho": (0.8, 3.0), #(0.8, 1.05)
    "input_scaling": (0.1, 10.0),
}

N_SAMPLES = 40          # random configs
SEEDS = [0, 1, 2]       # report mean ± std
N_HID = 100
CONNECTIVITY = 1.0
RESULTS_DIR = Path("hyperparam_search_Adiac")

RESULTS_DIR.mkdir(exist_ok=True)

# =============================
# Sampling function
# =============================

def sample_params():
    gamma = random.uniform(*SEARCH_SPACE["gamma"])
    epsilon = random.uniform(*SEARCH_SPACE["epsilon"])

    return {
        "gamma": gamma,
        "epsilon": epsilon,
        "gamma_range": random.uniform(*SEARCH_SPACE["gamma_range"]),
        "dt" : random.uniform(*SEARCH_SPACE["dt"]),
        "epsilon_range": random.uniform(*SEARCH_SPACE["epsilon_range"]),
        "rho": random.uniform(*SEARCH_SPACE["rho"]),
        "input_scaling": random.uniform(*SEARCH_SPACE["input_scaling"]),
    }

# =============================
# Run experiments
# =============================

all_results = []
failed_configs = []

for i in range(N_SAMPLES):
    params = sample_params()
    print(f"\n🔍 Config {i+1}/{N_SAMPLES}: {params}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", "Adiac_spiking_ron.py",
            "--n_hid", str(N_HID),
            "--dt", str(params["dt"]),
            "--gamma", str(params["gamma"]),
            "--epsilon", str(params["epsilon"]),
            "--gamma_range", str(params["gamma_range"]),
            "--epsilon_range", str(params["epsilon_range"]),
            "--rho", str(params["rho"]),
            "--inp_scaling", str(params["input_scaling"]),
            "--theta_lif", "0.05",
            "--theta_rf", "0.005",
            "--connectivity_lif2hrf", str(CONNECTIVITY),
            "--connectivity_hrf2lif", str(CONNECTIVITY),
            "--seed", str(seed),
            "--test_trials", "1",
            "--use_test"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
       

            result_file = max(
                Path("results_adiac").glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )

            with open(result_file) as f:
                res = json.load(f)

            res["config_id"] = i
            res["seed"] = seed
            all_results.append(res)
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Config {i+1} with seed {seed} FAILED (numerical instability)")
            config_failed = True
            break  # Skip remaining seeds for this config
    
    if config_failed:
        failed_configs.append({"config_id": i, "params": params})

# =============================
# Aggregate results
# =============================

summary = {}

for r in all_results:
    cid = r["config_id"]
    summary.setdefault(cid, []).append(r)

final_results = []

for cid, runs in summary.items():
    if len(runs) == len(SEEDS):  # Only include complete runs
        accs = [r["test_acc_mean"] for r in runs]
        final_results.append({
            "config_id": cid,
            "mean_test_acc": float(np.mean(accs)),
            "std_test_acc": float(np.std(accs)),
            "params": runs[0]["args"]
        })

final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)

with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

print("\n" + "="*70)
print(f"✅ Completed: {len(final_results)}/{N_SAMPLES} configs successful")
print(f"❌ Failed: {len(failed_configs)}/{N_SAMPLES} configs (numerical instability)")
print("="*70)

if final_results:
    print("\n🏆 Best configuration:")
    print(f"   Test accuracy: {final_results[0]['mean_test_acc']:.2f}%")
    print(f"   Params: {final_results[0]['params']}")
else:
    print("\n⚠️  No successful configurations!")
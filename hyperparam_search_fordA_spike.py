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
    "epsilon": (0.02, 0.15),
    "gamma_range": (0.0, 3.0),
    "epsilon_range": (0.0, 0.1),
    "rho": (0.8, 1.05),
    "input_scaling": (0.5, 3.0),
}

N_SAMPLES = 40          # random configs
SEEDS = [0, 1, 2]       # report mean ± std
N_HID = 800
CONNECTIVITY = 1.0
RESULTS_DIR = Path("hparam_search_fordA")

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
        "epsilon_range": random.uniform(*SEARCH_SPACE["epsilon_range"]),
        "rho": random.uniform(*SEARCH_SPACE["rho"]),
        "input_scaling": random.uniform(*SEARCH_SPACE["input_scaling"]),
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
            "python", "FordA_spiking_RON.py",
            "--n_hid", str(N_HID),
            "--dt", "0.2",
            "--gamma", str(params["gamma"]),
            "--epsilon", str(params["epsilon"]),
            "--gamma_range", str(params["gamma_range"]),
            "--epsilon_range", str(params["epsilon_range"]),
            "--rho", str(params["rho"]),
            "--inp_scaling", str(params["input_scaling"]),
            "--theta_lif", "0.05",
            "--theta_rf", "0.005",
            "--connectivity", str(CONNECTIVITY),
            "--seed", str(seed),
            "--use_test"
        ]

        subprocess.run(cmd, check=True)

        result_file = max(
            Path("results_fordA").glob(f"*seed{seed}.json"),
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
    accs = [r["test_acc"] for r in runs]
    final_results.append({
        "config_id": cid,
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "params": runs[0]["args"]
    })

final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\n🏆 Best configuration:")
print(final_results[0])

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

import random
import json
import subprocess
from pathlib import Path
import numpy as np

# =============================
# SHD Hyperparameter Search
# =============================
# Key insight: SHD has 700 input channels vs psMNIST's 1 channel.
# The dynamics regime is completely different:
#   - inp_scaling must be MUCH lower (700 channels amplify input drive)
#   - input_density is a critical new parameter (sparse afferent connectivity)
#   - Readout regularization (C) matters more with high-dim features
#
# Strategy: random search over oscillator + input + readout params
# =============================

SEARCH_SPACE = {
    # Oscillator dynamics — tuned for SHD autocorrelation timescale (~6.6 Hz)
    # Target: sqrt(gamma)*dt ∈ [0.05, 0.38] → 2-15 oscillation cycles over 250 steps
    # Data: τ_autocorr = 151ms = 27 bins → dominant freq ~6.6 Hz
    "gamma":         (0.01, 0.5),     # small gamma for SLOW oscillations
    "dt":            (0.2, 0.8),      # larger dt → slower dynamics
    "epsilon":       (0.02, 0.12),    # half-life ≈ ln2/(dt*eps) ≈ 20-50 steps
    "gamma_range":   (0.02, 0.5),     # frequency diversity across neurons
    "epsilon_range": (0.0, 0.08),     # damping diversity
    "rho":           (0.7, 1.5),      # recurrent stability
    
    # Input — real data: 32 spikes/bin avg, 564/700 channels active
    # NOT sparse like assumed! Dense bursty activity.
    # Dense: I ≈ 32 * inp_scaling/2 = 16*inp_scaling → need very small inp_scaling
    # Sparse (density=0.1): I ≈ 32*(70/700) * inp_scaling/2 * 3.16 ≈ 5*inp_scaling
    "input_scaling": (0.02, 1.8),     # keep input moderate
    
    # Sparse input projection
    # With 564 active channels, density=0.1 → ~56 active inputs/neuron/step
    "input_density": (0.03, 0.3),
    
    # Thresholds — need higher theta_lif to avoid saturation given dense input
    "theta_lif":     (0.03, 0.3),     # higher range to control LIF saturation
    "theta_rf":      (0.001, 0.02),   # HRF spike threshold
}

# Readout regularization: sweep C (inverse regularization strength)
# Lower C = stronger regularization = less overfitting
# Default LogisticRegression C=1.0, but with 700 features we need more regularization
READOUT_C_VALUES = [0.001, 0.01, 0.1, 1.0]

N_SAMPLES = 50          # random configs
SEEDS = [0, 1, 2]       # report mean ± std
N_HID = 3000             # smaller reservoir to reduce overfitting
RESULTS_DIR = Path("hyperparam_search_SHD")

RESULTS_DIR.mkdir(exist_ok=True)

# =============================
# Sampling function
# =============================

def sample_params():
    """Sample random hyperparameters from search space."""
    params = {}
    for key, (lo, hi) in SEARCH_SPACE.items():
        # Log-uniform for parameters spanning orders of magnitude
        if key in ("input_scaling", "input_density", "theta_lif", "theta_rf", 
                    "epsilon", "dt"):
            params[key] = np.exp(random.uniform(np.log(lo), np.log(hi)))
        else:
            params[key] = random.uniform(lo, hi)
    
    # Sample readout C (log-uniform from the discrete set)
    params["readout_C"] = random.choice(READOUT_C_VALUES)
    
    return params

# =============================
# Run experiments
# =============================

all_results = []
failed_configs = []

for i in range(N_SAMPLES):
    params = sample_params()
    
    # Print compact summary
    print(f"\n🔍 Config {i+1}/{N_SAMPLES}: "
          f"inp={params['input_scaling']:.3f} "
          f"dens={params['input_density']:.3f} "
          f"dt={params['dt']:.3f} "
          f"γ={params['gamma']:.2f}±{params['gamma_range']:.2f} "
          f"ε={params['epsilon']:.4f}±{params['epsilon_range']:.3f} "
          f"ρ={params['rho']:.2f} "
          f"θlif={params['theta_lif']:.4f} "
          f"θrf={params['theta_rf']:.4f} "
          f"C={params['readout_C']}")

    config_failed = False
    for seed in SEEDS:
        cmd = [
            "python", "shd_spiking_RON.py",
            "--n_hid", str(N_HID),
            "--dt", str(params["dt"]),
            "--gamma", str(params["gamma"]),
            "--epsilon", str(params["epsilon"]),
            "--gamma_range", str(params["gamma_range"]),
            "--epsilon_range", str(params["epsilon_range"]),
            "--rho", str(params["rho"]),
            "--inp_scaling", str(params["input_scaling"]),
            "--input_density", str(params["input_density"]),
            "--theta_lif", str(params["theta_lif"]),
            "--theta_rf", str(params["theta_rf"]),
            "--connectivity_lif2hrf", "1.0",
            "--connectivity_hrf2lif", "1.0",
            "--seed", str(seed),
            "--test_trials", "1",
            "--use_test",
            "--readout_C", str(params["readout_C"]),
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)

            # Find the most recent result file for this seed
            result_file = max(
                Path("results_shd").glob(f"*seed{seed}.json"),
                key=lambda p: p.stat().st_mtime
            )

            with open(result_file) as f:
                res = json.load(f)

            res["config_id"] = i
            res["seed"] = seed
            res["readout_C"] = params["readout_C"]
            all_results.append(res)
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Config {i+1} seed {seed} FAILED")
            if e.stderr:
                # Print last 3 lines of stderr for debugging
                lines = e.stderr.strip().split('\n')
                for line in lines[-3:]:
                    print(f"   {line}")
            config_failed = True
            break
        except subprocess.TimeoutExpired:
            print(f"⏰ Config {i+1} seed {seed} TIMEOUT (>600s)")
            config_failed = True
            break
    
    if config_failed:
        failed_configs.append({"config_id": i, "params": params})
    else:
        # Print quick result for this config
        config_results = [r for r in all_results if r["config_id"] == i]
        if len(config_results) == len(SEEDS):
            accs = [r["test_acc_mean"] for r in config_results]
            train_accs = [r["train_acc_mean"] for r in config_results]
            print(f"   ✅ Test: {np.mean(accs):.2f}±{np.std(accs):.2f}%  "
                  f"Train: {np.mean(train_accs):.2f}%  "
                  f"Gap: {np.mean(train_accs)-np.mean(accs):.1f}%")

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
        test_accs = [r["test_acc_mean"] for r in runs]
        train_accs = [r["train_acc_mean"] for r in runs]
        final_results.append({
            "config_id": cid,
            "mean_test_acc": float(np.mean(test_accs)),
            "std_test_acc": float(np.std(test_accs)),
            "mean_train_acc": float(np.mean(train_accs)),
            "overfit_gap": float(np.mean(train_accs) - np.mean(test_accs)),
            "params": runs[0]["args"],
            "readout_C": runs[0].get("readout_C", 1.0),
        })

final_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(final_results, f, indent=2)

with open(RESULTS_DIR / "failed_configs.json", "w") as f:
    json.dump(failed_configs, f, indent=2)

# =============================
# Print results
# =============================

print("\n" + "="*70)
print(f"✅ Completed: {len(final_results)}/{N_SAMPLES} configs successful")
print(f"❌ Failed: {len(failed_configs)}/{N_SAMPLES} configs")
print("="*70)

if final_results:
    print(f"\n🏆 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Test%':<12} {'Train%':<10} {'Gap':<8} "
          f"{'inp_scl':<8} {'density':<8} {'C':<6} {'dt':<7} "
          f"{'gamma':<7} {'eps':<7} {'rho':<6}")
    print("-" * 100)
    
    for rank, r in enumerate(final_results[:10], 1):
        p = r["params"]
        print(f"{rank:<5} "
              f"{r['mean_test_acc']:.2f}±{r['std_test_acc']:.2f}  "
              f"{r['mean_train_acc']:.1f}%     "
              f"{r['overfit_gap']:.1f}%   "
              f"{p.get('inp_scaling', '?'):<8.3f}"
              f"{p.get('input_density', '?'):<8.3f}"
              f"{r.get('readout_C', '?'):<6}"
              f"{p.get('dt', '?'):<7.3f}"
              f"{p.get('gamma', '?'):<7.2f}"
              f"{p.get('epsilon', '?'):<7.4f}"
              f"{p.get('rho', '?'):<6.2f}")
    
    # Analyze trends
    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    top10 = final_results[:10]
    for param_name in ["inp_scaling", "input_density", "dt", "gamma", "epsilon", "rho"]:
        top_vals = [r["params"].get(param_name, 0) for r in top10]
        all_vals = [r["params"].get(param_name, 0) for r in final_results]
        print(f"  {param_name:>15}: top10={np.mean(top_vals):.4f}±{np.std(top_vals):.4f}  "
              f"all={np.mean(all_vals):.4f}±{np.std(all_vals):.4f}")
    
    # Regularization analysis
    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in READOUT_C_VALUES:
        c_results = [r for r in final_results if r.get("readout_C") == C_val]
        if c_results:
            c_accs = [r["mean_test_acc"] for r in c_results]
            c_gaps = [r["overfit_gap"] for r in c_results]
            print(f"  C={C_val:<6}: n={len(c_results)}, "
                  f"test={np.mean(c_accs):.2f}±{np.std(c_accs):.2f}%, "
                  f"gap={np.mean(c_gaps):.1f}%")

else:
    print("\n⚠️  No successful configurations!")
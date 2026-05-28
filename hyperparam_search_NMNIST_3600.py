"""
Hyperparameter search for Spiking RON on N-MNIST at N_hid=3600.

Key fix vs previous version
-----------------------------
The original search called nMNIST_spiking_ron.py as a subprocess for each
(config, seed) pair. Each subprocess re-loaded and re-binned the full N-MNIST
dataset from raw events using tonic (60k train + 10k test samples, 30 time
bins). With 40 configs x 3 seeds = 120 subprocess calls, and each call taking
~10-15 min for data loading alone, the job timed out before most configs ran.

This version runs entirely in-process:
  1. Load the dataset ONCE into pre-binned tensors (DiskCachedDataset).
  2. For each config, build the reservoir and extract features in-process.
  3. Fit logistic regression on the features.
  4. No subprocess overhead, no repeated tonic cache reads.

This makes each config take ~2-4 min instead of ~15+ min.

Fixed params: dt, gamma, gamma_range, epsilon, epsilon_range (from 512-neuron search)
Searched params: inp_scaling, input_density, rho, theta_lif, theta_rf,
                 connectivity_lif2hrf, readout_C
readout_mode: 'final' for search (3600 features), re-run best with rms_std_final

RNG fix (v3)
------------
set_seed(seed) inside the inner loop resets global random/numpy state, which
was corrupting sample_params() calls in subsequent outer-loop iterations —
causing configs 1..N to all sample identical parameters.

Fix: use dedicated Random/RandomState instances for config sampling, isolated
from the torch/numpy seeds used for model initialisation. All configs are
pre-generated before any set_seed() call.

Resume support
--------------
If summary_intermediate.json exists, completed config_ids are skipped and
their results are re-loaded, so a timed-out job can be continued by simply
re-submitting the same SLURM script.
"""

import os
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

import tonic
import tonic.transforms as tonic_transforms
from tonic import DiskCachedDataset

from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse


# =============================================================================
# Fixed oscillator parameters (from 512-neuron best config)
# =============================================================================

FIXED = {
    "dt":                   0.1092124883046145,
    "gamma":                0.10895240475386166,
    "gamma_range":          0.36156780786060716,
    "epsilon":              0.02076624190152689,
    "epsilon_range":        0.08275192805570303,
    "tau_filter":           20.0,
    "spatial_factor":       2,
    "num_steps":            30,
    "readout_mode":         "final",
    "connectivity_hrf2lif": 1.0,
}

# =============================================================================
# Search space
# =============================================================================

SEARCH_SPACE = {
    "inp_scaling":   (0.02,  0.5,  "log"),
    "input_density": (0.01,  0.2,  "log"),
    "rho":           (0.8,   1.6,  "linear"),
    "theta_lif":     (0.05,  0.5,  "log"),
    "theta_rf":      (0.01,  0.15, "log"),
}

CONNECTIVITY_OPTIONS = [0.1, 0.2, 0.5, 1.0]
CONNECTIVITY_WEIGHTS = [0.15, 0.40, 0.25, 0.20]

READOUT_C_VALUES  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
READOUT_C_WEIGHTS = [0.05,  0.10,  0.25, 0.30, 0.20, 0.10]

# =============================================================================
# Search settings
# =============================================================================

N_SAMPLES   = 40
N_SEEDS     = 2
N_HID       = 3600
DATA_DIR    = "data/NMNIST"
RESULTS_DIR = Path("hyperparam_search_NMNIST_nhid3600")
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed):
    """Sets global RNG state for model initialisation. Do NOT call this before
    config sampling — use the dedicated rng/np_rng instances instead."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def sample_params(rng, np_rng):
    """Sample a config using isolated RNG instances that are never reset by
    set_seed(), so model seeds cannot corrupt the sampling sequence."""
    params = dict(FIXED)
    for key, (lo, hi, scale) in SEARCH_SPACE.items():
        if scale == "log":
            params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            params[key] = float(rng.uniform(lo, hi))
    params["connectivity_lif2hrf"] = rng.choices(
        CONNECTIVITY_OPTIONS, weights=CONNECTIVITY_WEIGHTS)[0]
    params["readout_C"] = rng.choices(
        READOUT_C_VALUES, weights=READOUT_C_WEIGHTS)[0]
    return params


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    if input_density >= 1.0:
        return n_inp * n_hid
    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0
    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale
    return int(mask.sum().item())


def extract_features(model, loader, device):
    model.eval()
    feats, labels_all = [], []
    r_hrf_l, r_lif_l  = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, ncols=80, desc="  extract", leave=False):
            x = x.to(device)
            features, r = model(x)
            feats.append(features.cpu())
            r_hrf_l.append(r["r_hrf"])
            r_lif_l.append(r["r_lif"])
            labels_all.append(y)
    return (torch.cat(feats, dim=0).numpy(),
            torch.cat(labels_all, dim=0).numpy(),
            torch.stack(r_hrf_l).mean().item(),
            torch.stack(r_lif_l).mean().item())


def save_results(all_results, failed_configs, path_intermediate):
    """Atomically save all results seen so far (sorted by test acc)."""
    sorted_results = sorted(all_results, key=lambda x: x["mean_test_acc"], reverse=True)
    tmp = str(path_intermediate) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted_results, f, indent=2)
    os.replace(tmp, path_intermediate)  # atomic on POSIX
    # Also keep a live failed list
    failed_path = RESULTS_DIR / "failed_configs.json"
    with open(failed_path, "w") as f:
        json.dump(failed_configs, f, indent=2)


# =============================================================================
# Data loading — done ONCE, reused for every config
# =============================================================================

def build_loaders(data_dir, num_steps, spatial_factor, batch=256):
    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    H, W, C     = sensor_size[1], sensor_size[0], sensor_size[2]
    H_ds        = H // spatial_factor
    W_ds        = W // spatial_factor
    n_inp       = C * H_ds * W_ds

    frame_transform = tonic_transforms.ToFrame(
        sensor_size=sensor_size, n_time_bins=num_steps
    )

    def collate_fn(batch_data):
        xs, ys = [], []
        for frames, label in batch_data:
            t = torch.tensor(frames, dtype=torch.float32)
            if spatial_factor > 1:
                T_ = t.size(0)
                t  = t.view(T_*C, 1, H, W)
                t  = F.avg_pool2d(t, kernel_size=spatial_factor, stride=spatial_factor)
                t  = t.view(T_, C, H_ds, W_ds)
            t = t.reshape(t.size(0), -1)
            t = (t > 0).float()
            xs.append(t); ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    os.makedirs(data_dir, exist_ok=True)
    cache_tr = os.path.join(data_dir, f'cache_train_T{num_steps}_sf{spatial_factor}')
    cache_te = os.path.join(data_dir, f'cache_test_T{num_steps}_sf{spatial_factor}')

    train_ds = DiskCachedDataset(
        tonic.datasets.NMNIST(save_to=data_dir, train=True,  transform=frame_transform),
        cache_path=cache_tr)
    test_ds  = DiskCachedDataset(
        tonic.datasets.NMNIST(save_to=data_dir, train=False, transform=frame_transform),
        cache_path=cache_te)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print(f"Dataset loaded: {len(train_ds)} train, {len(test_ds)} test")
    print(f"n_inp={n_inp}  num_steps={num_steps}  spatial_factor={spatial_factor}")
    return train_loader, test_loader, n_inp


# =============================================================================
# Main search
# =============================================================================

def main():
    # ── Dedicated sampling RNGs — NEVER reset by set_seed() ──────────────────
    # These are plain Python/numpy instances, not the global random/np state.
    sampling_rng    = random.Random(42)
    sampling_np_rng = np.random.RandomState(42)

    # Pre-generate ALL configs before any model seed is set, so set_seed()
    # calls inside the inner loop cannot affect the sampling sequence.
    all_configs = [sample_params(sampling_rng, sampling_np_rng)
                   for _ in range(N_SAMPLES)]

    # Save the config plan so a resumed run uses the exact same configs
    configs_path = RESULTS_DIR / "configs.json"
    if not configs_path.exists():
        with open(configs_path, "w") as f:
            json.dump(all_configs, f, indent=2)
        print(f"Saved {N_SAMPLES} sampled configs → {configs_path}")
    else:
        # On resume, reload the original configs to guarantee reproducibility
        with open(configs_path) as f:
            all_configs = json.load(f)
        print(f"Loaded existing config plan from {configs_path}")

    # ── Resume support ────────────────────────────────────────────────────────
    intermediate_path = RESULTS_DIR / "summary_intermediate.json"
    all_results, failed_configs = [], []
    completed_ids = set()

    if intermediate_path.exists():
        with open(intermediate_path) as f:
            all_results = json.load(f)
        completed_ids = {r["config_id"] for r in all_results}
        print(f"\n▶  Resuming: {len(completed_ids)} configs already completed, "
              f"{N_SAMPLES - len(completed_ids)} remaining.")

    # Also reload any previously failed configs so we don't overwrite them
    failed_path = RESULTS_DIR / "failed_configs.json"
    if failed_path.exists():
        with open(failed_path) as f:
            failed_configs = json.load(f)
        failed_ids = {r["config_id"] for r in failed_configs}
        completed_ids |= failed_ids

    # ── Device & meta ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Search: {N_SAMPLES} configs x {N_SEEDS} seeds = "
          f"{N_SAMPLES * N_SEEDS} runs")
    print(f"Fixed: {FIXED}")
    print("=" * 70)

    # Load data ONCE
    print("\n=== Loading N-MNIST (once) ===")
    t0 = time.time()
    train_loader, test_loader, n_inp = build_loaders(
        DATA_DIR, FIXED["num_steps"], FIXED["spatial_factor"]
    )
    print(f"Data ready in {time.time()-t0:.1f}s")

    seq_length = FIXED["num_steps"]

    gamma_range   = FIXED["gamma_range"]
    epsilon_range = FIXED["epsilon_range"]
    gamma_tuple   = (
        max(FIXED["gamma"]   - gamma_range   / 2, 1e-6),
        FIXED["gamma"]   + gamma_range   / 2,
    )
    epsilon_tuple = (
        max(FIXED["epsilon"] - epsilon_range / 2, 1e-6),
        FIXED["epsilon"] + epsilon_range / 2,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, params in enumerate(all_configs):

        if i in completed_ids:
            print(f"\nConfig {i+1}/{N_SAMPLES}: already done, skipping.")
            continue

        print(f"\nConfig {i+1}/{N_SAMPLES}: "
              f"inp={params['inp_scaling']:.3f}  "
              f"dens={params['input_density']:.3f}  "
              f"rho={params['rho']:.3f}  "
              f"th_lif={params['theta_lif']:.4f}  "
              f"th_rf={params['theta_rf']:.4f}  "
              f"lif2hrf={params['connectivity_lif2hrf']}  "
              f"C={params['readout_C']}")

        seed_results = []
        config_ok    = True

        for seed in range(N_SEEDS):
            try:
                # set_seed only affects torch/numpy for model init — sampling
                # RNG (sampling_rng / sampling_np_rng) is untouched.
                set_seed(seed)

                model = spiking_coESN_rescaled_II(
                    n_inp        = n_inp,
                    n_hid        = N_HID,
                    dt           = FIXED["dt"],
                    gamma        = gamma_tuple,
                    epsilon      = epsilon_tuple,
                    rho          = params["rho"],
                    input_scaling= params["inp_scaling"],
                    theta_lif    = params["theta_lif"],
                    theta_rf     = params["theta_rf"],
                    tau_filter   = FIXED["tau_filter"],
                    sparse_lif2hrf       = (params["connectivity_lif2hrf"] < 1.0),
                    connectivity_lif2hrf = params["connectivity_lif2hrf"],
                    sparse_hrf2lif       = False,
                    connectivity_hrf2lif = 1.0,
                    device       = device,
                    readout_mode = FIXED["readout_mode"],
                ).to(device)

                apply_sparse_input_projection(
                    model, params["input_density"], n_inp, N_HID, device
                )

                t0 = time.time()
                train_feats, train_labels, r_hrf, r_lif = extract_features(
                    model, train_loader, device
                )
                test_feats,  test_labels,  _,     _     = extract_features(
                    model, test_loader,  device
                )
                t_extract = time.time() - t0

                scaler      = preprocessing.StandardScaler().fit(train_feats)
                train_feats = scaler.transform(train_feats)
                test_feats  = scaler.transform(test_feats)

                t0 = time.time()
                clf = LogisticRegression(
                    max_iter=1000, verbose=0, n_jobs=1,
                    C=params["readout_C"],
                ).fit(train_feats, train_labels)
                t_lr = time.time() - t0

                train_acc = clf.score(train_feats, train_labels) * 100
                test_acc  = clf.score(test_feats,  test_labels)  * 100

                r_flag = ""
                if r_hrf > 0.4:   r_flag = " ⚠️ HRF_SAT"
                if r_hrf < 0.005: r_flag = " ⚠️ HRF_SILENT"

                print(f"  seed={seed}: test={test_acc:.2f}%  "
                      f"train={train_acc:.2f}%  "
                      f"r_hrf={r_hrf:.4f}{r_flag}  "
                      f"extract={t_extract:.0f}s  lr={t_lr:.0f}s")

                snn_energy = estimate_snn_energy_sparse(
                    r_hrf=r_hrf, r_lif=r_lif,
                    n_hid=N_HID, T=seq_length,
                    lif2hrf_connections=model.n_lif2hrf_connections,
                    include_lif=True,
                )

                seed_results.append({
                    "seed":           seed,
                    "test_acc_mean":  test_acc,
                    "train_acc_mean": train_acc,
                    "r_hrf":          r_hrf,
                    "r_lif":          r_lif,
                    "energy_J":       snn_energy["Energy_J"],
                })

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  seed={seed}: FAILED — {e}")
                config_ok = False
                break

        if config_ok and len(seed_results) == N_SEEDS:
            test_accs  = [r["test_acc_mean"]  for r in seed_results]
            train_accs = [r["train_acc_mean"] for r in seed_results]
            mean_test  = float(np.mean(test_accs))
            std_test   = float(np.std(test_accs))
            print(f"  ✅ mean test: {mean_test:.2f} ± {std_test:.2f}%  "
                  f"gap: {np.mean(train_accs)-mean_test:.1f}%")
            all_results.append({
                "config_id":            i,
                "mean_test_acc":        mean_test,
                "std_test_acc":         std_test,
                "mean_train_acc":       float(np.mean(train_accs)),
                "overfit_gap":          float(np.mean(train_accs) - mean_test),
                "connectivity_lif2hrf": params["connectivity_lif2hrf"],
                "readout_C":            params["readout_C"],
                "params":               params,
                "seed_results":         seed_results,
            })
        else:
            failed_configs.append({"config_id": i, "params": params})

        # ── Save after EVERY config ───────────────────────────────────────────
        save_results(all_results, failed_configs, intermediate_path)
        if all_results:
            best_so_far = max(all_results, key=lambda x: x["mean_test_acc"])
            print(f"  💾 Saved  (best so far: config {best_so_far['config_id']} "
                  f"@ {best_so_far['mean_test_acc']:.2f}%)")

    # ── Final summary ─────────────────────────────────────────────────────────
    all_results.sort(key=lambda x: x["mean_test_acc"], reverse=True)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ {len(all_results)}/{N_SAMPLES} configs completed  "
          f"❌ {len(failed_configs)}/{N_SAMPLES} failed")
    print("=" * 70)

    if not all_results:
        print("No successful configs."); return

    print(f"\n🏆 TOP 10:")
    print(f"{'Rk':<4} {'Test%':<20} {'Gap':<7} {'lif2hrf':<9} {'C':<7} "
          f"{'inp':<8} {'dens':<8} {'rho':<6} {'th_lif':<8} {'th_rf'}")
    print("-" * 90)
    for rank, r in enumerate(all_results[:10], 1):
        p = r["params"]
        print(f"{rank:<4}"
              f"{r['mean_test_acc']:.2f}±{r['std_test_acc']:.2f}      "
              f"{r['overfit_gap']:.1f}%  "
              f"{r['connectivity_lif2hrf']:<9}"
              f"{r['readout_C']:<7}"
              f"{p['inp_scaling']:<8.3f}"
              f"{p['input_density']:<8.3f}"
              f"{p['rho']:<6.3f}"
              f"{p['theta_lif']:<8.4f}"
              f"{p['theta_rf']:.4f}")

    best = all_results[0]
    bp   = best["params"]
    print(f"\n📋  Best config (test={best['mean_test_acc']:.2f}±"
          f"{best['std_test_acc']:.2f}%):")
    print(f"  inp_scaling          = {bp['inp_scaling']:.6f}")
    print(f"  input_density        = {bp['input_density']:.6f}")
    print(f"  rho                  = {bp['rho']:.6f}")
    print(f"  theta_lif            = {bp['theta_lif']:.6f}")
    print(f"  theta_rf             = {bp['theta_rf']:.6f}")
    print(f"  connectivity_lif2hrf = {bp['connectivity_lif2hrf']}")
    print(f"  readout_C            = {best['readout_C']}")
    print(f"\nNOTE: readout_mode=final was used for search.")
    print(f"Re-run best config with --readout_mode rms_std_final for final result.")


if __name__ == "__main__":
    main()

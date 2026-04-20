"""
Representational analysis: HRF reservoir (s-RON) vs LIF reservoir (LIF-RC)

Analyses:
  1. CKA                  - class discriminability from reservoir states
  2. Effective Dim        - Participation Ratio, LUD, ASE (Renyi entropy)
  3. Frequency Selectivity - distribution of preferred frequencies + Q-factor
  4. Memory Capacity      - classical Jaeger MC (linear + nonlinear)
  5. Linear Probe         - logistic regression accuracy on reservoir features

Each analysis targets a different facet of reservoir quality, following the
richness metrics in Lukoševičius & Jaeger (2009, Sec. 6.1) and
Gallicchio & Micheli (2021) on deep reservoir richness.

Usage:
    # Run everything on one dataset:
    python analyze_representations.py --dataset sMNIST --analysis all

    # Just the new analyses:
    python analyze_representations.py --dataset sMNIST --analysis mc ase_lud probe q_factor

    # Dataset-independent only (fastest, reviewer-proof core):
    python analyze_representations.py --dataset sMNIST --analysis mc
"""

import argparse
import os
import json
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils_aurora import spiking_coESN_rescaled_II, spiking_LIF_reservoir
from esn import spectral_norm_scaling
from utils          import get_FordA_data, get_mnist_data
from ucr_data_utils import get_SHD_data

try:
    import tonic
    import tonic.transforms as tonic_transforms
    from tonic import DiskCachedDataset
    from torch.utils.data import DataLoader
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False


# =============================================================================
# Best configs (UNCHANGED from your original file)
# =============================================================================

HRF_CONFIGS = {
    'fordA': dict(
        n_hid=800, dt=0.051, rho=0.75, inp_scaling=0.6247,
        gamma=(7.0124 - 3.01/2., 7.0124 + 3.01/2.),
        epsilon=(0.1528 - 0.419/2., 0.1528 + 0.419/2.),
        theta_lif=0.0824, theta_rf=0.0010, tau_filter=6.1,
        connectivity_lif2hrf=1.0, connectivity_hrf2lif=1.0,
        input_density=1.0, num_steps=500, n_inp=1, readout_mode='final',
    ),
    'sMNIST': dict(
        n_hid=800, dt=0.042, rho=0.99, inp_scaling=2.0,
        gamma=(2.7 - 2.0/2., 2.7 + 2.0/2.),
        epsilon=(0.08 - 1.0/2., 0.08 + 1.0/2.),
        theta_lif=0.05, theta_rf=0.005, tau_filter=20.0,
        connectivity_lif2hrf=1.0, connectivity_hrf2lif=1.0,
        input_density=1.0, num_steps=784, n_inp=1, readout_mode='final',
    ),
    'shd': dict(
        n_hid=3000, dt=0.223, rho=1.16, inp_scaling=0.23,
        gamma=(0.036 - 0.268/2., 0.036 + 0.268/2.),
        epsilon=(0.06 - 0.063/2., 0.06 + 0.063/2.),
        theta_lif=1.0, theta_rf=0.013, tau_filter=20.0,
        connectivity_lif2hrf=1.0, connectivity_hrf2lif=1.0,
        input_density=0.036, num_steps=250, n_inp=700, readout_mode='final',
    ),
    'dvs_gesture': dict(
        n_hid=3000, dt=0.259, rho=1.581, inp_scaling=0.1129,
        gamma=(0.0456 - 0.1304/2., 0.0456 + 0.1304/2.),
        epsilon=(0.0354 - 0.0989/2., 0.0354 + 0.0989/2.),
        theta_lif=2.9678, theta_rf=0.03628, tau_filter=20.0,
        connectivity_lif2hrf=1.0, connectivity_hrf2lif=1.0,
        input_density=0.0306, num_steps=200, n_inp=2048, readout_mode='final',
    ),
}

LIF_CONFIGS = {
    'fordA': dict(
        n_hid=800, dt=0.068, rho=0.875, inp_scaling=4.0365,
        tau_m=36.68, tau_m_range=25.85,
        theta_res=0.15659, theta_res_range=0.01254,
        theta_lif=0.1458, tau_filter=20.0,
        connectivity_lif2res=1.0, connectivity_res2enc=1.0,
        input_density=1.0, num_steps=500, n_inp=1, readout_mode='final',
    ),
    'sMNIST': dict(
        n_hid=800, dt=0.034, rho=0.976, inp_scaling=0.8471,
        tau_m=36.19, tau_m_range=3.60,
        theta_res=0.00254, theta_res_range=0.01180,
        theta_lif=0.1507, tau_filter=20.0,
        connectivity_lif2res=1.0, connectivity_res2enc=1.0,
        input_density=1.0, num_steps=784, n_inp=1, readout_mode='final',
    ),
    'shd': dict(
        n_hid=3000, dt=0.084, rho=1.172, inp_scaling=0.1962,
        tau_m=13.01, tau_m_range=2.38,
        theta_res=0.09598, theta_res_range=0.00375,
        theta_lif=0.4188, tau_filter=20.0,
        connectivity_lif2res=1.0, connectivity_res2enc=1.0,
        input_density=0.036, num_steps=250, n_inp=700, readout_mode='final',
    ),
    'dvs_gesture': dict(
        n_hid=3000, dt=0.056, rho=0.866, inp_scaling=0.0310,
        tau_m=58.59, tau_m_range=23.17,
        theta_res=0.02224, theta_res_range=0.00194,
        theta_lif=0.5327, tau_filter=20.0,
        connectivity_lif2res=1.0, connectivity_res2enc=1.0,
        input_density=0.0306, num_steps=200, n_inp=2048, readout_mode='final',
    ),
}

FREQ_SWEEP_DEFAULTS = {
    'fordA':       (0.01,  2.0,  6000),
    'sMNIST':      (0.01,  2.0,  6000),
    'shd':         (0.001, 2.0, 50000),
    'dvs_gesture': (0.001, 2.0, 50000),
}

# Memory capacity defaults (cheap: ~30s-2min per model per dataset for N=800,
# longer for SHD/DVS because n_hid=3000). Tuned for clean signal with low cost.
MC_DEFAULTS = dict(
    T_mc=4000,         # total noise sequence length
    washout=500,       # discard initial transient
    k_max=150,         # max delay to probe
    n_noise_seeds=1,   # how many noise sequences to average MC over
)


# =============================================================================
# Model builders (UNCHANGED)
# =============================================================================

def build_hrf_model(dataset, device, seed=42):
    cfg = HRF_CONFIGS[dataset]
    torch.manual_seed(seed)
    model = spiking_coESN_rescaled_II(
        n_inp=cfg['n_inp'], n_hid=cfg['n_hid'], dt=cfg['dt'],
        gamma=cfg['gamma'], epsilon=cfg['epsilon'], rho=cfg['rho'],
        input_scaling=cfg['inp_scaling'],
        theta_lif=cfg['theta_lif'], theta_rf=cfg['theta_rf'],
        tau_filter=cfg['tau_filter'],
        sparse_lif2hrf=(cfg['connectivity_lif2hrf'] < 1.0),
        connectivity_lif2hrf=cfg['connectivity_lif2hrf'],
        sparse_hrf2lif=(cfg['connectivity_hrf2lif'] < 1.0),
        connectivity_hrf2lif=cfg['connectivity_hrf2lif'],
        device=device, readout_mode=cfg['readout_mode'],
    ).to(device)
    return model, cfg


def build_lif_model(dataset, device, seed=42):
    cfg = LIF_CONFIGS[dataset]
    torch.manual_seed(seed)
    model = spiking_LIF_reservoir(
        n_inp=cfg['n_inp'], n_hid=cfg['n_hid'], dt=cfg['dt'],
        tau_m=cfg['tau_m'], tau_m_range=cfg['tau_m_range'],
        theta_res=cfg['theta_res'], theta_res_range=cfg['theta_res_range'],
        rho=cfg['rho'], input_scaling=cfg['inp_scaling'],
        theta_lif=cfg['theta_lif'], tau_filter=cfg['tau_filter'],
        sparse_lif2res=(cfg['connectivity_lif2res'] < 1.0),
        connectivity_lif2res=cfg['connectivity_lif2res'],
        sparse_res2enc=(cfg['connectivity_res2enc'] < 1.0),
        connectivity_res2enc=cfg['connectivity_res2enc'],
        device=device, readout_mode=cfg['readout_mode'],
    ).to(device)
    return model, cfg


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    if input_density >= 1.0:
        return
    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0
    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale


# =============================================================================
# Dataset loaders
# =============================================================================

def load_dataset(dataset, args, device, want_train=False):
    """
    Returns (test_loader, n_inp, needs_reshape, train_loader_or_None).

    For the linear-probe analysis we need a train loader too. Not all of your
    loaders expose one cleanly — for those cases we return None and the probe
    will fall back to 5-fold CV on the test set.
    """
    if dataset == 'fordA':
        # get_FordA_data returns numpy arrays, not a clean (train, val, test).
        # We reuse whatever your existing call returns for test; probe will use CV.
        train_loader, _, test_loader = get_FordA_data(64, 120, whole_train=True)
        return test_loader, 1, False, (train_loader if want_train else None)

    elif dataset == 'sMNIST':
        train_loader, _, test_loader = get_mnist_data(256, 100)
        return test_loader, 1, True, (train_loader if want_train else None)

    elif dataset == 'shd':
        cfg = HRF_CONFIGS['shd']
        train_loader, _, test_loader = get_SHD_data(
            batch_train=128, batch_test=256,
            data_dir=args.data_dir,
            num_steps=cfg['num_steps'], max_time=1.4
        )
        return test_loader, 700, False, (train_loader if want_train else None)

    elif dataset == 'dvs_gesture':
        assert TONIC_AVAILABLE
        cfg = HRF_CONFIGS['dvs_gesture']
        sf = 4
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        H, W, C = sensor_size[1], sensor_size[0], sensor_size[2]
        H_ds, W_ds = H // sf, W // sf
        n_inp = C * H_ds * W_ds

        frame_transform = tonic_transforms.ToFrame(
            sensor_size=sensor_size, n_time_bins=cfg['num_steps']
        )

        def collate_fn(batch):
            xs, ys = [], []
            for frames, label in batch:
                t = torch.tensor(frames, dtype=torch.float32)
                T_ = t.size(0)
                t = t.view(T_ * C, 1, H, W)
                t = F.avg_pool2d(t, kernel_size=sf, stride=sf)
                t = t.view(T_, C, H_ds, W_ds).reshape(T_, -1)
                t = (t > 0).float()
                xs.append(t)
                ys.append(label)
            return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

        os.makedirs(args.data_dir, exist_ok=True)
        test_ds_raw = tonic.datasets.DVSGesture(
            save_to=args.data_dir, train=False, transform=frame_transform)
        cache = os.path.join(args.data_dir,
                             f'cache_test_T{cfg["num_steps"]}_sf{sf}')
        test_ds = DiskCachedDataset(test_ds_raw, cache_path=cache)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)

        train_loader = None
        if want_train:
            train_ds_raw = tonic.datasets.DVSGesture(
                save_to=args.data_dir, train=True, transform=frame_transform)
            cache_tr = os.path.join(args.data_dir,
                                    f'cache_train_T{cfg["num_steps"]}_sf{sf}')
            train_ds = DiskCachedDataset(train_ds_raw, cache_path=cache_tr)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      collate_fn=collate_fn, num_workers=4)

        return test_loader, n_inp, False, train_loader


# =============================================================================
# Shared helper: extract reservoir states for a loader
# =============================================================================

def extract_states(loader, model, device, needs_reshape, max_samples=500,
                   feature='mean'):
    """
    feature: 'mean' | 'rms' | 'final' — the temporal statistic of hy used per
    sample. Returns (features_np, labels_np).
    """
    assert feature in ('mean', 'rms', 'final')
    model.eval()
    all_feats, all_labels = [], []
    n_collected = 0

    original_mode = model.readout_mode
    if feature in ('mean', 'final'):
        model.readout_mode = feature
    else:  # rms — use rms_std_final and slice first n_hid dims
        model.readout_mode = 'rms_std_final'

    with torch.no_grad():
        for x, y in loader:
            if n_collected >= max_samples:
                break
            x = x.to(device)
            if needs_reshape:
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
            feats, _ = model(x)
            if feature == 'rms':
                feats = feats[:, :model.n_hid]
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy())
            n_collected += x.shape[0]

    model.readout_mode = original_mode

    feats  = np.concatenate(all_feats,  axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples].ravel()
    return feats, labels


def extract_temporal_states(loader, model, device, needs_reshape,
                             max_samples=200, subsample_t=1):
    """
    Collect reservoir activations at every timestep of every sample:
      returns (n_samples * T_effective, n_hid) numpy array of continuous states.

    This is the richness-appropriate representation — per-sample pooled
    features massively undercount the dynamical dimensionality. Richness
    metrics (PR, LUD, ASE) should be computed on the temporal trajectory
    (cf. Gallicchio & Micheli 2021, where metrics are computed over time
    for a single long sequence).

    subsample_t: keep every k-th timestep to control memory (default 1, keep all).
    """
    model.eval()
    all_states = []
    n_collected = 0
    n_hid = model.n_hid

    with torch.no_grad():
        for x, y in loader:
            if n_collected >= max_samples:
                break
            x = x.to(device)
            if needs_reshape:
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)

            B, T, _ = x.shape
            states_t = torch.zeros(B, T, n_hid, device=device)

            if isinstance(model, spiking_coESN_rescaled_II):
                hy = torch.zeros(B, n_hid, device=device)
                hz = torch.zeros(B, n_hid, device=device)
                ref = torch.zeros(B, n_hid, device=device)
                s  = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    hy, hz, s, ref, lif_v, _ = model.bio_cell(
                        x[:, t], hy, hz, lif_v, s, ref_period=ref)
                    states_t[:, t] = hy
            else:  # spiking_LIF_reservoir
                res_v = torch.zeros(B, n_hid, device=device)
                res_s = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    res_v, res_s, lif_v, _ = model.bio_cell(
                        x[:, t], res_v, res_s, lif_v)
                    states_t[:, t] = res_v

            # Discard short initial transient (10% of T) before stacking
            washout = max(1, T // 10)
            states_np = states_t[:, washout::subsample_t].cpu().numpy()
            # shape: (B, T_kept, n_hid) → (B * T_kept, n_hid)
            states_np = states_np.reshape(-1, n_hid)
            all_states.append(states_np)
            n_collected += B

    return np.concatenate(all_states, axis=0)


# =============================================================================
# Analysis 1: CKA (unchanged, just slightly cleaned up)
# =============================================================================

def centered_kernel_alignment(X, Y_kernel):
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_X  = X @ X.T
    K_Xc = H @ K_X @ H
    K_Yc = H @ Y_kernel @ H
    hsic_xy = np.sum(K_Xc * K_Yc)
    hsic_xx = np.sum(K_Xc * K_Xc)
    hsic_yy = np.sum(K_Yc * K_Yc)
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def run_cka(hrf_model, lif_model, loader, device, needs_reshape,
            dataset, out_dir, max_samples=500, cka_feature='mean'):
    print(f"\n--- Analysis 1: CKA (feature='{cka_feature}') ---")

    hrf_feats, labels = extract_states(loader, hrf_model, device,
                                       needs_reshape, max_samples, cka_feature)
    lif_feats, _      = extract_states(loader, lif_model, device,
                                       needs_reshape, max_samples, cka_feature)

    n = min(len(labels), len(hrf_feats), len(lif_feats))
    hrf_feats, lif_feats, labels = hrf_feats[:n], lif_feats[:n], labels[:n]
    Y = (labels[:, None] == labels[None, :]).astype(float)

    hrf_norm = hrf_feats / (np.linalg.norm(hrf_feats, axis=1, keepdims=True) + 1e-8)
    lif_norm = lif_feats / (np.linalg.norm(lif_feats, axis=1, keepdims=True) + 1e-8)

    cka_hrf = centered_kernel_alignment(hrf_norm, Y)
    cka_lif = centered_kernel_alignment(lif_norm, Y)

    print(f"  n={n} samples | HRF: {cka_hrf:.4f} | LIF: {cka_lif:.4f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(['s-RON\n(HRF)', 'LIF-RC'], [cka_hrf, cka_lif],
                  color=['#2166AC', '#D6604D'], width=0.5, edgecolor='black')
    ax.set_ylabel('CKA with class labels')
    ax.set_title(f'Linear CKA — {dataset} (feature: {cka_feature})')
    ax.set_ylim(0, min(1.0, max(cka_hrf, cka_lif) * 1.3))
    for bar, val in zip(bars, [cka_hrf, cka_lif]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cka_{dataset}_{cka_feature}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {'cka_hrf': cka_hrf, 'cka_lif': cka_lif, 'cka_feature': cka_feature}


# =============================================================================
# Analysis 2: Effective Dim (PR) + LUD + ASE  [EXPANDED]
# =============================================================================

def participation_ratio(X):
    """Effective dimensionality. X: (n_samples, n_features)."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / max(X_centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    sum1 = eigvals.sum()
    sum2 = (eigvals ** 2).sum()
    if sum2 < 1e-12:
        return 0.0
    return float(sum1 ** 2 / sum2)


def linearly_uncoupled_dynamics(X, theta=0.9):
    """
    LUD (Gallicchio & Micheli 2021, Eq. 8): number of PCs needed to explain
    at least `theta` of the normalized reservoir variability, where
    R_j = sigma_j / sum(sigma_k) is normalized SINGULAR-value relevance.

    X: (n_samples, n_features).
    """
    # Use singular values directly (more stable than eigvals of covariance)
    X_c = X - X.mean(axis=0)
    try:
        s = np.linalg.svd(X_c, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0
    s = np.maximum(s, 0)
    total = s.sum()
    if total < 1e-12:
        return 0
    normalized = s / total
    cumsum = np.cumsum(normalized)
    # smallest d such that cumsum[d-1] >= theta
    d = int(np.argmax(cumsum >= theta)) + 1
    return d


def average_state_entropy(X, kernel_scale=0.3):
    """
    Rényi quadratic entropy estimator (Ozturk et al. 2007;
    Gallicchio & Micheli 2021 Eq. 5-6). Higher = more diverse activations.

    For each sample x_i in R^d (here x_i is a reservoir state at one timestep,
    or a per-sample feature vector), computes -log(1/d^2 * sum_{j,k} K(x_ij, x_ik))
    where K is a Gaussian kernel with width = kernel_scale * std of activations,
    and averages across samples.

    X: (n_samples, n_features). We treat each sample as a state vector.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    # Kernel width per-sample would be expensive; use global std as in
    # Ozturk et al. footnote.
    sigma = kernel_scale * (X.std() + 1e-8)

    entropies = np.zeros(n_samples)
    for i in range(n_samples):
        x = X[i]  # (n_features,)
        # Pairwise squared distances between components
        # diff[j,k] = x[j] - x[k]; |diff|^2 has shape (n_features, n_features)
        # K_{jk} = exp(-|x_j - x_k|^2 / (2 sigma^2))
        # Using broadcasting:
        diff = x[:, None] - x[None, :]
        K = np.exp(-(diff ** 2) / (2.0 * sigma ** 2 + 1e-12))
        kernel_sum = K.sum() / (n_features ** 2)
        entropies[i] = -np.log(kernel_sum + 1e-12)
    return float(entropies.mean())


def run_eff_dim_richness(hrf_model, lif_model, loader, device, needs_reshape,
                         dataset, out_dir, feature='mean', max_samples=1000,
                         mode='temporal', temporal_n_samples=100,
                         temporal_subsample=2, ase_subset=2000):
    """
    Computes PR, LUD, and ASE for both reservoirs.

    mode:
      'temporal' (default, recommended): collect reservoir activations at
         every timestep across `temporal_n_samples` sequences, giving an
         (N, n_hid) matrix with N = temporal_n_samples * T_effective. This
         matches how richness metrics are computed in Gallicchio & Micheli
         (2021) — over the temporal trajectory, not one-vector-per-sample.
      'pooled': use per-sample pooled features (original behavior). Useful
         as a sanity-check comparison but systematically undercounts
         dimensionality.

    temporal_subsample: keep every k-th timestep (memory control for long
         sequences like sMNIST with T=784).
    ase_subset: ASE is O(N * n_hid^2); randomly subsample this many state
         vectors for ASE estimation to keep it tractable.
    """
    print(f"\n--- Analysis 2: Effective Dim + LUD + ASE "
          f"(mode='{mode}', feature='{feature}') ---")

    if mode == 'temporal':
        print(f"  Collecting temporal states "
              f"({temporal_n_samples} seqs, every {temporal_subsample} steps)...")
        X_hrf = extract_temporal_states(
            loader, hrf_model, device, needs_reshape,
            max_samples=temporal_n_samples, subsample_t=temporal_subsample)
        X_lif = extract_temporal_states(
            loader, lif_model, device, needs_reshape,
            max_samples=temporal_n_samples, subsample_t=temporal_subsample)
        print(f"  State matrix shapes — HRF: {X_hrf.shape}  LIF: {X_lif.shape}")
    elif mode == 'pooled':
        X_hrf, _ = extract_states(loader, hrf_model, device, needs_reshape,
                                   max_samples=max_samples, feature=feature)
        X_lif, _ = extract_states(loader, lif_model, device, needs_reshape,
                                   max_samples=max_samples, feature=feature)
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'temporal' or 'pooled')")

    pr_hrf  = participation_ratio(X_hrf)
    pr_lif  = participation_ratio(X_lif)
    lud_hrf = linearly_uncoupled_dynamics(X_hrf, theta=0.9)
    lud_lif = linearly_uncoupled_dynamics(X_lif, theta=0.9)

    # ASE is expensive: subsample state vectors for estimation
    def _ase_on_subset(X, k):
        if X.shape[0] > k:
            idx = np.random.RandomState(0).choice(X.shape[0], size=k, replace=False)
            X = X[idx]
        return average_state_entropy(X)

    print(f"  Computing ASE (subset of {ase_subset} states)...")
    ase_hrf = _ase_on_subset(X_hrf, ase_subset)
    ase_lif = _ase_on_subset(X_lif, ase_subset)

    n_hrf = X_hrf.shape[1]
    n_lif = X_lif.shape[1]

    print(f"  PR    HRF: {pr_hrf:7.1f}/{n_hrf} ({pr_hrf/n_hrf*100:5.1f}%)  |  "
          f"LIF: {pr_lif:7.1f}/{n_lif} ({pr_lif/n_lif*100:5.1f}%)")
    print(f"  LUD   HRF: {lud_hrf:4d}  |  LIF: {lud_lif:4d}  (PCs for 90% var)")
    print(f"  ASE   HRF: {ase_hrf:7.3f}  |  LIF: {ase_lif:7.3f}  (Rényi Q-entropy)")

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    metrics = [('Participation Ratio', pr_hrf, pr_lif),
               ('LUD (90% var)',       lud_hrf, lud_lif),
               ('Avg State Entropy',   ase_hrf, ase_lif)]
    for ax, (name, v_hrf, v_lif) in zip(axes, metrics):
        bars = ax.bar(['s-RON\n(HRF)', 'LIF-RC'], [v_hrf, v_lif],
                      color=['#2166AC', '#D6604D'], width=0.5, edgecolor='black')
        ax.set_title(f'{name}\n{dataset}', fontsize=11)
        for bar, val in zip(bars, [v_hrf, v_lif]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'richness_summary_{dataset}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'pr_hrf': pr_hrf, 'pr_lif': pr_lif,
        'pr_frac_hrf': pr_hrf / n_hrf, 'pr_frac_lif': pr_lif / n_lif,
        'lud_hrf': lud_hrf, 'lud_lif': lud_lif,
        'ase_hrf': ase_hrf, 'ase_lif': ase_lif,
        'n_hid_hrf': n_hrf, 'n_hid_lif': n_lif,
        'mode': mode,
        'feature': feature,
    }


# =============================================================================
# Analysis 3: Frequency selectivity + Q-factor  [EXPANDED]
# =============================================================================

def frequency_sweep(model, model_type, dt, n_inp, n_hid, device,
                    freqs, T=5000, batch_size=1):
    """
    Returns:
      preferred_freq   : (n_hid,) preferred input freq per neuron
      response_matrix  : (n_freqs, n_hid) response power per (freq, neuron)
    """
    model.eval()
    t_vec = torch.arange(T, dtype=torch.float32, device=device)
    responses = np.zeros((len(freqs), n_hid))

    with torch.no_grad():
        for fi, freq in enumerate(freqs):
            sine = torch.sin(2 * np.pi * freq * t_vec * dt)
            x = sine.unsqueeze(0).unsqueeze(2).expand(batch_size, T, n_inp)
            B = batch_size
            states = torch.zeros(B, T, n_hid, device=device)

            if model_type == 'hrf':
                hy = torch.zeros(B, n_hid, device=device)
                hz = torch.zeros(B, n_hid, device=device)
                ref_period = torch.zeros(B, n_hid, device=device)
                s  = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    hy, hz, s, ref_period, lif_v, _ = model.bio_cell(
                        x[:, t], hy, hz, lif_v, s, ref_period=ref_period)
                    states[:, t, :] = hy
            elif model_type == 'lif':
                res_v = torch.zeros(B, n_hid, device=device)
                res_s = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    res_v, res_s, lif_v, _ = model.bio_cell(
                        x[:, t], res_v, res_s, lif_v)
                    states[:, t, :] = res_v

            states_np = states.mean(0).cpu().numpy()
            steady = states_np[T//2:, :]
            fft_vals = np.fft.rfft(steady, axis=0)
            power    = np.abs(fft_vals) ** 2
            fft_freqs = np.fft.rfftfreq(steady.shape[0], d=dt)
            bin_idx  = np.argmin(np.abs(fft_freqs - freq))
            responses[fi, :] = power[bin_idx, :]

    pref_idx  = np.argmax(responses, axis=0)
    pref_freq = np.array(freqs)[pref_idx]
    return pref_freq, responses


def compute_q_factor(freqs, responses):
    """
    Q-factor per neuron: f_pref / bandwidth_-3dB.
    A narrow-band resonator has high Q; a low-pass integrator has low Q.

    For each neuron:
      1. Find peak response power P_peak at f_pref.
      2. Find the -3dB points (where power = P_peak / 2) on either side.
      3. Bandwidth = f_high - f_low (interpolated on log-frequency axis).
      4. Q = f_pref / bandwidth.  If no lower -3dB point exists (low-pass
         behavior), Q is set to 0 (undefined / pure integration).

    freqs:     (n_freqs,)      log-spaced probe frequencies
    responses: (n_freqs, n_hid) response power per (freq, neuron)

    Returns: Q (n_hid,)  where Q=0 indicates low-pass / no resonance.
    """
    freqs = np.asarray(freqs)
    log_f = np.log10(freqs)
    n_freqs, n_hid = responses.shape
    Q = np.zeros(n_hid)

    for j in range(n_hid):
        r = responses[:, j]
        if r.max() <= 0:
            continue
        pk = np.argmax(r)
        P_peak = r[pk]
        half = P_peak / 2.0

        # Find lower -3dB crossing
        lower = None
        for k in range(pk - 1, -1, -1):
            if r[k] <= half:
                # Linear interp in log-frequency
                if r[k+1] - r[k] > 1e-12:
                    frac = (half - r[k]) / (r[k+1] - r[k])
                    lower = log_f[k] + frac * (log_f[k+1] - log_f[k])
                else:
                    lower = log_f[k]
                break
        # Find upper -3dB crossing
        upper = None
        for k in range(pk + 1, n_freqs):
            if r[k] <= half:
                if r[k-1] - r[k] > 1e-12:
                    frac = (r[k-1] - half) / (r[k-1] - r[k])
                    upper = log_f[k-1] + frac * (log_f[k] - log_f[k-1])
                else:
                    upper = log_f[k]
                break

        # Need both crossings for a well-defined Q (true band-pass)
        if lower is None or upper is None:
            Q[j] = 0.0  # low-pass or monotonic: no resonance
            continue

        f_low  = 10.0 ** lower
        f_high = 10.0 ** upper
        bw = f_high - f_low
        if bw <= 0:
            Q[j] = 0.0
        else:
            Q[j] = freqs[pk] / bw

    return Q


def run_freq_selectivity(hrf_model, lif_model, dataset, dt_hrf, dt_lif,
                         n_inp, n_hid_hrf, n_hid_lif, device, out_dir,
                         f_min_override=None, f_max_override=None,
                         T_sweep_override=None):
    print("\n--- Analysis 3: Frequency Selectivity + Q-factor ---")

    f_min_def, f_max_def, T_def = FREQ_SWEEP_DEFAULTS[dataset]
    f_min   = f_min_override   if f_min_override   is not None else f_min_def
    f_max   = f_max_override   if f_max_override   is not None else f_max_def
    T_sweep = T_sweep_override if T_sweep_override is not None else T_def

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), 30).tolist()
    print(f"  Range [{f_min:.4f}, {f_max:.2f}] Hz, T={T_sweep}")

    print("  Running HRF frequency sweep...")
    pref_hrf, resp_hrf = frequency_sweep(
        hrf_model, 'hrf', dt_hrf, n_inp, n_hid_hrf, device, freqs, T=T_sweep)

    print("  Running LIF frequency sweep...")
    pref_lif, resp_lif = frequency_sweep(
        lif_model, 'lif', dt_lif, n_inp, n_hid_lif, device, freqs, T=T_sweep)

    # Q-factors
    Q_hrf = compute_q_factor(freqs, resp_hrf)
    Q_lif = compute_q_factor(freqs, resp_lif)

    # Fraction of neurons with well-defined resonance (Q > 0)
    frac_res_hrf = float((Q_hrf > 0).mean())
    frac_res_lif = float((Q_lif > 0).mean())

    # Report stats only on resonant neurons (Q > 0)
    q_hrf_res = Q_hrf[Q_hrf > 0] if frac_res_hrf > 0 else np.array([0.0])
    q_lif_res = Q_lif[Q_lif > 0] if frac_res_lif > 0 else np.array([0.0])

    print(f"  HRF: pref_f mean={pref_hrf.mean():.4f} std={pref_hrf.std():.4f} | "
          f"Q (resonant only) mean={q_hrf_res.mean():.2f} | "
          f"frac_resonant={frac_res_hrf*100:.1f}%")
    print(f"  LIF: pref_f mean={pref_lif.mean():.4f} std={pref_lif.std():.4f} | "
          f"Q (resonant only) mean={q_lif_res.mean():.2f} | "
          f"frac_resonant={frac_res_lif*100:.1f}%")

    # --- Plot 1: preferred frequency histograms (same as before) ---
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 25)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pref, color, label in zip(
            axes, [pref_hrf, pref_lif],
            ['#2166AC', '#D6604D'],
            ['s-RON (HRF)', 'LIF-RC']):
        w = np.ones_like(pref) / len(pref)
        ax.hist(pref, bins=bins, weights=w, color=color, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.axvline(pref.mean(), color=color, linestyle='--', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_xlim(f_min * 0.9, f_max * 1.1)
        ax.set_xlabel('Preferred frequency (Hz)')
        ax.set_ylabel('Fraction of neurons')
        ax.set_title(f'{label} — {dataset}\n'
                     f'mean={pref.mean():.4f}, std={pref.std():.4f} Hz')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'freq_selectivity_{dataset}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Q-factor distributions (NEW) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # x-axis: log Q up to some sensible max
    q_all = np.concatenate([q_hrf_res, q_lif_res])
    q_max = np.percentile(q_all, 99) if q_all.size > 1 else 10.0
    q_max = max(q_max, 1.0)
    q_bins = np.linspace(0, q_max, 25)

    for ax, Q, frac, color, label in zip(
            axes, [Q_hrf, Q_lif], [frac_res_hrf, frac_res_lif],
            ['#2166AC', '#D6604D'],
            ['s-RON (HRF)', 'LIF-RC']):
        Q_plot = Q[Q > 0]  # only band-pass neurons
        if Q_plot.size == 0:
            Q_plot = np.array([0.0])
        w = np.ones_like(Q_plot) / len(Q)  # normalize by total N, not resonant
        ax.hist(Q_plot, bins=q_bins, weights=w, color=color, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.axvline(Q_plot.mean(), color=color, linestyle='--', linewidth=1.2)
        ax.set_xlabel('Q-factor (band-pass sharpness)')
        ax.set_ylabel('Fraction of total neurons')
        ax.set_title(f'{label} — {dataset}\n'
                     f'band-pass neurons: {frac*100:.1f}% | '
                     f'mean Q: {Q_plot.mean():.2f}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'q_factor_{dataset}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'pref_freq_hrf_mean': float(pref_hrf.mean()),
        'pref_freq_hrf_std':  float(pref_hrf.std()),
        'pref_freq_lif_mean': float(pref_lif.mean()),
        'pref_freq_lif_std':  float(pref_lif.std()),
        'q_hrf_mean_resonant':  float(q_hrf_res.mean()),
        'q_lif_mean_resonant':  float(q_lif_res.mean()),
        'q_hrf_median_resonant': float(np.median(q_hrf_res)),
        'q_lif_median_resonant': float(np.median(q_lif_res)),
        'frac_resonant_hrf': frac_res_hrf,
        'frac_resonant_lif': frac_res_lif,
        'f_min': f_min, 'f_max': f_max, 'T_sweep': T_sweep,
        # Per-neuron arrays for downstream scatter plotting
        'pref_freq_hrf_array': pref_hrf.tolist(),
        'pref_freq_lif_array': pref_lif.tolist(),
        'q_hrf_array': Q_hrf.tolist(),
        'q_lif_array': Q_lif.tolist(),
    }


# =============================================================================
# Analysis 4: Memory Capacity (NEW)
# =============================================================================

def run_reservoir_on_noise(model, model_type, u_series, n_inp, device):
    """
    Drive reservoir with a 1D noise signal u_series broadcast to n_inp channels
    and collect per-timestep reservoir states.

    Returns states: (T, n_hid) numpy array of reservoir continuous-state values.
    """
    model.eval()
    T = len(u_series)
    n_hid = model.n_hid
    # Broadcast noise to all input channels with 1/sqrt(n_inp) normalization
    # so the total input drive is comparable across datasets.
    scale = 1.0 / np.sqrt(n_inp) if n_inp > 1 else 1.0
    u = torch.as_tensor(u_series, dtype=torch.float32, device=device) * scale
    x = u.view(1, T, 1).expand(1, T, n_inp)  # (1, T, n_inp)

    states = torch.zeros(1, T, n_hid, device=device)
    with torch.no_grad():
        if model_type == 'hrf':
            hy = torch.zeros(1, n_hid, device=device)
            hz = torch.zeros(1, n_hid, device=device)
            ref_period = torch.zeros(1, n_hid, device=device)
            s  = torch.zeros(1, n_hid, device=device)
            lif_v = torch.zeros(1, n_hid, device=device)
            for t in range(T):
                hy, hz, s, ref_period, lif_v, _ = model.bio_cell(
                    x[:, t], hy, hz, lif_v, s, ref_period=ref_period)
                states[:, t, :] = hy
        elif model_type == 'lif':
            res_v = torch.zeros(1, n_hid, device=device)
            res_s = torch.zeros(1, n_hid, device=device)
            lif_v = torch.zeros(1, n_hid, device=device)
            for t in range(T):
                res_v, res_s, lif_v, _ = model.bio_cell(
                    x[:, t], res_v, res_s, lif_v)
                states[:, t, :] = res_v
    return states[0].cpu().numpy()  # (T, n_hid)


def memory_capacity_curves(states, u, washout, k_max):
    """
    Classical Jaeger linear MC: train linear readout to reconstruct u(n-k),
    compute squared correlation r^2(u_target, y_k) for each delay k.

    states: (T, n_hid)
    u:      (T,)
    Returns:
      mc_per_k : (k_max,)  array of r^2 per delay
    """
    T, n_hid = states.shape
    mc = np.zeros(k_max)
    # Training window goes from max(k_max, washout) to T
    start = max(k_max, washout)
    # Feature matrix (with bias col)
    X = np.hstack([states[start:T], np.ones((T - start, 1))])
    # Precompute solver (ridge for numerical stability)
    alpha = 1e-6
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A_inv = np.linalg.pinv(A)

    for k in range(1, k_max + 1):
        target = u[start - k:T - k]
        if np.var(target) < 1e-12:
            mc[k-1] = 0.0
            continue
        w = A_inv @ X.T @ target
        y_pred = X @ w
        # r^2 = (cov(y,t) / (std(y)*std(t)))^2
        cov = np.cov(y_pred, target, ddof=0)
        denom = cov[0, 0] * cov[1, 1]
        if denom < 1e-12:
            mc[k-1] = 0.0
        else:
            mc[k-1] = float(max(0.0, cov[0, 1] ** 2 / denom))
    return mc


def run_memory_capacity(hrf_model, lif_model, hrf_cfg, lif_cfg,
                         n_inp, device, dataset, out_dir,
                         T_mc=None, washout=None, k_max=None, n_noise_seeds=None):
    """
    Classical Jaeger-style MC, independent of the dataset. Reports:
      - linear MC: reconstruction of u(n-k)
      - nonlinear MC: reconstruction of u(n-k)^2
      - MC curves (r^2 per delay) for plotting
    """
    print("\n--- Analysis 4: Memory Capacity (linear + nonlinear) ---")

    T_mc     = T_mc     if T_mc     is not None else MC_DEFAULTS['T_mc']
    washout  = washout  if washout  is not None else MC_DEFAULTS['washout']
    k_max    = k_max    if k_max    is not None else MC_DEFAULTS['k_max']
    n_noise_seeds = n_noise_seeds if n_noise_seeds is not None else MC_DEFAULTS['n_noise_seeds']

    print(f"  T={T_mc}, washout={washout}, k_max={k_max}, seeds={n_noise_seeds}")

    mc_lin_hrf_all, mc_nlin_hrf_all = [], []
    mc_lin_lif_all, mc_nlin_lif_all = [], []

    for sd in range(n_noise_seeds):
        rng = np.random.RandomState(1000 + sd)
        u = rng.uniform(-0.8, 0.8, size=T_mc).astype(np.float32)

        states_hrf = run_reservoir_on_noise(
            hrf_model, 'hrf', u, n_inp, device)
        states_lif = run_reservoir_on_noise(
            lif_model, 'lif', u, n_inp, device)

        mc_lin_hrf  = memory_capacity_curves(states_hrf, u,      washout, k_max)
        mc_nlin_hrf = memory_capacity_curves(states_hrf, u ** 2, washout, k_max)
        mc_lin_lif  = memory_capacity_curves(states_lif, u,      washout, k_max)
        mc_nlin_lif = memory_capacity_curves(states_lif, u ** 2, washout, k_max)

        mc_lin_hrf_all.append(mc_lin_hrf)
        mc_nlin_hrf_all.append(mc_nlin_hrf)
        mc_lin_lif_all.append(mc_lin_lif)
        mc_nlin_lif_all.append(mc_nlin_lif)

    mc_lin_hrf  = np.mean(mc_lin_hrf_all,  axis=0)
    mc_nlin_hrf = np.mean(mc_nlin_hrf_all, axis=0)
    mc_lin_lif  = np.mean(mc_lin_lif_all,  axis=0)
    mc_nlin_lif = np.mean(mc_nlin_lif_all, axis=0)

    MC_lin_hrf  = float(mc_lin_hrf.sum())
    MC_nlin_hrf = float(mc_nlin_hrf.sum())
    MC_lin_lif  = float(mc_lin_lif.sum())
    MC_nlin_lif = float(mc_nlin_lif.sum())

    print(f"  Linear MC    — HRF: {MC_lin_hrf:7.2f}  |  LIF: {MC_lin_lif:7.2f}")
    print(f"  Nonlinear MC — HRF: {MC_nlin_hrf:7.2f}  |  LIF: {MC_nlin_lif:7.2f}")

    # Plot MC curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    delays = np.arange(1, k_max + 1)
    for ax, (mc_h, mc_l, title) in zip(
            axes,
            [((mc_lin_hrf,  mc_lin_lif,  'Linear MC:  reconstruct u(n−k)')),
             ((mc_nlin_hrf, mc_nlin_lif, 'Nonlinear MC:  reconstruct u(n−k)²'))]):
        mh, ml, title = mc_h, mc_l, title
        ax.plot(delays, mh, color='#2166AC', label='s-RON (HRF)', lw=2)
        ax.plot(delays, ml, color='#D6604D', label='LIF-RC',      lw=2)
        ax.set_xlabel('Delay k')
        ax.set_ylabel('r² (forgetting curve)')
        ax.set_title(f'{title}\n({dataset}, T={T_mc})')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'memory_capacity_{dataset}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'MC_linear_hrf':    MC_lin_hrf,
        'MC_linear_lif':    MC_lin_lif,
        'MC_nonlinear_hrf': MC_nlin_hrf,
        'MC_nonlinear_lif': MC_nlin_lif,
        'T_mc': T_mc, 'washout': washout, 'k_max': k_max,
        'n_noise_seeds': n_noise_seeds,
        'mc_lin_curve_hrf':  mc_lin_hrf.tolist(),
        'mc_lin_curve_lif':  mc_lin_lif.tolist(),
        'mc_nlin_curve_hrf': mc_nlin_hrf.tolist(),
        'mc_nlin_curve_lif': mc_nlin_lif.tolist(),
    }


# =============================================================================
# Analysis 5: Linear probe (logistic regression on reservoir features)
# =============================================================================

def _run_cv_probe(X_hrf, X_lif, y, label_scaled=True, n_splits_max=5,
                  random_state=0, C=1.0):
    """
    Shared helper: stratified k-fold CV on logistic regression for two
    feature sets, returning (acc_hrf, std_hrf, acc_lif, std_lif, n_splits).
    """
    if label_scaled:
        sc_h = StandardScaler().fit(X_hrf)
        sc_l = StandardScaler().fit(X_lif)
        X_hrf, X_lif = sc_h.transform(X_hrf), sc_l.transform(X_lif)

    y = y.astype(int)
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    n_splits = min(n_splits_max, n_classes, int(class_counts.min()))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_h = cross_val_score(
            LogisticRegression(max_iter=1000, C=C, n_jobs=-1),
            X_hrf, y, cv=skf, n_jobs=-1)
        s_l = cross_val_score(
            LogisticRegression(max_iter=1000, C=C, n_jobs=-1),
            X_lif, y, cv=skf, n_jobs=-1)
    return (float(s_h.mean()), float(s_h.std()),
            float(s_l.mean()), float(s_l.std()), int(n_splits))


def run_linear_probe(hrf_model, lif_model, test_loader, train_loader,
                      device, needs_reshape, dataset, out_dir,
                      feature='mean', max_train=2000, max_test=1000):
    """
    Train logistic regression on reservoir features (NOT the full readout that
    the paper uses — this is intentionally a simpler, more standard probe).

    If train_loader is available: main result = train/test accuracy on
    disjoint splits. We ALSO run a 5-fold CV on the test set as a sanity
    check, in case the primary train/test gap is surprising.

    If train_loader is None: fall back to 5-fold CV on the test set only.
    """
    print(f"\n--- Analysis 5: Linear probe (feature='{feature}') ---")

    def _extract(loader, model, mx):
        return extract_states(loader, model, device, needs_reshape,
                              max_samples=mx, feature=feature)

    cv_acc_h = cv_std_h = cv_acc_l = cv_std_l = None
    cv_n_splits = None

    if train_loader is not None:
        # --- Primary: train/test with disjoint splits ---
        X_train_hrf, y_train = _extract(train_loader, hrf_model, max_train)
        X_train_lif, _       = _extract(train_loader, lif_model, max_train)
        X_test_hrf,  y_test  = _extract(test_loader,  hrf_model, max_test)
        X_test_lif,  _       = _extract(test_loader,  lif_model, max_test)

        sc_h = StandardScaler().fit(X_train_hrf)
        sc_l = StandardScaler().fit(X_train_lif)
        X_train_hrf_s = sc_h.transform(X_train_hrf)
        X_test_hrf_s  = sc_h.transform(X_test_hrf)
        X_train_lif_s = sc_l.transform(X_train_lif)
        X_test_lif_s  = sc_l.transform(X_test_lif)

        clf_h = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf_h.fit(X_train_hrf_s, y_train)
        acc_h = float(clf_h.score(X_test_hrf_s, y_test))

        clf_l = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf_l.fit(X_train_lif_s, y_train)
        acc_l = float(clf_l.score(X_test_lif_s, y_test))

        std_h = std_l = 0.0  # single-split, no std
        mode = 'train_test'

        # --- Sanity check: 5-fold CV on the TEST set only ---
        # Uses a fresh StandardScaler inside each fold (handled by _run_cv_probe).
        # This confirms the train/test result is not an artifact of the
        # particular train/test split or distribution shift.
        print("  Running CV sanity check on test set...")
        cv_acc_h, cv_std_h, cv_acc_l, cv_std_l, cv_n_splits = _run_cv_probe(
            X_test_hrf, X_test_lif, y_test)
        print(f"  [sanity] CV on test only — "
              f"HRF: {cv_acc_h:.4f} ± {cv_std_h:.4f}  |  "
              f"LIF: {cv_acc_l:.4f} ± {cv_std_l:.4f}  "
              f"({cv_n_splits}-fold)")

    else:
        print("  No train loader — falling back to 5-fold CV on test set only.")
        X_hrf, y = _extract(test_loader, hrf_model, max_test)
        X_lif, _ = _extract(test_loader, lif_model, max_test)
        acc_h, std_h, acc_l, std_l, n_sp = _run_cv_probe(X_hrf, X_lif, y)
        mode = f'{n_sp}fold_cv_on_test'

    print(f"  Linear probe [{mode}] — HRF: {acc_h:.4f} ± {std_h:.4f}  |  "
          f"LIF: {acc_l:.4f} ± {std_l:.4f}")

    result = {
        'probe_acc_hrf':  acc_h,
        'probe_acc_lif':  acc_l,
        'probe_std_hrf':  std_h,
        'probe_std_lif':  std_l,
        'probe_mode':     mode,
        'probe_feature':  feature,
    }
    # Include CV sanity check values when available
    if cv_acc_h is not None:
        result.update({
            'probe_cv_acc_hrf':  cv_acc_h,
            'probe_cv_acc_lif':  cv_acc_l,
            'probe_cv_std_hrf':  cv_std_h,
            'probe_cv_std_lif':  cv_std_l,
            'probe_cv_n_splits': cv_n_splits,
        })
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Representational analysis: HRF vs LIF reservoirs'
    )
    parser.add_argument('--dataset', required=True,
                        choices=['fordA', 'sMNIST', 'shd', 'dvs_gesture'])
    parser.add_argument('--data_dir', type=str, default='data')

    # Analysis selection: 'all' runs everything; otherwise list individual names
    parser.add_argument('--analysis', nargs='+', default=['all'],
                        choices=['cka', 'eff_dim', 'freq_selectivity',
                                 'mc', 'probe', 'all'],
                        help="Analyses to run. 'all' runs everything.")

    parser.add_argument('--cka_feature', type=str, default='mean',
                        choices=['mean', 'rms', 'final'],
                        help='Feature used for CKA, eff_dim, and probe.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu',  action='store_true')
    parser.add_argument('--max_samples_cka',   type=int, default=500)
    parser.add_argument('--max_samples_probe_train', type=int, default=2000)
    parser.add_argument('--max_samples_probe_test',  type=int, default=1000)

    # Output location — default keeps new results separate from any
    # previous 'analysis_results' runs so nothing is overwritten.
    parser.add_argument('--results_root', type=str,
                        default='analysis_results_extended',
                        help="Folder name (relative to script dir) OR absolute "
                             "path where per-dataset results are written. "
                             "Default: 'analysis_results_extended' — this is "
                             "separate from the legacy 'analysis_results' "
                             "folder so prior runs are preserved.")

    # Freq sweep overrides
    parser.add_argument('--f_min',   type=float, default=None)
    parser.add_argument('--f_max',   type=float, default=None)
    parser.add_argument('--T_sweep', type=int,   default=None)

    # MC overrides
    parser.add_argument('--T_mc',          type=int, default=None)
    parser.add_argument('--mc_washout',    type=int, default=None)
    parser.add_argument('--mc_k_max',      type=int, default=None)
    parser.add_argument('--mc_seeds',      type=int, default=None)

    # Effective-dimensionality overrides
    parser.add_argument('--eff_dim_mode', default='temporal',
                        choices=['temporal', 'pooled'],
                        help="'temporal' (default, recommended): metrics on "
                             "per-timestep reservoir states. 'pooled': on "
                             "per-sample pooled features (less informative).")
    parser.add_argument('--temporal_n_samples', type=int, default=100,
                        help='Number of sequences to collect temporal states from.')
    parser.add_argument('--temporal_subsample', type=int, default=2,
                        help='Keep every k-th timestep (controls memory).')

    args = parser.parse_args()

    # Resolve 'all' -> every analysis. After seeing results on all four
    # datasets, we may decide to drop some (MC, eff_dim) from the main paper.
    # For now, run everything to gather evidence.
    if 'all' in args.analysis:
        args.analysis = ['cka', 'eff_dim', 'freq_selectivity', 'mc', 'probe']

    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Support both absolute and relative paths for results_root
    if os.path.isabs(args.results_root):
        results_root = args.results_root
    else:
        results_root = os.path.join(script_dir, args.results_root)
    out_dir = os.path.join(results_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 72)
    print(f'REPRESENTATIONAL ANALYSIS  |  dataset={args.dataset}')
    print(f'Analyses:      {args.analysis}')
    print(f'CKA/probe feature: {args.cka_feature}')
    print(f'Device:        {device}')
    print(f'Output dir:    {out_dir}')
    print('=' * 72)

    print('\nBuilding HRF model (s-RON)...')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed);    random.seed(args.seed)
    hrf_model, hrf_cfg = build_hrf_model(args.dataset, device, seed=args.seed)
    apply_sparse_input_projection(
        hrf_model, hrf_cfg['input_density'],
        hrf_cfg['n_inp'], hrf_cfg['n_hid'], device)

    print('Building LIF model (LIF-RC)...')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed);    random.seed(args.seed)
    lif_model, lif_cfg = build_lif_model(args.dataset, device, seed=args.seed)
    apply_sparse_input_projection(
        lif_model, lif_cfg['input_density'],
        lif_cfg['n_inp'], lif_cfg['n_hid'], device)

    all_results = {'dataset': args.dataset, 'seed': args.seed,
                   'cka_feature': args.cka_feature}

    # ---- Do we need dataset loaders? ----
    needs_loader = any(a in args.analysis
                       for a in ('cka', 'eff_dim', 'probe'))
    needs_train_loader = 'probe' in args.analysis

    if needs_loader:
        print('\nLoading dataset...')
        test_loader, n_inp, needs_reshape, train_loader = load_dataset(
            args.dataset, args, device, want_train=needs_train_loader)
    else:
        test_loader, train_loader = None, None
        n_inp, needs_reshape = hrf_cfg['n_inp'], False

    # --- 1. CKA ---
    if 'cka' in args.analysis:
        all_results['cka'] = run_cka(
            hrf_model, lif_model, test_loader, device, needs_reshape,
            args.dataset, out_dir,
            max_samples=args.max_samples_cka,
            cka_feature=args.cka_feature)

    # --- 2. Effective dim + LUD + ASE ---
    if 'eff_dim' in args.analysis:
        all_results['eff_dim'] = run_eff_dim_richness(
            hrf_model, lif_model, test_loader, device, needs_reshape,
            args.dataset, out_dir,
            feature=args.cka_feature, max_samples=1000,
            mode=args.eff_dim_mode,
            temporal_n_samples=args.temporal_n_samples,
            temporal_subsample=args.temporal_subsample)

    # --- 3. Frequency selectivity + Q-factor ---
    if 'freq_selectivity' in args.analysis:
        all_results['freq_selectivity'] = run_freq_selectivity(
            hrf_model, lif_model,
            dataset=args.dataset,
            dt_hrf=hrf_cfg['dt'], dt_lif=lif_cfg['dt'],
            n_inp=hrf_cfg['n_inp'],
            n_hid_hrf=hrf_cfg['n_hid'], n_hid_lif=lif_cfg['n_hid'],
            device=device, out_dir=out_dir,
            f_min_override=args.f_min,
            f_max_override=args.f_max,
            T_sweep_override=args.T_sweep)

    # --- 4. Memory capacity (dataset-INDEPENDENT; uses i.i.d. noise) ---
    if 'mc' in args.analysis:
        all_results['mc'] = run_memory_capacity(
            hrf_model, lif_model, hrf_cfg, lif_cfg,
            n_inp=hrf_cfg['n_inp'], device=device,
            dataset=args.dataset, out_dir=out_dir,
            T_mc=args.T_mc, washout=args.mc_washout,
            k_max=args.mc_k_max, n_noise_seeds=args.mc_seeds)

    # --- 5. Linear probe ---
    if 'probe' in args.analysis:
        all_results['probe'] = run_linear_probe(
            hrf_model, lif_model, test_loader, train_loader,
            device, needs_reshape, args.dataset, out_dir,
            feature=args.cka_feature,
            max_train=args.max_samples_probe_train,
            max_test=args.max_samples_probe_test)

    # --- Save summary ---
    summary_path = os.path.join(
        out_dir, f'analysis_summary_{args.dataset}_{args.cka_feature}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # --- Print compact summary table ---
    print(f'\n{"="*72}')
    print(f'SUMMARY  |  dataset={args.dataset}')
    print(f'{"="*72}')
    if 'cka' in all_results:
        r = all_results['cka']
        print(f"  CKA         HRF: {r['cka_hrf']:8.4f}  LIF: {r['cka_lif']:8.4f}")
    if 'eff_dim' in all_results:
        r = all_results['eff_dim']
        print(f"  PR          HRF: {r['pr_hrf']:8.1f}  LIF: {r['pr_lif']:8.1f}")
        print(f"  LUD         HRF: {r['lud_hrf']:8d}  LIF: {r['lud_lif']:8d}")
        print(f"  ASE         HRF: {r['ase_hrf']:8.3f}  LIF: {r['ase_lif']:8.3f}")
    if 'mc' in all_results:
        r = all_results['mc']
        print(f"  MC linear   HRF: {r['MC_linear_hrf']:8.2f}  "
              f"LIF: {r['MC_linear_lif']:8.2f}")
        print(f"  MC nonlin.  HRF: {r['MC_nonlinear_hrf']:8.2f}  "
              f"LIF: {r['MC_nonlinear_lif']:8.2f}")
    if 'freq_selectivity' in all_results:
        r = all_results['freq_selectivity']
        print(f"  Pref-f std  HRF: {r['pref_freq_hrf_std']:8.4f}  "
              f"LIF: {r['pref_freq_lif_std']:8.4f}")
        print(f"  Q (mean)    HRF: {r['q_hrf_mean_resonant']:8.2f}  "
              f"LIF: {r['q_lif_mean_resonant']:8.2f}")
        print(f"  % band-pass HRF: {r['frac_resonant_hrf']*100:7.1f}%  "
              f"LIF: {r['frac_resonant_lif']*100:7.1f}%")
    if 'probe' in all_results:
        r = all_results['probe']
        print(f"  Probe acc   HRF: {r['probe_acc_hrf']:8.4f}  "
              f"LIF: {r['probe_acc_lif']:8.4f}  [{r['probe_mode']}]")
        if 'probe_cv_acc_hrf' in r:
            print(f"  Probe CV    HRF: {r['probe_cv_acc_hrf']:8.4f}  "
                  f"LIF: {r['probe_cv_acc_lif']:8.4f}  "
                  f"[{r['probe_cv_n_splits']}-fold on test, sanity check]")
    print(f'\n  Results saved to: {out_dir}')


if __name__ == '__main__':
    main()
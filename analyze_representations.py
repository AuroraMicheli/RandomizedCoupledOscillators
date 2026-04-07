"""
Representational analysis: HRF reservoir (s-RON) vs LIF reservoir (LIF-RC)

Three analyses:
  1. Centered Kernel Alignment (CKA)
  2. Effective Dimensionality (Participation Ratio)
  3. Frequency Selectivity

Usage:
    python analyze_representations.py --dataset shd --data_dir data/SHD
    python analyze_representations.py --dataset shd --analysis freq_selectivity \
        --f_min 0.01 --T_sweep 8000




python analyze_representations.py --dataset fordA --analysis freq_selectivity
python analyze_representations.py --dataset sMNIST --analysis freq_selectivity
python analyze_representations.py --dataset shd --data_dir data/SHD --analysis freq_selectivity --f_min 0.01 --T_sweep 8000
python analyze_representations.py --dataset dvs_gesture --data_dir data/DVSGesture --analysis freq_selectivity --f_min 0.01 --T_sweep 8000
"""



"""
Representational analysis: HRF reservoir (s-RON) vs LIF reservoir (LIF-RC)

Three analyses:
  1. Centered Kernel Alignment (CKA)
  2. Effective Dimensionality (Participation Ratio)
  3. Frequency Selectivity

Usage:
    python analyze_representations.py --dataset fordA --analysis cka --cka_feature mean
    python analyze_representations.py --dataset fordA --analysis cka --cka_feature rms
    python analyze_representations.py --dataset fordA --analysis cka --cka_feature final

    python analyze_representations.py --dataset fordA --analysis freq_selectivity
    python analyze_representations.py --dataset sMNIST --analysis freq_selectivity
    python analyze_representations.py --dataset shd --data_dir data/SHD --analysis freq_selectivity --f_min 0.001 --T_sweep 20000
    python analyze_representations.py --dataset dvs_gesture --data_dir data/DVSGesture --analysis freq_selectivity --f_min 0.001 --T_sweep 20000
"""

import argparse
import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# Best configs
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

# Per-dataset frequency sweep defaults.
# f_min is set low enough to capture LIF distributions which pile up at very
# low frequencies.  f_max is capped at 2.0 Hz for all datasets because the
# preferred frequency distributions are empirically empty above ~1 Hz,
# avoiding wasted space on the right side of the histograms.
# T_sweep must be long enough for reliable DFT resolution at f_min:
#   frequency resolution = 1 / (T_sweep/2 * dt)
#   => T_sweep = 2 / (f_min * dt)
# We round up generously.
FREQ_SWEEP_DEFAULTS = {
    #           f_min    f_max   T_sweep
    'fordA':    (0.01,  2.0,    6000),
    'sMNIST':   (0.01,  2.0,    6000),
    'shd':      (0.001,  2.0,    50000),
    'dvs_gesture': (0.001, 2.0,  50000),
}


# =============================================================================
# Model builders
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

def load_dataset(dataset, args, device):
    if dataset == 'fordA':
        train_loader, _, test_loader = get_FordA_data(64, 120, whole_train=True)
        return test_loader, 1, False

    elif dataset == 'sMNIST':
        _, _, test_loader = get_mnist_data(256, 100)
        return test_loader, 1, True

    elif dataset == 'shd':
        cfg = HRF_CONFIGS['shd']
        _, _, test_loader = get_SHD_data(
            batch_train=128, batch_test=256,
            data_dir=args.data_dir,
            num_steps=cfg['num_steps'], max_time=1.4
        )
        return test_loader, 700, False

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
        return test_loader, n_inp, False


# =============================================================================
# Analysis 1: CKA
# =============================================================================

def centered_kernel_alignment(X, Y_kernel):
    n = X.shape[0]

    def center(K):
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X  = X @ X.T
    K_Xc = center(K_X)
    K_Yc = center(Y_kernel)

    hsic_xy = np.sum(K_Xc * K_Yc)
    hsic_xx = np.sum(K_Xc * K_Xc)
    hsic_yy = np.sum(K_Yc * K_Yc)

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def extract_states(loader, model, device, needs_reshape, max_samples=500,
                   cka_feature='mean'):
    """
    Extract per-sample reservoir features for CKA.

    cka_feature options:
      'mean'  : temporal mean of reservoir state.
      'rms'   : sqrt(mean(h^2)) per neuron — nonzero for oscillatory neurons.
      'final' : reservoir state at the last timestep.
    """
    assert cka_feature in ('mean', 'rms', 'final'), \
        f"cka_feature must be 'mean', 'rms', or 'final', got '{cka_feature}'"

    model.eval()
    all_feats, all_labels = [], []
    n_collected = 0

    original_mode = model.readout_mode
    if cka_feature in ('mean', 'final'):
        model.readout_mode = cka_feature
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
            if cka_feature == 'rms':
                feats = feats[:, :model.n_hid]
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy())
            n_collected += x.shape[0]

    model.readout_mode = original_mode

    feats  = np.concatenate(all_feats,  axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    labels = labels.ravel()
    return feats, labels


def run_cka(hrf_model, lif_model, loader, device, needs_reshape,
            dataset, out_dir, max_samples=500, cka_feature='mean'):
    print(f"\n--- Analysis 1: CKA (feature='{cka_feature}') ---")

    hrf_feats, labels = extract_states(loader, hrf_model, device,
                                       needs_reshape, max_samples, cka_feature)
    lif_feats, _      = extract_states(loader, lif_model, device,
                                       needs_reshape, max_samples, cka_feature)

    n = min(len(labels), len(hrf_feats), len(lif_feats))
    hrf_feats, lif_feats, labels = hrf_feats[:n], lif_feats[:n], labels[:n]
    labels = labels.ravel()
    Y = (labels[:, None] == labels[None, :]).astype(float)

    hrf_norm = hrf_feats / (np.linalg.norm(hrf_feats, axis=1, keepdims=True) + 1e-8)
    lif_norm = lif_feats / (np.linalg.norm(lif_feats, axis=1, keepdims=True) + 1e-8)

    print(f"  Using {n} samples, {hrf_norm.shape[1]} HRF features, "
          f"{lif_norm.shape[1]} LIF features")

    cka_hrf = centered_kernel_alignment(hrf_norm, Y)
    cka_lif = centered_kernel_alignment(lif_norm, Y)

    print(f"  CKA (HRF):   {cka_hrf:.4f}")
    print(f"  CKA (LIF):   {cka_lif:.4f}")
    print(f"  Improvement: {(cka_hrf - cka_lif) / (cka_lif + 1e-8) * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(['s-RON\n(HRF)', 'LIF-RC'], [cka_hrf, cka_lif],
                  color=['#2166AC', '#D6604D'], width=0.5, edgecolor='black')
    ax.set_ylabel('CKA with class labels', fontsize=12)
    ax.set_title(f'Linear CKA — {dataset}\n(feature: {cka_feature})', fontsize=12)
    ax.set_ylim(0, min(1.0, max(cka_hrf, cka_lif) * 1.3))
    for bar, val in zip(bars, [cka_hrf, cka_lif]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f'cka_{dataset}_{cka_feature}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'cka_hrf': cka_hrf, 'cka_lif': cka_lif, 'cka_feature': cka_feature}


# =============================================================================
# Analysis 2: Effective Dimensionality
# =============================================================================

def participation_ratio(X):
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    sum1 = eigvals.sum()
    sum2 = (eigvals ** 2).sum()
    if sum2 < 1e-12:
        return 0.0
    return float(sum1 ** 2 / sum2)


def run_eff_dim(hrf_model, lif_model, loader, device, needs_reshape,
                dataset, out_dir, cka_feature='mean'):
    print(f"\n--- Analysis 2: Effective Dimensionality (feature='{cka_feature}') ---")

    hrf_feats, _ = extract_states(loader, hrf_model, device, needs_reshape,
                                  max_samples=1000, cka_feature=cka_feature)
    lif_feats, _ = extract_states(loader, lif_model, device, needs_reshape,
                                  max_samples=1000, cka_feature=cka_feature)

    pr_hrf = participation_ratio(hrf_feats)
    pr_lif = participation_ratio(lif_feats)
    n_hid_hrf = hrf_feats.shape[1]
    n_hid_lif = lif_feats.shape[1]

    print(f"  PR (HRF):  {pr_hrf:.1f}  /  {n_hid_hrf} neurons  "
          f"({pr_hrf/n_hid_hrf*100:.1f}% of capacity)")
    print(f"  PR (LIF):  {pr_lif:.1f}  /  {n_hid_lif} neurons  "
          f"({pr_lif/n_hid_lif*100:.1f}% of capacity)")

    return {
        'pr_hrf': pr_hrf, 'pr_lif': pr_lif,
        'pr_frac_hrf': pr_hrf / n_hid_hrf,
        'pr_frac_lif': pr_lif / n_hid_lif,
        'n_hid_hrf': n_hid_hrf, 'n_hid_lif': n_hid_lif,
    }


# =============================================================================
# Analysis 3: Frequency Selectivity
# =============================================================================

def get_preferred_frequency(model, model_type, dt, n_inp, n_hid, device,
                             freqs, T=5000, batch_size=1):
    model.eval()
    t_vec = torch.arange(T, dtype=torch.float32, device=device)
    all_responses = np.zeros((len(freqs), n_hid))

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
                    hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
                        x[:, t], hy, hz, lif_v, s, ref_period=ref_period)
                    states[:, t, :] = hy

            elif model_type == 'lif':
                res_v = torch.zeros(B, n_hid, device=device)
                res_s = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    res_v, res_s, lif_v, lif_s = model.bio_cell(
                        x[:, t], res_v, res_s, lif_v)
                    states[:, t, :] = res_v

            states_np = states.mean(0).cpu().numpy()
            steady    = states_np[T//2:, :]
            fft_vals  = np.fft.rfft(steady, axis=0)
            power     = np.abs(fft_vals) ** 2
            fft_freqs = np.fft.rfftfreq(steady.shape[0], d=dt)
            bin_idx   = np.argmin(np.abs(fft_freqs - freq))
            all_responses[fi, :] = power[bin_idx, :]

    preferred_idx  = np.argmax(all_responses, axis=0)
    preferred_freq = np.array(freqs)[preferred_idx]
    return preferred_freq, all_responses


def run_freq_selectivity(hrf_model, lif_model, dataset, dt_hrf, dt_lif,
                         n_inp, n_hid_hrf, n_hid_lif, device, out_dir,
                         f_min_override=None, f_max_override=None,
                         T_sweep_override=None):
    print("\n--- Analysis 3: Frequency Selectivity ---")

    f_min_def, f_max_def, T_def = FREQ_SWEEP_DEFAULTS[dataset]
    f_min   = f_min_override   if f_min_override   is not None else f_min_def
    f_max   = f_max_override   if f_max_override   is not None else f_max_def
    T_sweep = T_sweep_override if T_sweep_override is not None else T_def

    # Both models use the same f_min and f_max so histograms are directly
    # comparable. Each model uses its own dt for the sweep signal construction.
    freqs_hrf = np.logspace(np.log10(f_min), np.log10(f_max), 30).tolist()
    freqs_lif = np.logspace(np.log10(f_min), np.log10(f_max), 30).tolist()

    print(f"  Sweep range: [{f_min:.4f}, {f_max:.2f}] Hz  "
          f"(30 log-spaced freqs), T={T_sweep}")
    print(f"  HRF dt={dt_hrf}  |  LIF dt={dt_lif}")
    print(f"  DFT resolution at f_min: "
          f"HRF ~{1.0/(T_sweep/2*dt_hrf):.5f} Hz  "
          f"LIF ~{1.0/(T_sweep/2*dt_lif):.5f} Hz")

    print("  Running HRF frequency sweep...")
    pref_hrf, responses_hrf = get_preferred_frequency(
        hrf_model, 'hrf', dt_hrf, n_inp, n_hid_hrf, device, freqs_hrf, T=T_sweep)

    print("  Running LIF frequency sweep...")
    pref_lif, responses_lif = get_preferred_frequency(
        lif_model, 'lif', dt_lif, n_inp, n_hid_lif, device, freqs_lif, T=T_sweep)

    print(f"  HRF preferred freq: mean={pref_hrf.mean():.4f}  "
          f"std={pref_hrf.std():.4f}  "
          f"range=[{pref_hrf.min():.4f}, {pref_hrf.max():.4f}]")
    print(f"  LIF preferred freq: mean={pref_lif.mean():.4f}  "
          f"std={pref_lif.std():.4f}  "
          f"range=[{pref_lif.min():.4f}, {pref_lif.max():.4f}]")

    bins = np.logspace(np.log10(f_min), np.log10(f_max), 25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, pref, color, label in zip(
            axes,
            [pref_hrf, pref_lif],
            ['#2166AC', '#D6604D'],
            ['s-RON (HRF)', 'LIF-RC']):
        w = np.ones_like(pref) / len(pref)
        ax.hist(pref, bins=bins, weights=w, color=color, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.axvline(pref.mean(), color=color, linestyle='--', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_xlim(f_min * 0.9, f_max * 1.1)
        ax.set_xlabel('Preferred frequency (Hz)', fontsize=12)
        ax.set_ylabel('Fraction of neurons', fontsize=12)
        ax.set_title(f'{label} — {dataset}\n'
                     f'mean={pref.mean():.4f}, std={pref.std():.4f} Hz',
                     fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'freq_selectivity_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {
        'pref_freq_hrf_mean': float(pref_hrf.mean()),
        'pref_freq_hrf_std':  float(pref_hrf.std()),
        'pref_freq_lif_mean': float(pref_lif.mean()),
        'pref_freq_lif_std':  float(pref_lif.std()),
        'f_min': f_min,
        'f_max': f_max,
        'T_sweep': T_sweep,
        'freqs_hrf': freqs_hrf,
        'freqs_lif': freqs_lif,
        'pref_freq_hrf_array': pref_hrf.tolist(),
        'pref_freq_lif_array': pref_lif.tolist(),
    }


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
    parser.add_argument('--analysis', nargs='+',
                        default=['cka', 'eff_dim', 'freq_selectivity'],
                        choices=['cka', 'eff_dim', 'freq_selectivity'])
    parser.add_argument('--cka_feature', type=str, default='mean',
                        choices=['mean', 'rms', 'final'],
                        help='Feature used for CKA and eff_dim.')
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--cpu',             action='store_true')
    parser.add_argument('--max_samples_cka', type=int,   default=500)
    # Frequency sweep overrides (defaults are in FREQ_SWEEP_DEFAULTS)
    parser.add_argument('--f_min',   type=float, default=None,
                        help='Override lower bound of frequency sweep (Hz).')
    parser.add_argument('--f_max',   type=float, default=None,
                        help='Override upper bound of frequency sweep (Hz).')
    parser.add_argument('--T_sweep', type=int,   default=None,
                        help='Override number of timesteps per frequency.')
    args = parser.parse_args()

    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'analysis_results', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(f'REPRESENTATIONAL ANALYSIS  |  dataset={args.dataset}')
    print(f'Analyses:    {args.analysis}')
    print(f'CKA feature: {args.cka_feature}')
    print(f'Device:      {device}')
    print('=' * 70)

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

    if 'cka' in args.analysis or 'eff_dim' in args.analysis:
        print('\nLoading dataset...')
        loader, n_inp, needs_reshape = load_dataset(args.dataset, args, device)

        if 'cka' in args.analysis:
            res = run_cka(hrf_model, lif_model, loader, device,
                          needs_reshape, args.dataset, out_dir,
                          max_samples=args.max_samples_cka,
                          cka_feature=args.cka_feature)
            all_results['cka'] = res

        if 'eff_dim' in args.analysis:
            res = run_eff_dim(hrf_model, lif_model, loader, device,
                              needs_reshape, args.dataset, out_dir,
                              cka_feature=args.cka_feature)
            all_results['eff_dim'] = res
       
    if 'freq_selectivity' in args.analysis:
        res = run_freq_selectivity(
            hrf_model, lif_model,
            dataset=args.dataset,
            dt_hrf=hrf_cfg['dt'], dt_lif=lif_cfg['dt'],
            n_inp=hrf_cfg['n_inp'],
            n_hid_hrf=hrf_cfg['n_hid'], n_hid_lif=lif_cfg['n_hid'],
            device=device, out_dir=out_dir,
            f_min_override=args.f_min,
            f_max_override=args.f_max,
            T_sweep_override=args.T_sweep,
        )
        all_results['freq_selectivity'] = res

    summary_path = os.path.join(
        out_dir, f'analysis_summary_{args.dataset}_{args.cka_feature}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*70}\nSUMMARY\n{"="*70}')
    if 'cka' in all_results:
        r = all_results['cka']
        print(f"CKA [{args.cka_feature}]  — "
              f"HRF: {r['cka_hrf']:.4f}  |  LIF: {r['cka_lif']:.4f}")
    if 'eff_dim' in all_results:
        r = all_results['eff_dim']
        print(f"Eff. Dim [{args.cka_feature}]  — "
              f"HRF: {r['pr_hrf']:.1f} ({r['pr_frac_hrf']*100:.2f}%)  "
              f"|  LIF: {r['pr_lif']:.1f} ({r['pr_frac_lif']*100:.2f}%)")
    if 'freq_selectivity' in all_results:
        r = all_results['freq_selectivity']
        print(f"Freq. std — HRF: {r['pref_freq_hrf_std']:.4f} Hz  "
              f"|  LIF: {r['pref_freq_lif_std']:.4f} Hz")
    print(f'\nAll results saved to: {out_dir}')


if __name__ == '__main__':
    main()






'''


import argparse
import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# Best configs
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
        connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
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
        connectivity_lif2res=0.2, connectivity_res2enc=1.0,
        input_density=1.0, num_steps=500, n_inp=1, readout_mode='final',
    ),
    'sMNIST': dict(
        n_hid=800, dt=0.034, rho=0.976, inp_scaling=0.8471,
        tau_m=36.19, tau_m_range=3.60,
        theta_res=0.00254, theta_res_range=0.01180,
        theta_lif=0.1507, tau_filter=20.0,
        connectivity_lif2res=0.2, connectivity_res2enc=1.0,
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


# =============================================================================
# Model builders
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

def load_dataset(dataset, args, device):
    if dataset == 'fordA':
        train_loader, _, test_loader = get_FordA_data(64, 120, whole_train=True)
        return test_loader, 1, False

    elif dataset == 'sMNIST':
        _, _, test_loader = get_mnist_data(256, 100)
        return test_loader, 1, True

    elif dataset == 'shd':
        cfg = HRF_CONFIGS['shd']
        _, _, test_loader = get_SHD_data(
            batch_train=128, batch_test=256,
            data_dir=args.data_dir,
            num_steps=cfg['num_steps'], max_time=1.4
        )
        return test_loader, 700, False

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
        return test_loader, n_inp, False


# =============================================================================
# Analysis 1: CKA
# =============================================================================

def centered_kernel_alignment(X, Y_kernel):
    n = X.shape[0]

    def center(K):
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X  = X @ X.T
    K_Xc = center(K_X)
    K_Yc = center(Y_kernel)

    hsic_xy = np.sum(K_Xc * K_Yc)
    hsic_xx = np.sum(K_Xc * K_Xc)
    hsic_yy = np.sum(K_Yc * K_Yc)

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def extract_states(loader, model, device, needs_reshape, max_samples=500):
    """Use temporal mean state (readout_mode='mean') for richer representation."""
    model.eval()
    all_feats, all_labels = [], []
    n_collected = 0

    original_mode = model.readout_mode
    model.readout_mode = 'mean'

    with torch.no_grad():
        for x, y in loader:
            if n_collected >= max_samples:
                break
            x = x.to(device)
            if needs_reshape:
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
            feats, _ = model(x)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy())
            n_collected += x.shape[0]

    model.readout_mode = original_mode

    feats  = np.concatenate(all_feats,  axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    labels = labels.ravel()
    return feats, labels


def run_cka(hrf_model, lif_model, loader, device, needs_reshape,
            dataset, out_dir, max_samples=500):
    print("\n--- Analysis 1: CKA ---")

    hrf_feats, labels = extract_states(loader, hrf_model, device,
                                       needs_reshape, max_samples)
    lif_feats, _      = extract_states(loader, lif_model, device,
                                       needs_reshape, max_samples)

    n = min(len(labels), len(hrf_feats), len(lif_feats))
    hrf_feats, lif_feats, labels = hrf_feats[:n], lif_feats[:n], labels[:n]
    labels = labels.ravel()
    Y = (labels[:, None] == labels[None, :]).astype(float)
    assert Y.shape == (n, n), f"Y shape error: {Y.shape}"

    hrf_norm = hrf_feats / (np.linalg.norm(hrf_feats, axis=1, keepdims=True) + 1e-8)
    lif_norm = lif_feats / (np.linalg.norm(lif_feats, axis=1, keepdims=True) + 1e-8)

    print(f"  Using {n} samples, {hrf_norm.shape[1]} HRF features, "
          f"{lif_norm.shape[1]} LIF features")

    cka_hrf = centered_kernel_alignment(hrf_norm, Y)
    cka_lif = centered_kernel_alignment(lif_norm, Y)

    print(f"  CKA (HRF):   {cka_hrf:.4f}")
    print(f"  CKA (LIF):   {cka_lif:.4f}")
    print(f"  Improvement: {(cka_hrf - cka_lif) / (cka_lif + 1e-8) * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(['s-RON\n(HRF)', 'LIF-RC'], [cka_hrf, cka_lif],
                  color=['#2196F3', '#FF9800'], width=0.5, edgecolor='black')
    ax.set_ylabel('CKA with class labels', fontsize=12)
    ax.set_title(f'Linear CKA — {dataset}', fontsize=13)
    ax.set_ylim(0, min(1.0, max(cka_hrf, cka_lif) * 1.3))
    for bar, val in zip(bars, [cka_hrf, cka_lif]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f'cka_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'cka_hrf': cka_hrf, 'cka_lif': cka_lif}


# =============================================================================
# Analysis 2: Effective Dimensionality
# =============================================================================

def participation_ratio(X):
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    sum1 = eigvals.sum()
    sum2 = (eigvals ** 2).sum()
    if sum2 < 1e-12:
        return 0.0
    return float(sum1 ** 2 / sum2)


def run_eff_dim(hrf_model, lif_model, loader, device, needs_reshape, dataset, out_dir):
    print("\n--- Analysis 2: Effective Dimensionality ---")

    hrf_feats, _ = extract_states(loader, hrf_model, device, needs_reshape,
                                  max_samples=1000)
    lif_feats, _ = extract_states(loader, lif_model, device, needs_reshape,
                                  max_samples=1000)

    pr_hrf = participation_ratio(hrf_feats)
    pr_lif = participation_ratio(lif_feats)
    n_hid_hrf = hrf_feats.shape[1]
    n_hid_lif = lif_feats.shape[1]

    print(f"  PR (HRF):  {pr_hrf:.1f}  /  {n_hid_hrf} neurons  "
          f"({pr_hrf/n_hid_hrf*100:.1f}% of capacity)")
    print(f"  PR (LIF):  {pr_lif:.1f}  /  {n_hid_lif} neurons  "
          f"({pr_lif/n_hid_lif*100:.1f}% of capacity)")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(['s-RON\n(HRF)', 'LIF-RC'], [pr_hrf, pr_lif],
                color=['#2196F3', '#FF9800'], width=0.5, edgecolor='black')
    axes[0].set_ylabel('Participation Ratio', fontsize=12)
    axes[0].set_title(f'Effective Dimensionality — {dataset}', fontsize=12)
    for i, val in enumerate([pr_hrf, pr_lif]):
        axes[0].text(i, val + 0.5, f'{val:.1f}', ha='center', fontsize=11)

    frac_hrf = pr_hrf / n_hid_hrf
    frac_lif = pr_lif / n_hid_lif
    axes[1].bar(['s-RON\n(HRF)', 'LIF-RC'], [frac_hrf, frac_lif],
                color=['#2196F3', '#FF9800'], width=0.5, edgecolor='black')
    axes[1].set_ylabel('Fraction of capacity used', fontsize=12)
    axes[1].set_title(f'PR / n_hid — {dataset}', fontsize=12)
    axes[1].set_ylim(0, max(frac_hrf, frac_lif) * 1.3 + 0.001)
    for i, val in enumerate([frac_hrf, frac_lif]):
        axes[1].text(i, val + 0.0002, f'{val:.4f}', ha='center', fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, f'eff_dim_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {
        'pr_hrf': pr_hrf, 'pr_lif': pr_lif,
        'pr_frac_hrf': frac_hrf, 'pr_frac_lif': frac_lif,
        'n_hid_hrf': n_hid_hrf, 'n_hid_lif': n_hid_lif,
    }


# =============================================================================
# Analysis 3: Frequency Selectivity
# =============================================================================

def get_preferred_frequency(model, model_type, dt, n_inp, n_hid, device,
                             freqs, T=5000, batch_size=1):
    model.eval()
    t_vec = torch.arange(T, dtype=torch.float32, device=device)
    all_responses = np.zeros((len(freqs), n_hid))

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
                    hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
                        x[:, t], hy, hz, lif_v, s, ref_period=ref_period)
                    states[:, t, :] = hy

            elif model_type == 'lif':
                res_v = torch.zeros(B, n_hid, device=device)
                res_s = torch.zeros(B, n_hid, device=device)
                lif_v = torch.zeros(B, n_hid, device=device)
                for t in range(T):
                    res_v, res_s, lif_v, lif_s = model.bio_cell(
                        x[:, t], res_v, res_s, lif_v)
                    states[:, t, :] = res_v

            states_np = states.mean(0).cpu().numpy()
            steady    = states_np[T//2:, :]
            fft_vals  = np.fft.rfft(steady, axis=0)
            power     = np.abs(fft_vals) ** 2
            fft_freqs = np.fft.rfftfreq(steady.shape[0], d=dt)
            bin_idx   = np.argmin(np.abs(fft_freqs - freq))
            all_responses[fi, :] = power[bin_idx, :]

    preferred_idx  = np.argmax(all_responses, axis=0)
    preferred_freq = np.array(freqs)[preferred_idx]
    return preferred_freq, all_responses


def run_freq_selectivity(hrf_model, lif_model, dataset, dt_hrf, dt_lif,
                         n_inp, n_hid_hrf, n_hid_lif, device, out_dir,
                         f_min_override=None, T_sweep_override=None):
    print("\n--- Analysis 3: Frequency Selectivity ---")

    T_sweep       = T_sweep_override   if T_sweep_override   is not None else 5000
    f_min_default = f_min_override     if f_min_override     is not None else 0.05

    f_min_hrf = f_min_default
    f_max_hrf = 1.0 / (4.0 * dt_hrf)
    freqs_hrf = np.logspace(np.log10(f_min_hrf), np.log10(f_max_hrf), 30).tolist()

    f_min_lif = f_min_default
    f_max_lif = 1.0 / (4.0 * dt_lif)
    freqs_lif = np.logspace(np.log10(f_min_lif), np.log10(f_max_lif), 30).tolist()

    print(f"  HRF sweep: {len(freqs_hrf)} freqs "
          f"{f_min_hrf:.4f}--{f_max_hrf:.2f} Hz, dt={dt_hrf}, T={T_sweep}")
    print(f"  LIF sweep: {len(freqs_lif)} freqs "
          f"{f_min_lif:.4f}--{f_max_lif:.2f} Hz, dt={dt_lif}, T={T_sweep}")

    print("  Running HRF frequency sweep...")
    pref_hrf, responses_hrf = get_preferred_frequency(
        hrf_model, 'hrf', dt_hrf, n_inp, n_hid_hrf, device, freqs_hrf, T=T_sweep)

    print("  Running LIF frequency sweep...")
    pref_lif, responses_lif = get_preferred_frequency(
        lif_model, 'lif', dt_lif, n_inp, n_hid_lif, device, freqs_lif, T=T_sweep)

    print(f"  HRF preferred freq: mean={pref_hrf.mean():.4f}  "
          f"std={pref_hrf.std():.4f}  "
          f"range=[{pref_hrf.min():.4f}, {pref_hrf.max():.4f}]")
    print(f"  LIF preferred freq: mean={pref_lif.mean():.4f}  "
          f"std={pref_lif.std():.4f}  "
          f"range=[{pref_lif.min():.4f}, {pref_lif.max():.4f}]")

    f_lo = min(f_min_hrf, f_min_lif) * 0.9
    f_hi = max(f_max_hrf, f_max_lif) * 1.1
    bins = np.logspace(np.log10(f_lo), np.log10(f_hi), 25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    axes[0].hist(pref_hrf, bins=bins, color='#2196F3', alpha=0.8,
                 edgecolor='black', linewidth=0.5)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Preferred frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Number of neurons', fontsize=12)
    axes[0].set_title(f's-RON (HRF) — {dataset}\n'
                      f'mean={pref_hrf.mean():.4f}, std={pref_hrf.std():.4f} Hz',
                      fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(pref_lif, bins=bins, color='#FF9800', alpha=0.8,
                 edgecolor='black', linewidth=0.5)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Preferred frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Number of neurons', fontsize=12)
    axes[1].set_title(f'LIF-RC — {dataset}\n'
                      f'mean={pref_lif.mean():.4f}, std={pref_lif.std():.4f} Hz',
                      fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Frequency Selectivity of Reservoir Neurons', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f'freq_selectivity_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pref_hrf, bins=bins, color='#2196F3', alpha=0.6, edgecolor='black',
            linewidth=0.3,
            label=f's-RON (HRF)  mean={pref_hrf.mean():.4f}, std={pref_hrf.std():.4f}')
    ax.hist(pref_lif, bins=bins, color='#FF9800', alpha=0.6, edgecolor='black',
            linewidth=0.3,
            label=f'LIF-RC  mean={pref_lif.mean():.4f}, std={pref_lif.std():.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('Preferred frequency (Hz)', fontsize=12)
    ax.set_ylabel('Number of neurons', fontsize=12)
    ax.set_title(f'Frequency Selectivity — {dataset}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(out_dir, f'freq_selectivity_{dataset}_overlay.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    return {
        'pref_freq_hrf_mean': float(pref_hrf.mean()),
        'pref_freq_hrf_std':  float(pref_hrf.std()),
        'pref_freq_lif_mean': float(pref_lif.mean()),
        'pref_freq_lif_std':  float(pref_lif.std()),
        'freqs_hrf': freqs_hrf,
        'freqs_lif': freqs_lif,
        # Per-neuron arrays — needed for accurate histogram plotting
        'pref_freq_hrf_array': pref_hrf.tolist(),
        'pref_freq_lif_array': pref_lif.tolist(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Representational analysis: HRF vs LIF reservoirs'
    )
    parser.add_argument('--dataset', required=True,
                        choices=['fordA', 'sMNIST', 'shd', 'dvs_gesture'])
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory (needed for SHD and DVS Gesture)')
    parser.add_argument('--analysis', nargs='+',
                        default=['cka', 'eff_dim', 'freq_selectivity'],
                        choices=['cka', 'eff_dim', 'freq_selectivity'],
                        help='Which analyses to run')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--cpu',            action='store_true')
    parser.add_argument('--max_samples_cka',type=int,   default=500,
                        help='Max test samples for CKA (<=500 for speed)')
    parser.add_argument('--f_min',          type=float, default=None,
                        help='Min frequency for sweep (Hz). '
                             'Default 0.05. Use ~0.01 for SHD/DVS slow dynamics.')
    parser.add_argument('--T_sweep',        type=int,   default=None,
                        help='Timesteps per frequency in sweep. '
                             'Default 5000. Use 8000-10000 for low-freq resolution.')
    args = parser.parse_args()

    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'analysis_results', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(f'REPRESENTATIONAL ANALYSIS  |  dataset={args.dataset}')
    print(f'Analyses: {args.analysis}')
    print(f'Device:   {device}')
    print(f'Output:   {out_dir}')
    if args.f_min:
        print(f'f_min override:    {args.f_min} Hz')
    if args.T_sweep:
        print(f'T_sweep override:  {args.T_sweep}')
    print('=' * 70)

    # Reset seed independently before each model so that the random
    # initialization of one model does not affect the other.
    # This ensures reproducible and comparable results across runs.
    print('\nBuilding HRF model (s-RON)...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    hrf_model, hrf_cfg = build_hrf_model(args.dataset, device, seed=args.seed)
    apply_sparse_input_projection(
        hrf_model, hrf_cfg['input_density'],
        hrf_cfg['n_inp'], hrf_cfg['n_hid'], device)

    print('Building LIF model (LIF-RC)...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    lif_model, lif_cfg = build_lif_model(args.dataset, device, seed=args.seed)
    apply_sparse_input_projection(
        lif_model, lif_cfg['input_density'],
        lif_cfg['n_inp'], lif_cfg['n_hid'], device)

    all_results = {'dataset': args.dataset, 'seed': args.seed}

    if 'cka' in args.analysis or 'eff_dim' in args.analysis:
        print('\nLoading dataset...')
        loader, n_inp, needs_reshape = load_dataset(args.dataset, args, device)

        if 'cka' in args.analysis:
            res = run_cka(hrf_model, lif_model, loader, device,
                          needs_reshape, args.dataset, out_dir,
                          max_samples=args.max_samples_cka)
            all_results['cka'] = res

        if 'eff_dim' in args.analysis:
            res = run_eff_dim(hrf_model, lif_model, loader, device,
                              needs_reshape, args.dataset, out_dir)
            all_results['eff_dim'] = res

    if 'freq_selectivity' in args.analysis:
        res = run_freq_selectivity(
            hrf_model, lif_model,
            dataset=args.dataset,
            dt_hrf=hrf_cfg['dt'], dt_lif=lif_cfg['dt'],
            n_inp=hrf_cfg['n_inp'],
            n_hid_hrf=hrf_cfg['n_hid'], n_hid_lif=lif_cfg['n_hid'],
            device=device, out_dir=out_dir,
            f_min_override=args.f_min,
            T_sweep_override=args.T_sweep,
        )
        all_results['freq_selectivity'] = res

    summary_path = os.path.join(out_dir, f'analysis_summary_{args.dataset}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    if 'cka' in all_results:
        r = all_results['cka']
        print(f"CKA       — HRF: {r['cka_hrf']:.4f}  |  LIF: {r['cka_lif']:.4f}")
    if 'eff_dim' in all_results:
        r = all_results['eff_dim']
        print(f"Eff. Dim  — HRF: {r['pr_hrf']:.1f} ({r['pr_frac_hrf']*100:.2f}%)  "
              f"|  LIF: {r['pr_lif']:.1f} ({r['pr_frac_lif']*100:.2f}%)")
    if 'freq_selectivity' in all_results:
        r = all_results['freq_selectivity']
        print(f"Freq. std — HRF: {r['pref_freq_hrf_std']:.4f} Hz  "
              f"|  LIF: {r['pref_freq_lif_std']:.4f} Hz")
    print(f'\nAll results saved to: {out_dir}')


if __name__ == '__main__':
    main()

'''










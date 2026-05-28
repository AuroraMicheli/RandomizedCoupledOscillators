"""
Readout ablation script for s-RON (LIF-HRF).

Runs two ablation tables in a single job per dataset:

  Table A — Readout aggregation ablation
  ---------------------------------------
  How the HRF membrane state hy is pooled over time.
  Single-statistic modes (final/mean/rms/std) have identical feature
  dimension n_hid and identical readout parameter counts.
  rms_std_final has 3*n_hid features (noted in table).
  Modes: final | mean | rms | std | rms_std_final

  Table B — Readout signal ablation (membrane vs spikes)
  -------------------------------------------------------
  Same aggregation (temporal mean), different signal:
    membrane -> mean of hy over time       (n_hid features)
    spikes   -> mean spike rate per neuron (n_hid features)
  Identical feature dimension and readout parameter count.
  Modes: mean | spikes_mean

  NOTE: 'mean' is shared between Table A and Table B.
  It runs once and its result is reused for both tables.

Key design:
  - Seed is reset to the same value before building each model within a
    trial, so all modes share identical reservoir weights -> fair comparison.
  - sMNIST and fordA are re-run (not reused from previous results) so that
    spikes_mean is included and all results share the same seed/setup.

Supports: fordA | shd | sMNIST | dvs_gesture

Usage:
    python readout_ablation.py --dataset sMNIST      --use_test --test_trials 3
    python readout_ablation.py --dataset fordA       --use_test --test_trials 3
    python readout_ablation.py --dataset shd         --use_test --test_trials 3 --data_dir data/SHD
    python readout_ablation.py --dataset dvs_gesture --use_test --test_trials 3 --data_dir data/DVSGesture

Results saved to ablation_readout/ as one JSON per dataset.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from utils_aurora import spiking_coESN_rescaled_II

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
# Best s-RON configs per dataset.
# Taken directly from your DATASET_CONFIGS / best-config scripts.
# gamma/epsilon stored as center + range (matching your script args).
# Converted to (min, max) for the model constructor via
# gamma_epsilon_to_range() below.
#
# connectivity_lif2hrf=0.2 for all datasets (matches your actual runs).
# readout_C is a dict so each mode can have its own value.
# spikes_mean starts with the same C — adjust if results look poor.
# =============================================================================


BEST_CONFIGS = {
    'sMNIST': dict(
        n_hid                = 800,
        dt                   = 0.042,
        gamma                = 2.7,
        gamma_range          = 2.0,
        epsilon              = 0.08,
        epsilon_range        = 1.0,
        rho                  = 0.99,
        inp_scaling          = 2.0,
        theta_lif            = 0.05,
        theta_rf             = 0.005,
        tau_filter           = 20.0,
        connectivity_lif2hrf = 0.2,
        connectivity_hrf2lif = 1.0,
        input_density        = 1.0,
        num_steps            = 784,
        max_time             = 1.4,
        spatial_factor       = 4,
        readout_C = {
            'final':           1.0,
            'mean':            1.0,
            'rms':             1.0,
            'std':             1.0,
            'rms_std_final':   1.0,
            'spikes_mean':     1.0,
        },
    ),

    'fordA': dict(
        n_hid                = 800,
        dt                   = 0.051,
        gamma                = 7.0124,
        gamma_range          = 3.01,
        epsilon              = 0.1528,
        epsilon_range        = 0.419,
        rho                  = 0.75,
        inp_scaling          = 0.6247,
        theta_lif            = 0.0824,
        theta_rf             = 0.0010,
        tau_filter           = 6.1,
        connectivity_lif2hrf = 0.2,
        connectivity_hrf2lif = 1.0,
        input_density        = 1.0,
        num_steps            = 500,
        max_time             = 1.4,
        spatial_factor       = 4,
        readout_C = {
            'final':           0.1,
            'mean':            0.1,
            'rms':             0.1,
            'std':             0.1,
            'rms_std_final':   0.1,
            'spikes_mean':     0.1,
        },
    ),

    'shd': dict(
        n_hid                = 3000,
        dt                   = 0.223,
        gamma                = 0.036,
        gamma_range          = 0.268,
        epsilon              = 0.06,
        epsilon_range        = 0.063,
        rho                  = 1.16,
        inp_scaling          = 0.23,
        theta_lif            = 1.0,
        theta_rf             = 0.013,
        tau_filter           = 20.0,
        connectivity_lif2hrf = 0.2,
        connectivity_hrf2lif = 1.0,
        input_density        = 0.036,
        num_steps            = 250,
        max_time             = 1.4,
        spatial_factor       = 4,
        readout_C = {
            'final':           0.01,
            'mean':            0.01,
            'rms':             0.01,
            'std':             0.01,
            'rms_std_final':   0.01,
            'spikes_mean':     0.01,
        },
    ),

    'dvs_gesture': dict(
        n_hid                = 3000,
        dt                   = 0.259,
        gamma                = 0.0456,
        gamma_range          = 0.1304,
        epsilon              = 0.0354,
        epsilon_range        = 0.0989,
        rho                  = 1.581,
        inp_scaling          = 0.1129,
        theta_lif            = 2.9678,
        theta_rf             = 0.03628,
        tau_filter           = 20.0,
        connectivity_lif2hrf = 0.2,
        connectivity_hrf2lif = 1.0,
        input_density        = 0.0306,
        num_steps            = 200,
        max_time             = 1.4,
        spatial_factor       = 4,
        readout_C = {
            'final':           0.01,
            'mean':            0.01,
            'rms':             0.01,
            'std':             0.01,
            'rms_std_final':   0.01,
            'spikes_mean':     0.01,
        },
    ),
}


def gamma_epsilon_to_range(center, rng):
    """Convert center + range to (min, max) tuple for model constructor."""
    lo = max(center - rng / 2.0, 1e-6)
    hi = center + rng / 2.0
    return (lo, hi)


# =============================================================================
# Modes
# =============================================================================

TABLE_A_MODES = ['final', 'mean', 'rms', 'std', 'rms_std_final']
TABLE_B_MODES = ['mean', 'spikes_mean']
ALL_MODES     = list(dict.fromkeys(TABLE_A_MODES + TABLE_B_MODES))
# -> ['final', 'mean', 'rms', 'std', 'rms_std_final', 'spikes_mean']
# 'mean' runs once and is reused for both Table A and Table B


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
    scale          = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale
    return int(mask.sum().item())


# =============================================================================
# Dataset loaders
# =============================================================================

def load_fordA(args, device):
    n_inp, n_out = 1, 2
    train_loader, _, test_loader = get_FordA_data(args.batch, 120, whole_train=True)
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"FordA: seq={seq_length}, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, 1.0


def load_shd(args, device):
    n_inp, n_out = 700, 20
    train_loader, _, test_loader = get_SHD_data(
        batch_train=args.batch, batch_test=256,
        data_dir=args.data_dir, num_steps=args.num_steps,
        max_time=args.max_time,
    )
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"SHD: seq={seq_length}, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


def load_smnist(args, device):
    n_inp, n_out = 1, 10
    train_loader, _, test_loader = get_mnist_data(args.batch, 100)
    print(f"sMNIST: seq=784, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, 784, 1.0


def load_dvs_gesture(args, device):
    assert TONIC_AVAILABLE, "pip install tonic"
    sensor_size_orig = tonic.datasets.DVSGesture.sensor_size
    H_orig, W_orig, C = sensor_size_orig[1], sensor_size_orig[0], sensor_size_orig[2]
    sf               = args.spatial_factor
    H_ds, W_ds       = H_orig // sf, W_orig // sf
    n_inp, n_out     = C * H_ds * W_ds, 11

    frame_transform = tonic_transforms.ToFrame(
        sensor_size=sensor_size_orig, n_time_bins=args.num_steps
    )

    def collate_fn(batch):
        xs, ys = [], []
        for frames, label in batch:
            t = torch.tensor(frames, dtype=torch.float32)
            if sf > 1:
                T_ = t.size(0)
                t  = t.view(T_*C, 1, H_orig, W_orig)
                t  = F.avg_pool2d(t, kernel_size=sf, stride=sf)
                t  = t.view(T_, C, H_ds, W_ds)
            t = t.reshape(t.size(0), -1)
            t = (t > 0).float()
            xs.append(t); ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    os.makedirs(args.data_dir, exist_ok=True)
    train_ds = DiskCachedDataset(
        tonic.datasets.DVSGesture(
            save_to=args.data_dir, train=True, transform=frame_transform),
        cache_path=os.path.join(
            args.data_dir, f'cache_train_T{args.num_steps}_sf{sf}'),
    )
    test_ds = DiskCachedDataset(
        tonic.datasets.DVSGesture(
            save_to=args.data_dir, train=False, transform=frame_transform),
        cache_path=os.path.join(
            args.data_dir, f'cache_test_T{args.num_steps}_sf{sf}'),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"DVS Gesture: {len(train_ds)} train, {len(test_ds)} test, "
          f"n_inp={n_inp}, seq={seq_length}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


DATASET_LOADERS = {
    'fordA':       load_fordA,
    'shd':         load_shd,
    'sMNIST':      load_smnist,
    'dvs_gesture': load_dvs_gesture,
}


# =============================================================================
# Feature extraction and readout
# =============================================================================

def extract_features(model, loader, device, needs_reshape, split_name):
    model.eval()
    feats, labels_all = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, ncols=80, desc=f"  [{split_name}]"):
            xb = xb.to(device)
            if needs_reshape:
                xb = xb.reshape(xb.shape[0], 1, 784).permute(0, 2, 1)
            features, _ = model(xb)
            feats.append(features.cpu())
            labels_all.append(yb)
    return (torch.cat(feats, dim=0).numpy(),
            torch.cat(labels_all, dim=0).numpy())


def fit_readout(train_feats, train_labels, test_feats, test_labels,
                readout_C, dataset_name):
    scaler      = preprocessing.StandardScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    test_feats  = scaler.transform(test_feats)

    # n_jobs=1 prevents OOM: parallel workers each copy the full feature matrix.
    # With rms_std_final (3*n_hid features) on large datasets, n_jobs=-1
    # causes SIGKILL. Single-threaded is slower but safe.
    if dataset_name in ('shd', 'dvs_gesture'):
        clf = LogisticRegression(
            max_iter=2000, verbose=0, n_jobs=1,
            multi_class='multinomial', solver='lbfgs',
            C=readout_C,
        ).fit(train_feats, train_labels)
    else:
        clf = LogisticRegression(
            max_iter=2000, verbose=0, n_jobs=1,
            C=readout_C,
        ).fit(train_feats, train_labels)

    train_acc = clf.score(train_feats, train_labels) * 100
    test_acc  = clf.score(test_feats,  test_labels)  * 100
    n_params  = int(clf.coef_.size)
    return train_acc, test_acc, n_params


# =============================================================================
# Argument parser
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description='Readout ablation: Table A (aggregation) + '
                    'Table B (membrane mean vs spike mean)'
    )
    p.add_argument('--dataset', required=True,
                   choices=['fordA', 'shd', 'sMNIST', 'dvs_gesture'])
    p.add_argument('--data_dir',    type=str, default='data')
    p.add_argument('--batch',       type=int, default=128)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--test_trials', type=int, default=3)
    p.add_argument('--use_test',    action='store_true')
    p.add_argument('--cpu',         action='store_true')
    p.add_argument('--results_dir', type=str, default=None)
    return p


# =============================================================================
# Main
# =============================================================================

def main():
    args   = build_parser().parse_args()
    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    if args.results_dir is None:
        script_dir       = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(script_dir, 'ablation_readout')
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = BEST_CONFIGS[args.dataset]

    # Push dataset-specific params into args for loaders
    args.input_density  = cfg['input_density']
    args.num_steps      = cfg['num_steps']
    args.max_time       = cfg['max_time']
    args.spatial_factor = cfg['spatial_factor']

    # Convert gamma/epsilon center+range -> (min, max) for model constructor
    gamma_tuple   = gamma_epsilon_to_range(cfg['gamma'],   cfg['gamma_range'])
    epsilon_tuple = gamma_epsilon_to_range(cfg['epsilon'], cfg['epsilon_range'])

    print('=' * 70)
    print(f'READOUT ABLATION  —  {args.dataset}')
    print(f'Table A modes: {TABLE_A_MODES}')
    print(f'Table B modes: {TABLE_B_MODES}  (mean is shared)')
    print(f'Trials: {args.test_trials}   Seed: {args.seed}   Device: {device}')
    print(f'n_hid:  {cfg["n_hid"]}')
    print(f'connectivity_lif2hrf: {cfg["connectivity_lif2hrf"]}')
    print(f'gamma:   {gamma_tuple}')
    print(f'epsilon: {epsilon_tuple}')
    print('=' * 70)

    loader_fn = DATASET_LOADERS[args.dataset]
    (train_loader, test_loader,
     n_inp, n_out, seq_length, input_density) = loader_fn(args, device)

    needs_reshape = (args.dataset == 'sMNIST')

    # Per-mode result accumulator
    results = {mode: {'test_accs': [], 'train_accs': [], 'n_params': None}
               for mode in ALL_MODES}

    for trial in range(args.test_trials):
        print(f"\n{'='*70}")
        print(f"TRIAL {trial+1}/{args.test_trials}")
        print(f"{'='*70}")

        for mode in ALL_MODES:
            print(f"\n--- Mode: {mode} ---")

            # Reset seed before every model build within this trial:
            # all modes share identical reservoir weights -> fair comparison
            set_seed(args.seed + trial)

            model = spiking_coESN_rescaled_II(
                n_inp        = n_inp,
                n_hid        = cfg['n_hid'],
                dt           = cfg['dt'],
                gamma        = gamma_tuple,
                epsilon      = epsilon_tuple,
                rho          = cfg['rho'],
                input_scaling= cfg['inp_scaling'],
                theta_lif    = cfg['theta_lif'],
                theta_rf     = cfg['theta_rf'],
                tau_filter   = cfg['tau_filter'],
                sparse_lif2hrf       = (cfg['connectivity_lif2hrf'] < 1.0),
                connectivity_lif2hrf = cfg['connectivity_lif2hrf'],
                sparse_hrf2lif       = (cfg['connectivity_hrf2lif'] < 1.0),
                connectivity_hrf2lif = cfg['connectivity_hrf2lif'],
                device       = device,
                readout_mode = mode,
            ).to(device)

            if input_density < 1.0:
                apply_sparse_input_projection(
                    model, input_density, n_inp, cfg['n_hid'], device
                )

            train_feats, train_labels = extract_features(
                model, train_loader, device, needs_reshape, 'train'
            )
            print(f"  features: {train_feats.shape}")

            if args.use_test:
                test_feats, test_labels = extract_features(
                    model, test_loader, device, needs_reshape, 'test'
                )
            else:
                test_feats, test_labels = train_feats, train_labels

            readout_C = cfg['readout_C'][mode]
            train_acc, test_acc, n_params = fit_readout(
                train_feats, train_labels,
                test_feats,  test_labels,
                readout_C, args.dataset,
            )
            print(f"  train: {train_acc:.2f}%   test: {test_acc:.2f}%   "
                  f"params: {n_params:,}   C: {readout_C}")

            results[mode]['test_accs'].append(test_acc)
            results[mode]['train_accs'].append(train_acc)
            results[mode]['n_params'] = n_params

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregated = {}
    for mode in ALL_MODES:
        r = results[mode]
        aggregated[mode] = {
            'test_mean':  float(np.mean(r['test_accs'])),
            'test_std':   float(np.std(r['test_accs'])),
            'train_mean': float(np.mean(r['train_accs'])),
            'train_std':  float(np.std(r['train_accs'])),
            'n_params':   r['n_params'],
            'test_accs':  [float(a) for a in r['test_accs']],
            'train_accs': [float(a) for a in r['train_accs']],
        }

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TABLE A — Readout aggregation  ({args.dataset})")
    print(f"{'='*70}")
    print(f"  {'Mode':<22} {'Test acc':>14}   {'Params':>10}")
    for mode in TABLE_A_MODES:
        a = aggregated[mode]
        print(f"  {mode:<22} "
              f"{a['test_mean']:6.2f} ± {a['test_std']:<5.2f}  "
              f"{a['n_params']:>10,}")

    print(f"\n{'='*70}")
    print(f"TABLE B — Membrane mean vs Spike mean  ({args.dataset})")
    print(f"{'='*70}")
    for label, mode in [('membrane (mean hy)   ', 'mean'),
                         ('spikes   (mean rate) ', 'spikes_mean')]:
        a = aggregated[mode]
        print(f"  {label}  "
              f"{a['test_mean']:.2f} ± {a['test_std']:.2f}%   "
              f"params={a['n_params']:,}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'dataset':       args.dataset,
        'n_hid':         cfg['n_hid'],
        'n_trials':      args.test_trials,
        'base_seed':     args.seed,
        'seq_length':    int(seq_length),
        'table_a_modes': TABLE_A_MODES,
        'table_b_modes': TABLE_B_MODES,
        'results':       aggregated,
        'config':        {k: v for k, v in cfg.items() if k != 'readout_C'},
    }

    fname = (f"readout_ablation_{args.dataset}"
             f"_nhid{cfg['n_hid']}"
             f"_trials{args.test_trials}"
             f"_seed{args.seed}.json")
    fpath = os.path.join(args.results_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {fpath}")


if __name__ == '__main__':
    main()
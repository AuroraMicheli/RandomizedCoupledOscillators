"""
Readout ablation study for s-RON: compare all readout modes across datasets.

Supported datasets: sMNIST, psMNIST, FordA, Adiac, npCIFAR10
Supported readout modes: final, mean, rms, std, rms_std_final

Usage:
    python train_readout_ablation.py --dataset sMNIST --readout_mode rms --use_test
    python train_readout_ablation.py --dataset FordA  --readout_mode std --use_test
    python train_readout_ablation.py --dataset Adiac  --readout_mode rms_std_final --use_test
    python train_readout_ablation.py --dataset psMNIST --readout_mode mean --use_test
    python train_readout_ablation.py --dataset npCIFAR10 --readout_mode rms --use_test

All dataset-specific hyperparameters are hardcoded from the best configurations
found during hyperparameter search. They can be overridden via CLI flags.

Results saved to: results_readout_ablation/<dataset>_<readout_mode>_nhid<N>_...json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse
from utils import get_mnist_data, get_FordA_data, get_Adiac_data, get_cifar_data


# =============================================================================
# Per-dataset best hyperparameter configs (n_hid=800, connectivity_lif2hrf=0.2)
# =============================================================================

DATASET_CONFIGS = {
    'sMNIST': dict(
        n_hid=800, dt=0.042,
        gamma=2.7,      gamma_range=2.0,
        epsilon=0.08,   epsilon_range=1.0,
        inp_scaling=2.0, rho=0.99,
        theta_lif=0.05, theta_rf=0.005, tau_filter=20.0,
        connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
        readout_C=1.0,
        n_inp=1, n_out=10, seq_length=784,
        batch=256, bs_test=100,
    ),
    'psMNIST': dict(
        n_hid=800, dt=0.047,
        gamma=2.62,     gamma_range=3.84,
        epsilon=0.24,   epsilon_range=1.86,
        inp_scaling=3.67, rho=1.55,
        theta_lif=0.05, theta_rf=0.005, tau_filter=20.0,
        connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
        readout_C=1.0,
        n_inp=1, n_out=10, seq_length=784,
        batch=256, bs_test=100,
        perm_seed=0,
    ),
    'FordA': dict(
        n_hid=800, dt=0.051,
        gamma=7.0124,   gamma_range=3.01,
        epsilon=0.1528, epsilon_range=0.419,
        inp_scaling=0.6247, rho=0.75,
        theta_lif=0.0824, theta_rf=0.0010, tau_filter=6.1,
        connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
        readout_C=0.1,
        n_inp=1, n_out=2, seq_length=None,  # detected from data
        batch=120, bs_test=120,
    ),
    'Adiac': dict(
        n_hid=800, dt=0.2213,
        gamma=1.3770,   gamma_range=3.5954,
        epsilon=0.01985, epsilon_range=0.2027,
        inp_scaling=13.0135, rho=0.8131,
        theta_lif=0.02290, theta_rf=0.01521, tau_filter=13.5399,
        connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
        readout_C=0.1,
        n_inp=1, n_out=37, seq_length=None,  # detected from data
        batch=120, bs_test=30,
    ),
    'npCIFAR10': dict(
    n_hid=800, dt=0.05480230405417246,
    gamma=0.11521922168692778,  gamma_range=1.424188539771601,
    epsilon=0.17793795333634432, epsilon_range=0.9628236973290385,
    inp_scaling=0.058263838195044056, rho=1.467075905961671,
    theta_lif=0.12421806046776554, theta_rf=0.15354214189263757,
    tau_filter=67.02898617235653,
    connectivity_lif2hrf=0.2, connectivity_hrf2lif=1.0,
    readout_C=1.0,
    n_inp=96, n_out=10, seq_length=1000,
    batch=100, bs_test=100,
),
}

READOUT_MODES = ['final', 'mean', 'rms', 'std', 'rms_std_final']


# =============================================================================
# Seed
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Feature dimension helper
# =============================================================================

def get_n_features(n_hid, readout_mode):
    """Return feature dimension for a given readout mode and reservoir size."""
    return n_hid * 3 if readout_mode == 'rms_std_final' else n_hid


# =============================================================================
# Data loaders
# =============================================================================

def load_data(dataset, cfg, device):
    """
    Load train/test loaders and any dataset-specific preprocessing state.
    Returns: train_loader, test_loader, extras (dict of dataset-specific items)
    """
    extras = {}

    if dataset == 'sMNIST':
        train_loader, _, test_loader = get_mnist_data(cfg['batch'], cfg['bs_test'])

    elif dataset == 'psMNIST':
        train_loader, _, test_loader = get_mnist_data(cfg['batch'], cfg['bs_test'])
        perm_rng = torch.Generator()
        perm_rng.manual_seed(cfg['perm_seed'])
        extras['perm'] = torch.randperm(784, generator=perm_rng).to(device)
        print(f"  Fixed permutation generated with perm_seed={cfg['perm_seed']}")

    elif dataset == 'FordA':
        train_loader, _, test_loader = get_FordA_data(
            cfg['batch'], cfg['bs_test'], whole_train=True)

    elif dataset == 'Adiac':
        train_loader, _, test_loader = get_Adiac_data(
            cfg['batch'], cfg['bs_test'], whole_train=True)

    elif dataset == 'npCIFAR10':
        train_loader, _, test_loader = get_cifar_data(cfg['batch'], cfg['bs_test'])
        # Fixed random padding shared across all trials
        extras['rand_pad'] = torch.randn(
            1, cfg['seq_length'] - 32, cfg['n_inp']
        ).to(device)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_loader, test_loader, extras


# =============================================================================
# Feature extraction
# =============================================================================

def extract_features(loader, model, device, dataset, extras):
    """
    Extract reservoir features and firing rates from a loader.
    Handles all dataset-specific input preprocessing.
    """
    model.eval()
    feats, labels_all = [], []
    r_tot_list, r_hrf_list, r_lif_list = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=80, desc="  Extracting", leave=False):
            x = x.to(device)

            # Dataset-specific input preprocessing
            if dataset == 'sMNIST':
                # (B, 1, 28, 28) -> (B, 784, 1)
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)

            elif dataset == 'psMNIST':
                # (B, 1, 28, 28) -> (B, 784, 1) -> permute pixels
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
                x = x[:, extras['perm'], :]

            elif dataset == 'npCIFAR10':
                # (B, 3, 32, 32) -> row-wise (B, 32, 96), then pad to (B, 1000, 96)
                B = x.shape[0]
                x = torch.cat(
                    (x.permute(0, 2, 1, 3).reshape(B, 32, 96),
                     extras['rand_pad'].expand(B, -1, -1)),
                    dim=1
                )
            # FordA and Adiac: input already in (B, T, n_inp) format

            features, r = model(x)
            feats.append(features.cpu())
            r_tot_list.append(r['r_total'])
            r_hrf_list.append(r['r_hrf'])
            r_lif_list.append(r['r_lif'])
            labels_all.append(y)

    feats      = torch.cat(feats,      dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    r_hrf = torch.stack(r_hrf_list).mean().item()
    r_lif = torch.stack(r_lif_list).mean().item()
    r_tot = torch.stack(r_tot_list).mean().item()
    return feats, labels_all, r_tot, r_hrf, r_lif


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Readout ablation study for s-RON'
    )

    # Required
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run on')
    parser.add_argument('--readout_mode', required=True,
                        choices=READOUT_MODES,
                        help='Readout mode to evaluate')

    # Override dataset defaults if needed
    parser.add_argument('--n_hid',              type=int,   default=None)
    parser.add_argument('--dt',                 type=float, default=None)
    parser.add_argument('--gamma',              type=float, default=None)
    parser.add_argument('--gamma_range',        type=float, default=None)
    parser.add_argument('--epsilon',            type=float, default=None)
    parser.add_argument('--epsilon_range',      type=float, default=None)
    parser.add_argument('--inp_scaling',        type=float, default=None)
    parser.add_argument('--rho',                type=float, default=None)
    parser.add_argument('--theta_lif',          type=float, default=None)
    parser.add_argument('--theta_rf',           type=float, default=None)
    parser.add_argument('--tau_filter',         type=float, default=None)
    parser.add_argument('--connectivity_lif2hrf', type=float, default=None)
    parser.add_argument('--connectivity_hrf2lif', type=float, default=None)
    parser.add_argument('--readout_C',          type=float, default=None)
    parser.add_argument('--batch',              type=int,   default=None)
    parser.add_argument('--perm_seed',          type=int,   default=None,
                        help='Permutation seed for psMNIST (default: 0)')

    # Experiment options
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--test_trials', type=int, default=3)
    parser.add_argument('--use_test',    action='store_true')
    parser.add_argument('--cpu',         action='store_true')
    parser.add_argument('--results_dir', type=str, default='results_readout_ablation')

    args = parser.parse_args()

    # Merge dataset defaults with any CLI overrides
    cfg = dict(DATASET_CONFIGS[args.dataset])  # copy
    for key in ['n_hid', 'dt', 'gamma', 'gamma_range', 'epsilon', 'epsilon_range',
                'inp_scaling', 'rho', 'theta_lif', 'theta_rf', 'tau_filter',
                'connectivity_lif2hrf', 'connectivity_hrf2lif', 'readout_C', 'batch']:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.perm_seed is not None:
        cfg['perm_seed'] = args.perm_seed

    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    n_features         = get_n_features(cfg['n_hid'], args.readout_mode)
    n_trainable_params = n_features * cfg['n_out']

    print('=' * 70)
    print(f'READOUT ABLATION  |  dataset={args.dataset}  '
          f'readout_mode={args.readout_mode}')
    print('=' * 70)
    print(f'  n_hid:            {cfg["n_hid"]}')
    print(f'  feature dim:      {n_features}')
    print(f'  trainable params: {n_trainable_params:,}  '
          f'[= {n_features} x {cfg["n_out"]}]')
    print(f'  connectivity_lif2hrf: {cfg["connectivity_lif2hrf"]}')
    print(f'  trials:           {args.test_trials}')
    print(f'  device:           {device}')
    print('=' * 70)

    # Load data
    print('\nLoading data...')
    train_loader, test_loader, extras = load_data(args.dataset, cfg, device)

    # Detect seq_length from data if not hardcoded (FordA, Adiac)
    if cfg['seq_length'] is None:
        sample_x, _ = next(iter(train_loader))
        cfg['seq_length'] = sample_x.shape[1]
        print(f'  seq_length detected from data: {cfg["seq_length"]}')

    gamma   = (cfg['gamma']   - cfg['gamma_range']   / 2.,
               cfg['gamma']   + cfg['gamma_range']   / 2.)
    epsilon = (cfg['epsilon'] - cfg['epsilon_range'] / 2.,
               cfg['epsilon'] + cfg['epsilon_range'] / 2.)

    sparse_lif2hrf = cfg['connectivity_lif2hrf'] < 1.0
    sparse_hrf2lif = cfg['connectivity_hrf2lif'] < 1.0

    all_test_accs, all_train_accs, all_energies = [], [], []
    all_sops, all_sops_hrf, all_sops_lif = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial + 1}/{args.test_trials}\n{'='*70}")
        set_seed(args.seed + trial)

        model = spiking_coESN_rescaled_II(
            n_inp=cfg['n_inp'], n_hid=cfg['n_hid'], dt=cfg['dt'],
            gamma=gamma, epsilon=epsilon, rho=cfg['rho'],
            input_scaling=cfg['inp_scaling'],
            theta_lif=cfg['theta_lif'], theta_rf=cfg['theta_rf'],
            tau_filter=cfg['tau_filter'],
            sparse_lif2hrf=sparse_lif2hrf,
            connectivity_lif2hrf=cfg['connectivity_lif2hrf'],
            sparse_hrf2lif=sparse_hrf2lif,
            connectivity_hrf2lif=cfg['connectivity_hrf2lif'],
            device=device,
            readout_mode=args.readout_mode,
        ).to(device)

        # Extract train features
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = \
            extract_features(train_loader, model, device, args.dataset, extras)
        print(f'  train features: {train_feats.shape}')

        # Extract test features
        if args.use_test:
            test_feats, test_labels, _, _, _ = \
                extract_features(test_loader, model, device, args.dataset, extras)
            print(f'  test features:  {test_feats.shape}')
        else:
            test_feats, test_labels = train_feats, train_labels

        # Replace inf/nan with 0 before scaling — numerical instability from
        # a small number of neurons with large transient activations.
        # Applies mainly to rms and rms_std_final which square hy values,
        # amplifying any transient large activations to inf.
        
        train_feats = np.where(np.isfinite(train_feats), train_feats, 0.0)
        test_feats  = np.where(np.isfinite(test_feats),  test_feats,  0.0)
        # Scale
        scaler      = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        # Readout
        clf = LogisticRegression(
            max_iter=5000, verbose=0, n_jobs=-1, C=cfg['readout_C']
        ).fit(train_feats, train_labels)

        train_acc = clf.score(train_feats, train_labels) * 100.0
        test_acc  = clf.score(test_feats,  test_labels)  * 100.0
        print(f'  train acc: {train_acc:.2f}%  test acc: {test_acc:.2f}%')
        print(f'  r_hrf={r_hrf_train:.4f}  r_lif={r_lif_train:.4f}')

        # Energy
        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_hrf_train, r_lif=r_lif_train,
            n_hid=cfg['n_hid'], T=cfg['seq_length'],
            lif2hrf_connections=model.n_lif2hrf_connections,
            include_lif=True,
        )
        print(f'  energy: {snn_energy["Energy_J"]:.3e} J  '
              f'SOPs: {snn_energy["SOPs"]:.3e}')

        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_sops.append(snn_energy['SOPs'])
        all_sops_hrf.append(snn_energy['HRF_SOPs'])
        all_sops_lif.append(snn_energy['LIF_SOPs'])
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    # Aggregate
    mean_test_acc  = float(np.mean(all_test_accs))
    std_test_acc   = float(np.std(all_test_accs))
    mean_train_acc = float(np.mean(all_train_accs))
    std_train_acc  = float(np.std(all_train_accs))
    mean_energy    = float(np.mean(all_energies))
    std_energy     = float(np.std(all_energies))

    print(f'\n{"="*70}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"="*70}')
    print(f'  Dataset:          {args.dataset}')
    print(f'  Readout mode:     {args.readout_mode}')
    print(f'  Feature dim:      {n_features}')
    print(f'  Trainable params: {n_trainable_params:,}')
    print(f'  n_hid:            {cfg["n_hid"]},  trials: {args.test_trials}')
    print(f'  Train acc:        {mean_train_acc:.2f}% ± {std_train_acc:.2f}%')
    print(f'  Test acc:         {mean_test_acc:.2f}% ± {std_test_acc:.2f}%')
    print(f'  Per-trial:        {[f"{a:.2f}" for a in all_test_accs]}')
    print(f'  Energy:           {mean_energy:.3e} ± {std_energy:.3e} J')
    print(f'{"="*70}')

    results = {
        'dataset':            args.dataset,
        'readout_mode':       args.readout_mode,
        'n_hid':              cfg['n_hid'],
        'n_features':         n_features,
        'n_trainable_params': n_trainable_params,
        'n_trials':           args.test_trials,
        'seed':               args.seed,
        'cfg':                cfg,
        'test_acc_mean':      mean_test_acc,
        'test_acc_std':       std_test_acc,
        'train_acc_mean':     mean_train_acc,
        'train_acc_std':      std_train_acc,
        'test_accs_all':      [float(x) for x in all_test_accs],
        'energy_J_mean':      mean_energy,
        'energy_J_std':       std_energy,
        'sops_mean':          float(np.mean(all_sops)),
        'sops_hrf_mean':      float(np.mean(all_sops_hrf)),
        'sops_lif_mean':      float(np.mean(all_sops_lif)),
        'r_hrf_mean':         float(np.mean(all_r_hrf)),
        'r_lif_mean':         float(np.mean(all_r_lif)),
        'r_tot_mean':         float(np.mean(all_r_total)),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    fname = (
        f"{args.dataset.lower()}_{args.readout_mode}"
        f"_nhid{cfg['n_hid']}"
        f"_lif{cfg['connectivity_lif2hrf']}"
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    path = os.path.join(args.results_dir, fname)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {path}')


if __name__ == '__main__':
    main()
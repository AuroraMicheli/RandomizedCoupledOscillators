"""
Pareto curve sweep: RON vs s-RON across N_hid values.
s-RON uses connectivity_lif2hrf=0.2 for all datasets.

This is a separate sweep from pareto_sweep.py (which used connectivity=1.0
for FordA and Adiac). Results are saved to a different directory so the
two sweeps can be compared side by side.

Usage:
    python pareto_sweep_lif02.py
    python pareto_sweep_lif02.py --datasets sMNIST FordA
    python pareto_sweep_lif02.py --models sron
    python pareto_sweep_lif02.py --cpu

Results saved to: pareto_results_lif02/pareto_summary.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from utils import get_mnist_data, get_FordA_data, get_Adiac_data, coESN
from utils_aurora import (
    spiking_coESN_rescaled_II,
    estimate_snn_energy_sparse,
    estimate_ann_energy,
)
from esn import spectral_norm_scaling


# =============================================================================
# Hyperparameter configs
# RON configs unchanged from original sweep.
# s-RON configs: connectivity_lif2hrf=0.2 for ALL datasets.
# =============================================================================

RON_CONFIGS = {
    'sMNIST': dict(
        dt=0.042,
        gamma=2.7,        gamma_range=2.0,
        epsilon=0.51,     epsilon_range=1.0,
        inp_scaling=1.0,
        rho=9.0,
        seq_length=784,
        n_inp=1,
        n_out=10,
    ),
    'FordA': dict(
        dt=0.2,
        gamma=1.0,        gamma_range=1.0,
        epsilon=5.0,      epsilon_range=10.0,
        inp_scaling=0.1,
        rho=0.9,
        seq_length=500,
        n_inp=1,
        n_out=2,
    ),
    'Adiac': dict(
        dt=0.01,
        gamma=3.0,        gamma_range=2.0,
        epsilon=5.0,      epsilon_range=10.0,
        inp_scaling=10.0,
        rho=9.0,
        seq_length=176,
        n_inp=1,
        n_out=37,
    ),
}

SRON_CONFIGS = {
    'sMNIST': dict(
        dt=0.042,
        gamma=2.7,        gamma_range=2.0,
        epsilon=0.08,     epsilon_range=1.0,
        inp_scaling=2.0,
        rho=0.99,
        theta_lif=0.05,
        theta_rf=0.005,
        tau_filter=20.0,
        connectivity_lif2hrf=0.2,   # same as original
        connectivity_hrf2lif=1.0,
        readout_mode='final',
        readout_C=1.0,
        seq_length=784,
        n_inp=1,
        n_out=10,
    ),
    'FordA': dict(
        dt=0.051,
        gamma=7.0124,     gamma_range=3.01,
        epsilon=0.1528,   epsilon_range=0.419,
        inp_scaling=0.6247,
        rho=0.75,
        theta_lif=0.0824,
        theta_rf=0.0010,
        tau_filter=6.1,
        connectivity_lif2hrf=0.2,   # changed from 1.0
        connectivity_hrf2lif=1.0,
        readout_mode='final',
        readout_C=0.1,
        seq_length=500,
        n_inp=1,
        n_out=2,
    ),
    'Adiac': dict(
        dt=0.2213,
        gamma=1.3770,     gamma_range=3.5954,
        epsilon=0.01985,  epsilon_range=0.2027,
        inp_scaling=13.0135,
        rho=0.8131,
        theta_lif=0.02290,
        theta_rf=0.01521,
        tau_filter=13.5399,
        connectivity_lif2hrf=0.2,   # changed from 1.0
        connectivity_hrf2lif=1.0,
        readout_mode='final',
        readout_C=0.1,
        seq_length=176,
        n_inp=1,
        n_out=37,
    ),
}

N_HID_VALUES = [100, 200, 400, 800, 1200, 2000, 3000]
N_TRIALS     = 4
SEED_BASE    = 42


# =============================================================================
# Helpers (identical to original sweep)
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_loaders(dataset, batch=120, bs_test=100):
    if dataset == 'sMNIST':
        train_loader, _, test_loader = get_mnist_data(batch, bs_test)
    elif dataset == 'FordA':
        train_loader, _, test_loader = get_FordA_data(
            batch, bs_test, whole_train=True)
    elif dataset == 'Adiac':
        train_loader, _, test_loader = get_Adiac_data(
            batch, bs_test, whole_train=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_loader, test_loader


def extract_ron_features(loader, model, device, dataset):
    model.eval()
    activations, ys = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, ncols=70, desc="  RON features", leave=False):
            x = x.to(device)
            if dataset == 'sMNIST':
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
            out = model(x)[-1][0]
            activations.append(out.cpu())
            ys.append(y)
    return torch.cat(activations).numpy(), torch.cat(ys).numpy()


def extract_sron_features(loader, model, device, dataset):
    model.eval()
    feats, ys = [], []
    r_hrf_list, r_lif_list = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, ncols=70, desc="  s-RON features", leave=False):
            x = x.to(device)
            if dataset == 'sMNIST':
                x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
            features, r = model(x)
            feats.append(features.cpu())
            r_hrf_list.append(r['r_hrf'])
            r_lif_list.append(r['r_lif'])
            ys.append(y)
    feats = torch.cat(feats).numpy()
    ys    = torch.cat(ys).numpy()
    r_hrf = torch.stack(r_hrf_list).mean().item()
    r_lif = torch.stack(r_lif_list).mean().item()
    return feats, ys, r_hrf, r_lif


def run_ron_trial(dataset, n_hid, cfg, train_loader, test_loader, device):
    gamma   = (cfg['gamma']   - cfg['gamma_range']   / 2.,
               cfg['gamma']   + cfg['gamma_range']   / 2.)
    epsilon = (cfg['epsilon'] - cfg['epsilon_range'] / 2.,
               cfg['epsilon'] + cfg['epsilon_range'] / 2.)

    model = coESN(cfg['n_inp'], n_hid, cfg['dt'], gamma, epsilon,
                  cfg['rho'], cfg['inp_scaling'], device=device).to(device)

    train_acts, train_ys = extract_ron_features(train_loader, model, device, dataset)
    scaler     = preprocessing.StandardScaler().fit(train_acts)
    train_acts = scaler.transform(train_acts)
    clf        = LogisticRegression(max_iter=2000, n_jobs=-1).fit(train_acts, train_ys)
    train_acc  = clf.score(train_acts, train_ys) * 100.0

    test_acts, test_ys = extract_ron_features(test_loader, model, device, dataset)
    test_acts = scaler.transform(test_acts)
    test_acc  = clf.score(test_acts, test_ys) * 100.0

    ann_energy = estimate_ann_energy(n_inp=cfg['n_inp'], n_hid=n_hid,
                                     T=cfg['seq_length'])
    return {
        'train_acc': float(train_acc),
        'test_acc':  float(test_acc),
        'energy_J':  float(ann_energy['Energy_J']),
        'MACs':      float(ann_energy['MACs']),
    }


def run_sron_trial(dataset, n_hid, cfg, train_loader, test_loader, device):
    gamma   = (cfg['gamma']   - cfg['gamma_range']   / 2.,
               cfg['gamma']   + cfg['gamma_range']   / 2.)
    epsilon = (cfg['epsilon'] - cfg['epsilon_range'] / 2.,
               cfg['epsilon'] + cfg['epsilon_range'] / 2.)

    model = spiking_coESN_rescaled_II(
        n_inp=cfg['n_inp'], n_hid=n_hid, dt=cfg['dt'],
        gamma=gamma, epsilon=epsilon, rho=cfg['rho'],
        input_scaling=cfg['inp_scaling'],
        theta_lif=cfg['theta_lif'], theta_rf=cfg['theta_rf'],
        tau_filter=cfg['tau_filter'],
        sparse_lif2hrf=(cfg['connectivity_lif2hrf'] < 1.0),
        connectivity_lif2hrf=cfg['connectivity_lif2hrf'],
        sparse_hrf2lif=(cfg['connectivity_hrf2lif'] < 1.0),
        connectivity_hrf2lif=cfg['connectivity_hrf2lif'],
        device=device,
        readout_mode=cfg['readout_mode'],
    ).to(device)

    train_feats, train_ys, r_hrf, r_lif = extract_sron_features(
        train_loader, model, device, dataset)
    scaler      = preprocessing.StandardScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    clf         = LogisticRegression(max_iter=2000, n_jobs=-1,
                                     C=cfg['readout_C']).fit(train_feats, train_ys)
    train_acc   = clf.score(train_feats, train_ys) * 100.0

    test_feats, test_ys, _, _ = extract_sron_features(
        test_loader, model, device, dataset)
    test_feats = scaler.transform(test_feats)
    test_acc   = clf.score(test_feats, test_ys) * 100.0

    snn_energy = estimate_snn_energy_sparse(
        r_hrf=r_hrf, r_lif=r_lif, n_hid=n_hid, T=cfg['seq_length'],
        lif2hrf_connections=model.n_lif2hrf_connections, include_lif=True)
    return {
        'train_acc':            float(train_acc),
        'test_acc':             float(test_acc),
        'energy_J':             float(snn_energy['Energy_J']),
        'SOPs':                 float(snn_energy['SOPs']),
        'r_hrf':                float(r_hrf),
        'r_lif':                float(r_lif),
        'connectivity_lif2hrf': float(cfg['connectivity_lif2hrf']),
    }


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(datasets, models, n_hid_values, n_trials, results_dir, device):
    all_results = {}
    os.makedirs(results_dir, exist_ok=True)

    for dataset in datasets:
        print(f"\n{'='*70}\nDATASET: {dataset}\n{'='*70}")
        all_results[dataset] = {}

        bs_test = 100 if dataset == 'sMNIST' else 120 if dataset == 'FordA' else 30
        train_loader, test_loader = get_loaders(dataset, batch=120, bs_test=bs_test)

        if dataset == 'FordA':
            sample_x, _ = next(iter(train_loader))
            seq_length   = sample_x.shape[1]
            RON_CONFIGS['FordA']['seq_length']  = seq_length
            SRON_CONFIGS['FordA']['seq_length'] = seq_length
            print(f"  FordA seq_length detected: {seq_length}")

        for model_name in models:
            print(f"\n  Model: {model_name}")
            all_results[dataset][model_name] = {}
            cfg = RON_CONFIGS[dataset] if model_name == 'ron' else SRON_CONFIGS[dataset]

            for n_hid in n_hid_values:
                print(f"\n  --- n_hid={n_hid} ---")
                trial_results = []

                for trial in range(n_trials):
                    print(f"    Trial {trial+1}/{n_trials}", end='  ', flush=True)
                    set_seed(SEED_BASE + trial)
                    t0 = time.time()
                    try:
                        if model_name == 'ron':
                            result = run_ron_trial(dataset, n_hid, cfg,
                                                   train_loader, test_loader, device)
                        else:
                            result = run_sron_trial(dataset, n_hid, cfg,
                                                    train_loader, test_loader, device)
                        result['trial'] = trial
                        elapsed = time.time() - t0
                        print(f"test={result['test_acc']:.2f}%  "
                              f"energy={result['energy_J']:.3e} J  ({elapsed:.0f}s)")
                        trial_results.append(result)
                    except Exception as e:
                        print(f"FAILED: {e}")
                        trial_results.append({'error': str(e), 'trial': trial})

                valid = [r for r in trial_results if 'error' not in r]
                if valid:
                    agg = {
                        'n_hid':          n_hid,
                        'n_trials_ok':    len(valid),
                        'test_acc_mean':  float(np.mean([r['test_acc']  for r in valid])),
                        'test_acc_std':   float(np.std( [r['test_acc']  for r in valid])),
                        'train_acc_mean': float(np.mean([r['train_acc'] for r in valid])),
                        'train_acc_std':  float(np.std( [r['train_acc'] for r in valid])),
                        'energy_J_mean':  float(np.mean([r['energy_J']  for r in valid])),
                        'energy_J_std':   float(np.std( [r['energy_J']  for r in valid])),
                        'trial_results':  trial_results,
                    }
                    print(f"    → mean: {agg['test_acc_mean']:.2f}±{agg['test_acc_std']:.2f}%  "
                          f"energy: {agg['energy_J_mean']:.3e}±{agg['energy_J_std']:.3e} J")
                else:
                    agg = {'n_hid': n_hid, 'n_trials_ok': 0,
                           'error': 'all trials failed', 'trial_results': trial_results}

                all_results[dataset][model_name][str(n_hid)] = agg

                # Filename includes lif02 tag so it never clashes with the original sweep
                fname = f"pareto_{dataset}_{model_name}_lif02_nhid{n_hid}.json"
                with open(os.path.join(results_dir, fname), 'w') as f:
                    json.dump(agg, f, indent=2)

    summary_path = os.path.join(results_dir, 'pareto_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}\nAll results saved to: {summary_path}\n{'='*70}")
    return all_results


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pareto sweep: RON vs s-RON (connectivity_lif2hrf=0.2)'
    )
    parser.add_argument('--datasets', nargs='+',
                        default=['sMNIST', 'FordA', 'Adiac'],
                        choices=['sMNIST', 'FordA', 'Adiac'])
    parser.add_argument('--models', nargs='+',
                        default=['ron', 'sron'],
                        choices=['ron', 'sron'])
    parser.add_argument('--n_hid_values', nargs='+', type=int,
                        default=N_HID_VALUES)
    parser.add_argument('--n_trials', type=int, default=N_TRIALS)
    parser.add_argument('--results_dir', type=str, default='pareto_results_lif02')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    print('=' * 70)
    print('PARETO SWEEP: RON vs s-RON  [connectivity_lif2hrf=0.2 for all datasets]')
    print('=' * 70)
    print(f'Datasets:      {args.datasets}')
    print(f'Models:        {args.models}')
    print(f'N_hid values:  {args.n_hid_values}')
    print(f'Trials:        {args.n_trials}')
    print(f'Device:        {device}')
    print(f'Results dir:   {args.results_dir}')
    print('=' * 70)

    run_sweep(
        datasets=args.datasets,
        models=args.models,
        n_hid_values=args.n_hid_values,
        n_trials=args.n_trials,
        results_dir=args.results_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
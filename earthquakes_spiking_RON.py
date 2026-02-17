import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import random
import json
import os
from utils_aurora import *

from esn import spectral_norm_scaling
from ucr_data_utils import get_Earthquakes_data


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(loader, model, device):
    """Extract reservoir features from data"""
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, ncols=80, desc="Extracting features"):
            x = x.to(device)
            features, r = model(x)
            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(y)
    
    if len(feats) == 0:
        return None, None, 0.0, 0.0, 0.0
    
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    return (
        feats,
        labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item()
    )


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on Earthquakes Dataset')
    
    # Model architecture
    parser.add_argument('--n_hid', type=int, default=256,
                       help='Number of hidden units (reservoir size)')
    parser.add_argument('--batch', type=int, default=120,
                       help='Batch size for training')
    
    # Oscillator parameters (same starting point as FordA)
    parser.add_argument('--dt', type=float, default=0.2,
                       help='Time step size')
    parser.add_argument('--gamma', type=float, default=1.88,
                       help='Gamma parameter (damping)')
    parser.add_argument('--epsilon', type=float, default=0.022,
                       help='Epsilon parameter (stiffness)')
    parser.add_argument('--gamma_range', type=float, default=2.64,
                       help='Range for gamma heterogeneity')
    parser.add_argument('--epsilon_range', type=float, default=0.068,
                       help='Range for epsilon heterogeneity')
    
    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=1.76,
                       help='Input scaling factor')
    parser.add_argument('--rho', type=float, default=0.95,
                       help='Spectral radius')
    
    # LIF/HRF parameters
    parser.add_argument('--theta_lif', type=float, default=0.05,
                       help='LIF threshold')
    parser.add_argument('--theta_rf', type=float, default=0.005,
                       help='HRF threshold')
    parser.add_argument('--tau_filter', type=float, default=20.0,
                       help='Filter time constant')
    
    # Sparse connectivity options
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                       help="Fraction of LIF→HRF connections (0-1), 1.0 = dense")
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0,
                       help="Fraction of HRF→LIF recurrent connections (0-1), 1.0 = dense")
    
    # Training options
    parser.add_argument('--cpu', action="store_true",
                       help="Force CPU usage")
    parser.add_argument('--use_test', action="store_true",
                       help="Evaluate on test set")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--test_trials', type=int, default=5,
                       help='Number of trials to compute mean and std on test')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory for UCR datasets')
    
    # Results
    parser.add_argument('--results_dir', type=str, default='results_earthquakes',
                       help="Directory to save results")
    
    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON EARTHQUAKES DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print(f"✅ Using device: {device}")
    
    n_inp = 1   # Earthquakes is univariate
    n_out = 2   # Binary classification
    bs_test = 120
    
    gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    print("\n=== Loading Earthquakes Dataset ===")
    train_loader, valid_loader, test_loader = get_Earthquakes_data(
        args.batch, bs_test, data_dir=args.data_dir, whole_train=True
    )
    print(f"✅ Loaded Earthquakes dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")

    sample_x, _ = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    print(f"   Sequence length: {seq_length}")

    # Store results across trials
    all_test_accs, all_train_accs, all_energies = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial + 1}/{args.test_trials}\n{'='*70}")

        print("\n=== Building Spiking RON ===")
        model = spiking_coESN_rescaled_II(
            n_inp=n_inp, n_hid=args.n_hid, dt=args.dt,
            gamma=gamma, epsilon=epsilon, rho=args.rho,
            input_scaling=args.inp_scaling, theta_lif=args.theta_lif,
            theta_rf=args.theta_rf, tau_filter=args.tau_filter,
            sparse_lif2hrf=use_sparse_lif2hrf,
            connectivity_lif2hrf=args.connectivity_lif2hrf,
            sparse_hrf2lif=use_sparse_hrf2lif,
            connectivity_hrf2lif=args.connectivity_hrf2lif,
            device=device
        ).to(device)

        print(f"✅ Model created")
        print(f"   LIF→HRF: {'SPARSE' if use_sparse_lif2hrf else 'DENSE'} ({args.connectivity_lif2hrf*100:.1f}%)")
        print(f"   HRF→LIF: {'SPARSE' if use_sparse_hrf2lif else 'DENSE'} ({args.connectivity_hrf2lif*100:.1f}%)")

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = extract_features(
            train_loader, model, device)
        print(f"✅ Training features: {train_feats.shape}")
        
        if args.use_test:
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test = extract_features(
                test_loader, model, device)
            print(f"✅ Test features: {test_feats.shape}")
        else:
            test_feats, test_labels = train_feats, train_labels
            r_tot_test, r_hrf_test, r_lif_test = r_tot_train, r_hrf_train, r_lif_train

        scaler = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)

        print("\n=== Training Logistic Regression Readout ===")
        clf = LogisticRegression(max_iter=1000, verbose=0, n_jobs=-1).fit(train_feats, train_labels)
        
        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc = clf.score(test_feats, test_labels) * 100
        
        print(f"✅ Train accuracy: {train_acc:.2f}%")
        print(f"✅ Test accuracy: {test_acc:.2f}%")
        print(f"   r_hrf={r_hrf_train:.4f}, r_lif={r_lif_train:.4f}")

        T = seq_length
        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_hrf_train, r_lif=r_lif_train, n_hid=args.n_hid, T=T,
            lif2hrf_connections=model.n_lif2hrf_connections, include_lif=True
        )
        print(f"   Energy: {snn_energy['Energy_J']:.3e} J")

        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    # Aggregate
    mean_test_acc = np.mean(all_test_accs)
    std_test_acc = np.std(all_test_accs)
    mean_train_acc = np.mean(all_train_accs)
    std_train_acc = np.std(all_train_accs)
    mean_energy = np.mean(all_energies)
    std_energy = np.std(all_energies)

    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: Earthquakes")
    print(f"Hidden units: {args.n_hid}, Trials: {args.test_trials}")
    print(f"Train accuracy:  {mean_train_acc:.2f}% ± {std_train_acc:.2f}%")
    print(f"Test accuracy:   {mean_test_acc:.2f}% ± {std_test_acc:.2f}%")
    print(f"Per-trial test:  {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy: {mean_energy:.3e} ± {std_energy:.3e} J")
    print(f"{'='*70}")

    results = {
        'dataset': 'Earthquakes',
        'args': vars(args),
        'n_trials': args.test_trials,
        'train_acc_mean': float(mean_train_acc),
        'train_acc_std': float(std_train_acc),
        'test_acc_mean': float(mean_test_acc),
        'test_acc_std': float(std_test_acc),
        'test_accs_all': [float(x) for x in all_test_accs],
        'r_hrf_mean': float(np.mean(all_r_hrf)),
        'r_lif_mean': float(np.mean(all_r_lif)),
        'r_tot_mean': float(np.mean(all_r_total)),
        'energy_J_mean': float(mean_energy),
        'energy_J_std': float(std_energy),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid': int(args.n_hid),
        'base_seed': int(args.seed),
        'sequence_length': int(seq_length)
    }

    os.makedirs(args.results_dir, exist_ok=True)
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = f"results_earthquakes_nhid{args.n_hid}_{conn_lif_str}_{conn_hrf_str}_trials{args.test_trials}_seed{args.seed}.json"
    results_path = os.path.join(args.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
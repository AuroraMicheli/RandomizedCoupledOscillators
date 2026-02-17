import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from utils_aurora import *

from esn import spectral_norm_scaling
from utils import get_mnist_data

import matplotlib.pyplot as plt
import random
import json
import os





########## MAIN with Rescaled Model and Time-Pooled Features ##########

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def estimate_snn_energy_double_sparse(
    r_hrf,
    r_lif,
    n_hid,
    T,
    lif2hrf_connections,
    hrf2lif_connections,
    include_lif=True,
    E_SOP=0.9e-12
):
    """
    Energy estimator for double-sparse connectivity (LIF→HRF AND HRF→LIF both sparse)
    """
    
    # --- HRF spikes (with sparse HRF→LIF recurrence) ---
    hrf_spikes = r_hrf * n_hid * T
    
    # Average fanout per HRF neuron (sparse recurrence)
    hrf_fanout = hrf2lif_connections / n_hid
    hrf_sops = hrf_spikes * hrf_fanout
    
    total_sops = hrf_sops

    # --- LIF spikes (with sparse LIF→HRF) ---
    if include_lif:
        lif_spikes = r_lif * n_hid * T
        
        # Average fanout per LIF neuron
        lif_fanout = lif2hrf_connections / n_hid
        lif_sops = lif_spikes * lif_fanout
        total_sops += lif_sops
    else:
        lif_sops = 0.0

    energy = total_sops * E_SOP

    return {
        "SOPs": total_sops,
        "Energy_J": energy,
        "HRF_SOPs": hrf_sops,
        "LIF_SOPs": lif_sops,
        "HRF_spikes": hrf_spikes,
        "LIF_spikes": lif_spikes if include_lif else 0.0,
        "HRF_fanout": hrf_fanout,
        "LIF_fanout": lif_fanout if include_lif else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Spiking coESN with Double-Sparse Connectivity — Permuted sMNIST')
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--batch', type=int, default=256)  
    parser.add_argument('--dt', type=float, default=0.047)
    parser.add_argument('--gamma', type=float, default=2.62)
    parser.add_argument('--epsilon', type=float, default=0.24)
    parser.add_argument('--gamma_range', type=float, default=3.84)
    parser.add_argument('--epsilon_range', type=float, default=1.86)

    parser.add_argument('--inp_scaling', type=float, default=3.67)   
    parser.add_argument('--theta_lif', type=float, default=0.05)
    parser.add_argument('--theta_rf', type=float, default=0.005)
    parser.add_argument('--tau_filter', type=float, default=20.0)

    parser.add_argument('--rho', type=float, default=1.55)
    parser.add_argument('--cpu', action="store_true")
    parser.add_argument('--use_test', action="store_true")

    # LIF→HRF sparse connectivity options
    parser.add_argument('--sparse_lif2hrf', action="store_true", 
                        help="Use sparse LIF→HRF connectivity")
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                        help="Fraction of LIF→HRF connections (0-1), 1.0 = dense")
    
    # HRF→LIF sparse recurrent connectivity options
    parser.add_argument('--sparse_hrf2lif', action="store_true",
                        help="Use sparse HRF→LIF recurrent connectivity")
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0,
                        help="Fraction of HRF→LIF recurrent connections (0-1), 1.0 = dense")

    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Permutation seed (separate from model seed so permutation is consistent across runs)
    parser.add_argument('--perm_seed', type=int, default=0,
                        help="Random seed for generating the fixed permutation (default: 0)")

    # Results directory
    parser.add_argument('--results_dir', type=str, default='results_psMNIST',
                        help="Directory to save results")

    args = parser.parse_args()

    print("=" * 70)
    print("DOUBLE SPARSE CONNECTIVITY EXPERIMENT — PERMUTED sMNIST (psMNIST)")
    print("=" * 70)
    print(args)
    print("=" * 70)

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"\n✅ Set random seed to {args.seed}")

    # --- setup ---
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print(f"✅ Using device: {device}")

    # --- Generate fixed permutation ---
    # Use a separate seed so the permutation is the same regardless of model seed
    perm_rng = torch.Generator()
    perm_rng.manual_seed(args.perm_seed)
    perm = torch.randperm(784, generator=perm_rng).to(device)
    print(f"✅ Generated fixed permutation with perm_seed={args.perm_seed}")
    
    n_inp = 1
    n_out = 10
    bs_test = 100
    gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

    # Determine if using sparse connectivity
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    # --- model ---
    model = spiking_coESN_rescaled_II(
        n_inp=n_inp,
        n_hid=args.n_hid,
        dt=args.dt,
        gamma=gamma,
        epsilon=epsilon,
        rho=args.rho,
        input_scaling=args.inp_scaling,
        theta_lif=args.theta_lif,
        theta_rf=args.theta_rf,
        tau_filter=args.tau_filter,
        sparse_lif2hrf=use_sparse_lif2hrf,
        connectivity_lif2hrf=args.connectivity_lif2hrf,
        connectivity_hrf2lif=args.connectivity_hrf2lif,
        sparse_hrf2lif=use_sparse_hrf2lif,
        device=device
    ).to(device)

    train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

    print("\n=== Connectivity Summary ===")
    print(f"LIF→HRF: {'SPARSE' if use_sparse_lif2hrf else 'DENSE'} ({args.connectivity_lif2hrf*100:.1f}%)")
    print(f"HRF→LIF: {'SPARSE' if use_sparse_hrf2lif else 'DENSE'} ({args.connectivity_hrf2lif*100:.1f}%)")
    print(f"Feature dimensionality: {3 * args.n_hid}")

    def extract_features(loader, model, device, perm):
        """Extract features with permuted pixel ordering (psMNIST)."""
        model.eval()
        feats, labels_all = [], []
        r_tot, r_hrf, r_lif = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, ncols=80, desc="Extracting features"):
                images = images.to(device)
                # Reshape MNIST image to a sequence of 784 1D inputs
                images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
                # Apply fixed permutation to the sequence order
                images = images[:, perm, :]
                
                features, r = model(images)
                feats.append(features.cpu())
                r_tot.append(r["r_total"])
                r_hrf.append(r["r_hrf"])
                r_lif.append(r["r_lif"])
                labels_all.append(labels)
        feats = torch.cat(feats, dim=0).numpy()
        labels_all = torch.cat(labels_all, dim=0).numpy()

        return (
            feats,
            labels_all,
            torch.stack(r_tot).mean().item(),
            torch.stack(r_hrf).mean().item(),
            torch.stack(r_lif).mean().item()
        )

    print("\n=== Extracting features (permuted sMNIST) ===")
    train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = extract_features(train_loader, model, device, perm)
    print(f"✅ Extracted training feature vector size: {train_feats.shape[1]}")

    valid_feats, valid_labels, r_tot_valid, r_hrf_valid, r_lif_valid = extract_features(valid_loader, model, device, perm)
    
    if args.use_test:
        test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test = extract_features(test_loader, model, device, perm)
    else:
        test_feats, test_labels = valid_feats, valid_labels
        r_tot_test, r_hrf_test, r_lif_test = r_tot_valid, r_hrf_valid, r_lif_valid

    # --- standardize ---
    scaler = preprocessing.StandardScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    valid_feats = scaler.transform(valid_feats)
    test_feats = scaler.transform(test_feats)

    # --- logistic regression readout ---
    print("\n=== Training logistic regression readout ===")
    clf = LogisticRegression(max_iter=1000, verbose=0, n_jobs=-1).fit(train_feats, train_labels)

    train_acc = clf.score(train_feats, train_labels) * 100
    valid_acc = clf.score(valid_feats, valid_labels) * 100
    test_acc = clf.score(test_feats, test_labels) * 100
    
    print(f"✅ Training accuracy: {train_acc:.2f}%")
    print(f"✅ Validation accuracy: {valid_acc:.2f}%")
    print(f"✅ Test accuracy: {test_acc:.2f}%")

    print(f"\n=== Firing Rate Statistics ===")
    print(f"Average firing rate r_hrf (train): {r_hrf_train:.4f}")
    print(f"Average firing rate r_lif (train): {r_lif_train:.4f}")
    print(f"Average firing rate r_total (train): {r_tot_train:.4f}")

    # ===== Energy with Double-Sparse Connectivity =====
    T = 784  # psMNIST timesteps (same length, permuted order)

    snn_energy = estimate_snn_energy_double_sparse(
        r_hrf=r_hrf_train,
        r_lif=r_lif_train,
        n_hid=args.n_hid,
        T=T,
        lif2hrf_connections=model.n_lif2hrf_connections,
        hrf2lif_connections=model.n_hrf2lif_connections,
        include_lif=True
    )

    print("\n=== Theoretical SNN Energy ===")
    print(f"Total SOPs: {snn_energy['SOPs']:.3e}")
    print(f"  HRF→LIF SOPs: {snn_energy['HRF_SOPs']:.3e} (fanout: {snn_energy['HRF_fanout']:.1f})")
    print(f"  LIF→HRF SOPs: {snn_energy['LIF_SOPs']:.3e} (fanout: {snn_energy['LIF_fanout']:.1f})")
    print(f"Energy (J): {snn_energy['Energy_J']:.3e}")

    # ===== Save Results =====
    results = {
        'dataset': 'psMNIST',
        'args': vars(args),
        'train_acc': float(train_acc),
        'valid_acc': float(valid_acc),
        'test_acc': float(test_acc),
        'r_hrf_train': float(r_hrf_train),
        'r_lif_train': float(r_lif_train),
        'r_tot_train': float(r_tot_train),
        'energy_J': float(snn_energy['Energy_J']),
        'SOPs': float(snn_energy['SOPs']),
        'HRF_SOPs': float(snn_energy['HRF_SOPs']),
        'LIF_SOPs': float(snn_energy['LIF_SOPs']),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid': int(args.n_hid),
        'seed': int(args.seed),
        'perm_seed': int(args.perm_seed)
    }

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save with descriptive filename
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = f"psMNIST_results_nhid{args.n_hid}_{conn_lif_str}_{conn_hrf_str}_seed{args.seed}.json"
    results_path = os.path.join(args.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")

    print("\n=== Final Results Summary ===")
    print(f"Model: Double-Sparse Spiking RON")
    print(f"Dataset: psMNIST (permuted sequential MNIST, perm_seed={args.perm_seed})")
    print(f"Hidden units: {args.n_hid}")
    print(f"LIF→HRF connectivity: {args.connectivity_lif2hrf*100:.0f}% ({model.n_lif2hrf_connections} connections)")
    print(f"HRF→LIF connectivity: {args.connectivity_hrf2lif*100:.0f}% ({model.n_hrf2lif_connections} connections)")
    print(f"Seed: {args.seed}")
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Validation accuracy: {valid_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"HRF firing rate: {r_hrf_train:.4f}")
    print(f"Energy efficiency: {snn_energy['Energy_J']:.3e} J")
    print("=" * 70)


if __name__ == "__main__":
    main()
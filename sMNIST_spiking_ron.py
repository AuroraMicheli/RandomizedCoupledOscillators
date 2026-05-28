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
from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse

from esn import spectral_norm_scaling
from utils import get_mnist_data

import random
import json
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, ncols=80, desc="Extracting features"):
            images = images.to(device)
            # Reshape MNIST image (B, 1, 28, 28) -> sequential (B, 784, 1)
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            features, r = model(images)
            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(labels)

    if len(feats) == 0:
        return None, None, 0.0, 0.0, 0.0

    feats      = torch.cat(feats,      dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return (
        feats, labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item()
    )


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on Sequential MNIST')

    # Model architecture
    parser.add_argument('--n_hid',  type=int, default=800)
    parser.add_argument('--batch',  type=int, default=256)

    # Oscillator parameters
    parser.add_argument('--dt',            type=float, default=0.042)
    parser.add_argument('--gamma',         type=float, default=2.7)
    parser.add_argument('--epsilon',       type=float, default=0.08)
    parser.add_argument('--gamma_range',   type=float, default=2.0)
    parser.add_argument('--epsilon_range', type=float, default=1.0)

    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=2.0)
    parser.add_argument('--rho',         type=float, default=0.99)

    # LIF/HRF parameters
    parser.add_argument('--theta_lif',  type=float, default=0.05)
    parser.add_argument('--theta_rf',   type=float, default=0.005)
    parser.add_argument('--tau_filter', type=float, default=20.0)

    # Sparse connectivity
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                        help="Fraction of LIF->HRF connections (0-1), 1.0 = dense")
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0,
                        help="Fraction of HRF->LIF recurrent connections (0-1), 1.0 = dense")

    # Training options
    parser.add_argument('--cpu',         action="store_true")
    parser.add_argument('--use_test',    action="store_true")
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--test_trials', type=int, default=5)
    parser.add_argument('--data_dir',    type=str, default='data')

    # Readout
    parser.add_argument('--readout_C', type=float, default=1.0,
                        help="Inverse regularization for logistic regression")
    #parser.add_argument('--readout_mode', type=str, default='final',
                        #choices=['final', 'mean', 'rms_std_final'],
                        #help="Reservoir readout strategy: "
                             #"'final' (last hy, n_hid features), "
                             #"'mean' (temporal mean, n_hid features), "
                             #"'rms_std_final' (RMS+Std+Final, 3*n_hid features)")
    parser.add_argument('--readout_mode', type=str, default='final',
                    choices=['final', 'mean', 'rms', 'std', 'rms_std_final', 'spikes_mean'])

    # Results
    parser.add_argument('--results_dir', type=str, default='results_smnist')

    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON SEQUENTIAL MNIST (sMNIST)")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = (torch.device("cuda") if torch.cuda.is_available() and not args.cpu
              else torch.device("cpu"))
    print(f"Using device: {device}")

    n_inp      = 1    # sMNIST is pixel-by-pixel (univariate)
    n_out      = 10   # 10 digit classes
    bs_test    = 100
    seq_length = 784  # 28*28 pixels

    gamma   = (args.gamma   - args.gamma_range   / 2., args.gamma   + args.gamma_range   / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    # ── Trainable parameter count (readout only; reservoir is fixed) ──────────
    # Convention matches RON paper Table 2: n_features * n_out, bias excluded
    n_features         = args.n_hid * 3 if args.readout_mode == 'rms_std_final' else args.n_hid
    n_trainable_params = n_features * n_out

    print(f"\n=== Readout Parameter Count ===")
    print(f"   Readout mode:      {args.readout_mode}")
    print(f"   Feature dimension: {n_features}  "
          f"({'3 x ' if args.readout_mode == 'rms_std_final' else ''}{args.n_hid} hidden units)")
    print(f"   Output classes:    {n_out}")
    print(f"   Trainable params:  {n_trainable_params:,}  [= {n_features} x {n_out}]")

    print("\n=== Loading Sequential MNIST Dataset ===")
    train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)
    print(f"Loaded sMNIST dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples:     {len(test_loader.dataset)}")
    print(f"   Sequence length:  {seq_length} (28x28 pixels, pixel-by-pixel)")

    all_test_accs, all_train_accs, all_energies = [], [], []
    all_sops, all_sops_hrf, all_sops_lif = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial + 1}/{args.test_trials}\n{'='*70}")

        print("\n=== Building Spiking RON ===")
        model = spiking_coESN_rescaled_II(
            n_inp=n_inp, n_hid=args.n_hid, dt=args.dt,
            gamma=gamma, epsilon=epsilon, rho=args.rho,
            input_scaling=args.inp_scaling,
            theta_lif=args.theta_lif, theta_rf=args.theta_rf,
            tau_filter=args.tau_filter,
            sparse_lif2hrf=use_sparse_lif2hrf,
            connectivity_lif2hrf=args.connectivity_lif2hrf,
            sparse_hrf2lif=use_sparse_hrf2lif,
            connectivity_hrf2lif=args.connectivity_hrf2lif,
            device=device,
            readout_mode=args.readout_mode,
        ).to(device)

        print(f"Model created -- readout_mode='{args.readout_mode}'")
        print(f"   n_hid:   {args.n_hid}")
        print(f"   LIF->HRF: {'SPARSE' if use_sparse_lif2hrf else 'DENSE'} ({args.connectivity_lif2hrf*100:.1f}%)")
        print(f"   HRF->LIF: {'SPARSE' if use_sparse_hrf2lif else 'DENSE'} ({args.connectivity_hrf2lif*100:.1f}%)")
        print(f"   Trainable readout params: {n_trainable_params:,}")

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = extract_features(
            train_loader, model, device)
        print(f"Training features: {train_feats.shape}")

        if args.use_test:
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test = extract_features(
                test_loader, model, device)
            print(f"Test features: {test_feats.shape}")
        else:
            test_feats, test_labels = train_feats, train_labels

        scaler      = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        print("\n=== Training Logistic Regression Readout ===")
        clf = LogisticRegression(
            max_iter=2000, verbose=0, n_jobs=2, C=args.readout_C
        ).fit(train_feats, train_labels)

        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc  = clf.score(test_feats,  test_labels)  * 100

        print(f"Train accuracy: {train_acc:.2f}%")
        print(f"Test accuracy:  {test_acc:.2f}%")
        print(f"r_hrf={r_hrf_train:.4f}, r_lif={r_lif_train:.4f}")

        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_hrf_train, r_lif=r_lif_train,
            n_hid=args.n_hid, T=seq_length,
            lif2hrf_connections=model.n_lif2hrf_connections,
            include_lif=True
        )
        print(f"Energy: {snn_energy['Energy_J']:.3e} J  |  "
              f"SOPs: {snn_energy['SOPs']:.3e}  "
              f"(HRF: {snn_energy['HRF_SOPs']:.3e}, LIF: {snn_energy['LIF_SOPs']:.3e})")

        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_sops.append(snn_energy['SOPs'])
        all_sops_hrf.append(snn_energy['HRF_SOPs'])
        all_sops_lif.append(snn_energy['LIF_SOPs'])
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    mean_test_acc  = np.mean(all_test_accs)
    std_test_acc   = np.std(all_test_accs)
    mean_train_acc = np.mean(all_train_accs)
    std_train_acc  = np.std(all_train_accs)
    mean_energy    = np.mean(all_energies)
    std_energy     = np.std(all_energies)
    mean_sops      = np.mean(all_sops)
    mean_sops_hrf  = np.mean(all_sops_hrf)
    mean_sops_lif  = np.mean(all_sops_lif)

    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset:          Sequential MNIST (sMNIST)")
    print(f"Hidden units:     {args.n_hid},  Trials: {args.test_trials}")
    print(f"Readout mode:     {args.readout_mode}")
    print(f"Feature dim:      {n_features}")
    print(f"Trainable params: {n_trainable_params:,}  [= {n_features} x {n_out}]")
    print(f"Sequence:         {seq_length} steps (pixel-by-pixel)")
    print(f"Train accuracy:   {mean_train_acc:.2f}% +/- {std_train_acc:.2f}%")
    print(f"Test accuracy:    {mean_test_acc:.2f}%  +/- {std_test_acc:.2f}%")
    print(f"Per-trial test:   {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy:           {mean_energy:.3e} +/- {std_energy:.3e} J")
    print(f"SOPs (total):     {mean_sops:.3e}  "
          f"(HRF: {mean_sops_hrf:.3e}, LIF: {mean_sops_lif:.3e})")
    print(f"{'='*70}")

    results = {
        'dataset': 'sMNIST',
        'args': vars(args),
        'n_trials': args.test_trials,
        'n_inp': n_inp,
        'n_out': n_out,
        'readout_mode': args.readout_mode,
        'readout_C': float(args.readout_C),
        'n_features': int(n_features),
        'n_trainable_params': int(n_trainable_params),
        'train_acc_mean': float(mean_train_acc),
        'train_acc_std':  float(std_train_acc),
        'test_acc_mean':  float(mean_test_acc),
        'test_acc_std':   float(std_test_acc),
        'test_accs_all':  [float(x) for x in all_test_accs],
        'r_hrf_mean':     float(np.mean(all_r_hrf)),
        'r_lif_mean':     float(np.mean(all_r_lif)),
        'r_tot_mean':     float(np.mean(all_r_total)),
        'energy_J_mean':  float(mean_energy),
        'energy_J_std':   float(std_energy),
        'sops_mean':      float(mean_sops),
        'sops_hrf_mean':  float(mean_sops_hrf),
        'sops_lif_mean':  float(mean_sops_lif),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid':           int(args.n_hid),
        'base_seed':       int(args.seed),
        'sequence_length': seq_length,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = (
        f"results_smnist_nhid{args.n_hid}_{conn_lif_str}_{conn_hrf_str}"
        f"_{args.readout_mode}"
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    results_path = os.path.join(args.results_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
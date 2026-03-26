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
from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse

from esn import spectral_norm_scaling
from utils import get_Adiac_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(loader, model, device, save_trajectories=False,
                     n_samples=50, last_steps=None):
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []
    trajectories_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, ncols=80, desc="Extracting features")):
            x = x.to(device)

            if save_trajectories and batch_idx == 0:
                steps_to_save = last_steps if last_steps is not None else x.size(1)
                features, r, trajectories = model(x, return_trajectories=True,
                                                  last_steps=steps_to_save)
                trajectories_list.append(trajectories)
            else:
                features, r = model(x)

            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(y)

    if len(feats) == 0:
        return None, None, 0.0, 0.0, 0.0, None

    feats      = torch.cat(feats,      dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    saved_trajectories = trajectories_list[0] if save_trajectories and trajectories_list else None
    return (
        feats, labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item(),
        saved_trajectories
    )


def visualize_hrf_trajectories(trajectories, n_samples=50,
                               save_path='hrf_trajectories_Adiac.png',
                               title_prefix='Train'):
    hy_traj = trajectories['hy']  # (B, last_steps, n_hid)
    B, T, n_hid = hy_traj.shape

    if n_samples > n_hid:
        n_samples = n_hid
    neuron_indices = np.linspace(0, n_hid - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax     = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_samples))
    for idx, neuron_idx in enumerate(neuron_indices):
        trajectory = hy_traj[0, :, neuron_idx].cpu().numpy()
        ax.plot(trajectory, color=colors[idx], alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('HRF State (hy)', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF Trajectories for {n_samples} Neurons', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T - 1])

    ax           = axes[1]
    heatmap_data = hy_traj[0, :, neuron_indices].cpu().numpy().T  # (n_samples, T)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF State Heatmap ({n_samples} Neurons)', fontsize=14)
    ytick_positions = np.linspace(0, n_samples - 1, min(10, n_samples), dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([neuron_indices[i] for i in ytick_positions])
    plt.colorbar(im, ax=ax, label='HRF State Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"HRF trajectories saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on Adiac Dataset')

    # Model architecture
    parser.add_argument('--n_hid',  type=int, default=800)
    parser.add_argument('--batch',  type=int, default=120)

    # Oscillator parameters — best config from hyperparam search (54.10% test)
    parser.add_argument('--dt',            type=float, default=0.2213)
    parser.add_argument('--gamma',         type=float, default=1.3770)
    parser.add_argument('--epsilon',       type=float, default=0.01985)
    parser.add_argument('--gamma_range',   type=float, default=3.5954)
    parser.add_argument('--epsilon_range', type=float, default=0.2027)

    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=13.0135)
    parser.add_argument('--rho',         type=float, default=0.8131)

    # LIF/HRF parameters
    parser.add_argument('--theta_lif',  type=float, default=0.02290)
    parser.add_argument('--theta_rf',   type=float, default=0.01521)
    parser.add_argument('--tau_filter', type=float, default=13.5399)

    # Sparse connectivity
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                       help="Fraction of LIF->HRF connections (0-1), 1.0 = dense")
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0,
                       help="Fraction of HRF->LIF recurrent connections (0-1), 1.0 = dense")

    # Training options
    parser.add_argument('--cpu',         action="store_true")
    parser.add_argument('--use_test',    action="store_true")
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--test_trials', type=int, default=3)
    parser.add_argument('--data_dir',    type=str, default='data')

    # Readout — best C from search was 0.1
    parser.add_argument('--readout_C', type=float, default=0.1,
                       help="Inverse regularization for logistic regression")
    parser.add_argument('--readout_mode', type=str, default='final',
                       choices=['final', 'mean', 'rms_std_final'],
                       help="Reservoir readout strategy: "
                            "'final' (last hy, n_hid features), "
                            "'mean' (temporal mean, n_hid features), "
                            "'rms_std_final' (RMS+Std+Final, 3*n_hid features)")

    # Visualization
    parser.add_argument('--visualize_trajectories', action="store_true")
    parser.add_argument('--viz_n_samples',  type=int,  default=50)
    parser.add_argument('--viz_last_steps', type=int,  default=None)
    parser.add_argument('--visualize_test', action="store_true")

    # Results
    parser.add_argument('--results_dir', type=str, default='results_adiac')

    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON ADIAC DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = (torch.device("cuda") if torch.cuda.is_available() and not args.cpu
              else torch.device("cpu"))
    print(f"Using device: {device}")

    n_inp   = 1    # Adiac is univariate
    n_out   = 37   # 37 classes
    bs_test = 30

    gamma   = (args.gamma   - args.gamma_range   / 2., args.gamma   + args.gamma_range   / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    print("\n=== Loading Adiac Dataset ===")
    train_loader, valid_loader, test_loader = get_Adiac_data(
        args.batch, bs_test, whole_train=True
    )
    print(f"Loaded Adiac dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples:     {len(test_loader.dataset)}")

    sample_x, _ = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    print(f"   Sequence length: {seq_length}")

    all_test_accs, all_train_accs, all_energies = [], [], []
    all_sops, all_sops_hrf, all_sops_lif = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial + 1}/{args.test_trials}\n{'='*70}")

        set_seed(args.seed + trial)

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
        print(f"   LIF->HRF: {'SPARSE' if use_sparse_lif2hrf else 'DENSE'} ({args.connectivity_lif2hrf*100:.1f}%)")
        print(f"   HRF->LIF: {'SPARSE' if use_sparse_hrf2lif else 'DENSE'} ({args.connectivity_hrf2lif*100:.1f}%)")

        save_traj = args.visualize_trajectories and trial == 0

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train, train_trajectories = \
            extract_features(train_loader, model, device,
                             save_trajectories=save_traj,
                             n_samples=args.viz_n_samples,
                             last_steps=args.viz_last_steps)
        print(f"Training features: {train_feats.shape}")

        if save_traj and train_trajectories is not None:
            print("\n=== Visualizing TRAIN HRF Trajectories ===")
            os.makedirs(args.results_dir, exist_ok=True)
            viz_path = os.path.join(args.results_dir,
                f'hrf_trajectories_TRAIN_nhid{args.n_hid}_trial{trial}.png')
            visualize_hrf_trajectories(
                train_trajectories, n_samples=args.viz_n_samples,
                save_path=viz_path, title_prefix='Adiac Train'
            )

        if args.use_test:
            save_test_traj = args.visualize_test and trial == 0
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, test_trajectories = \
                extract_features(test_loader, model, device,
                                 save_trajectories=save_test_traj,
                                 n_samples=args.viz_n_samples,
                                 last_steps=args.viz_last_steps)
            print(f"Test features: {test_feats.shape}")

            if save_test_traj and test_trajectories is not None:
                print("\n=== Visualizing TEST HRF Trajectories ===")
                viz_path_test = os.path.join(args.results_dir,
                    f'hrf_trajectories_TEST_nhid{args.n_hid}_trial{trial}.png')
                visualize_hrf_trajectories(
                    test_trajectories, n_samples=args.viz_n_samples,
                    save_path=viz_path_test, title_prefix='Adiac Test'
                )
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
    print(f"Dataset:        Adiac")
    print(f"Hidden units:   {args.n_hid}, Trials: {args.test_trials}")
    print(f"Readout mode:   {args.readout_mode}")
    print(f"Train accuracy: {mean_train_acc:.2f}% +/- {std_train_acc:.2f}%")
    print(f"Test accuracy:  {mean_test_acc:.2f}%  +/- {std_test_acc:.2f}%")
    print(f"Per-trial test: {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy:         {mean_energy:.3e} +/- {std_energy:.3e} J")
    print(f"SOPs (total):   {mean_sops:.3e}  "
          f"(HRF: {mean_sops_hrf:.3e}, LIF: {mean_sops_lif:.3e})")
    print(f"{'='*70}")

    results = {
        'dataset': 'Adiac',
        'args': vars(args),
        'n_trials': args.test_trials,
        'n_inp': n_inp,
        'n_out': n_out,
        'readout_mode': args.readout_mode,
        'readout_C': float(args.readout_C),
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
        'sequence_length': int(seq_length),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = (
        f"results_adiac_nhid{args.n_hid}_{conn_lif_str}_{conn_hrf_str}"
        f"_{args.readout_mode}"
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    results_path = os.path.join(args.results_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()


    
'''
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
from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse

from esn import spectral_norm_scaling
from utils import get_Adiac_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(loader, model, device, save_trajectories=False,
                     n_samples=50, last_steps=None):
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []
    trajectories_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, ncols=80, desc="Extracting features")):
            x = x.to(device)

            if save_trajectories and batch_idx == 0:
                steps_to_save = last_steps if last_steps is not None else x.size(1)
                features, r, trajectories = model(x, return_trajectories=True,
                                                  last_steps=steps_to_save)
                trajectories_list.append(trajectories)
            else:
                features, r = model(x)

            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(y)

    if len(feats) == 0:
        return None, None, 0.0, 0.0, 0.0, None

    feats      = torch.cat(feats,      dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    saved_trajectories = trajectories_list[0] if save_trajectories and trajectories_list else None
    return (
        feats, labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item(),
        saved_trajectories
    )


def visualize_hrf_trajectories(trajectories, n_samples=50,
                               save_path='hrf_trajectories_Adiac.png',
                               title_prefix='Train'):
    hy_traj = trajectories['hy']  # (B, last_steps, n_hid)
    B, T, n_hid = hy_traj.shape

    if n_samples > n_hid:
        n_samples = n_hid
    neuron_indices = np.linspace(0, n_hid - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax     = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_samples))
    for idx, neuron_idx in enumerate(neuron_indices):
        trajectory = hy_traj[0, :, neuron_idx].cpu().numpy()
        ax.plot(trajectory, color=colors[idx], alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('HRF State (hy)', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF Trajectories for {n_samples} Neurons', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T - 1])

    ax           = axes[1]
    heatmap_data = hy_traj[0, :, neuron_indices].cpu().numpy().T  # (n_samples, T)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF State Heatmap ({n_samples} Neurons)', fontsize=14)
    ytick_positions = np.linspace(0, n_samples - 1, min(10, n_samples), dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([neuron_indices[i] for i in ytick_positions])
    plt.colorbar(im, ax=ax, label='HRF State Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"HRF trajectories saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on Adiac Dataset')

    # Model architecture
    parser.add_argument('--n_hid',  type=int, default=800)
    parser.add_argument('--batch',  type=int, default=120)

    # Oscillator parameters
    parser.add_argument('--dt',            type=float, default=0.097)
    parser.add_argument('--gamma',         type=float, default=3.077)
    parser.add_argument('--epsilon',       type=float, default=0.02)
    parser.add_argument('--gamma_range',   type=float, default=3.28)
    parser.add_argument('--epsilon_range', type=float, default=0.20)

    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=9.87)
    parser.add_argument('--rho',         type=float, default=0.87)

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
    parser.add_argument('--test_trials', type=int, default=3)
    parser.add_argument('--data_dir',    type=str, default='data')

    # Readout
    parser.add_argument('--readout_C', type=float, default=1.0,
                       help="Inverse regularization for logistic regression")
    parser.add_argument('--readout_mode', type=str, default='final',
                       choices=['final', 'mean', 'rms_std_final'],
                       help="Reservoir readout strategy: "
                            "'final' (last hy, n_hid features), "
                            "'mean' (temporal mean, n_hid features), "
                            "'rms_std_final' (RMS+Std+Final, 3*n_hid features)")

    # Visualization
    parser.add_argument('--visualize_trajectories', action="store_true")
    parser.add_argument('--viz_n_samples',  type=int,  default=50)
    parser.add_argument('--viz_last_steps', type=int,  default=None)
    parser.add_argument('--visualize_test', action="store_true")

    # Results
    parser.add_argument('--results_dir', type=str, default='results_adiac')

    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON ADIAC DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = (torch.device("cuda") if torch.cuda.is_available() and not args.cpu
              else torch.device("cpu"))
    print(f"Using device: {device}")

    n_inp   = 1    # Adiac is univariate
    n_out   = 37   # 37 classes
    bs_test = 30

    gamma   = (args.gamma   - args.gamma_range   / 2., args.gamma   + args.gamma_range   / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    print("\n=== Loading Adiac Dataset ===")
    train_loader, valid_loader, test_loader = get_Adiac_data(
        args.batch, bs_test, whole_train=True
    )
    print(f"Loaded Adiac dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples:     {len(test_loader.dataset)}")

    sample_x, _ = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    print(f"   Sequence length: {seq_length}")

    all_test_accs, all_train_accs, all_energies = [], [], []
    all_sops, all_sops_hrf, all_sops_lif = [], [], []          # SOPs
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
            readout_mode=args.readout_mode,                    # NEW
        ).to(device)

        print(f"Model created -- readout_mode='{args.readout_mode}'")
        print(f"   LIF->HRF: {'SPARSE' if use_sparse_lif2hrf else 'DENSE'} ({args.connectivity_lif2hrf*100:.1f}%)")
        print(f"   HRF->LIF: {'SPARSE' if use_sparse_hrf2lif else 'DENSE'} ({args.connectivity_hrf2lif*100:.1f}%)")

        save_traj = args.visualize_trajectories and trial == 0

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train, train_trajectories = \
            extract_features(train_loader, model, device,
                             save_trajectories=save_traj,
                             n_samples=args.viz_n_samples,
                             last_steps=args.viz_last_steps)
        print(f"Training features: {train_feats.shape}")

        if save_traj and train_trajectories is not None:
            print("\n=== Visualizing TRAIN HRF Trajectories ===")
            os.makedirs(args.results_dir, exist_ok=True)
            viz_path = os.path.join(args.results_dir,
                f'hrf_trajectories_TRAIN_nhid{args.n_hid}_trial{trial}.png')
            visualize_hrf_trajectories(
                train_trajectories, n_samples=args.viz_n_samples,
                save_path=viz_path, title_prefix='Adiac Train'
            )

        if args.use_test:
            save_test_traj = args.visualize_test and trial == 0
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, test_trajectories = \
                extract_features(test_loader, model, device,
                                 save_trajectories=save_test_traj,
                                 n_samples=args.viz_n_samples,
                                 last_steps=args.viz_last_steps)
            print(f"Test features: {test_feats.shape}")

            if save_test_traj and test_trajectories is not None:
                print("\n=== Visualizing TEST HRF Trajectories ===")
                viz_path_test = os.path.join(args.results_dir,
                    f'hrf_trajectories_TEST_nhid{args.n_hid}_trial{trial}.png')
                visualize_hrf_trajectories(
                    test_trajectories, n_samples=args.viz_n_samples,
                    save_path=viz_path_test, title_prefix='Adiac Test'
                )
        else:
            test_feats, test_labels = train_feats, train_labels

        scaler      = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        print("\n=== Training Logistic Regression Readout ===")
        clf = LogisticRegression(
            max_iter=2000, verbose=0, n_jobs=-1, C=args.readout_C
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
        all_sops.append(snn_energy['SOPs'])                    # SOPs
        all_sops_hrf.append(snn_energy['HRF_SOPs'])            # SOPs
        all_sops_lif.append(snn_energy['LIF_SOPs'])            # SOPs
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    mean_test_acc  = np.mean(all_test_accs)
    std_test_acc   = np.std(all_test_accs)
    mean_train_acc = np.mean(all_train_accs)
    std_train_acc  = np.std(all_train_accs)
    mean_energy    = np.mean(all_energies)
    std_energy     = np.std(all_energies)
    mean_sops      = np.mean(all_sops)                         # SOPs
    mean_sops_hrf  = np.mean(all_sops_hrf)                     # SOPs
    mean_sops_lif  = np.mean(all_sops_lif)                     # SOPs

    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset:        Adiac")
    print(f"Hidden units:   {args.n_hid}, Trials: {args.test_trials}")
    print(f"Readout mode:   {args.readout_mode}")
    print(f"Train accuracy: {mean_train_acc:.2f}% +/- {std_train_acc:.2f}%")
    print(f"Test accuracy:  {mean_test_acc:.2f}%  +/- {std_test_acc:.2f}%")
    print(f"Per-trial test: {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy:         {mean_energy:.3e} +/- {std_energy:.3e} J")
    print(f"SOPs (total):   {mean_sops:.3e}  "
          f"(HRF: {mean_sops_hrf:.3e}, LIF: {mean_sops_lif:.3e})")
    print(f"{'='*70}")

    results = {
        'dataset': 'Adiac',
        'args': vars(args),
        'n_trials': args.test_trials,
        'n_inp': n_inp,
        'n_out': n_out,
        'readout_mode': args.readout_mode,                     # NEW
        'readout_C': float(args.readout_C),
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
        'sops_mean':      float(mean_sops),                    # SOPs
        'sops_hrf_mean':  float(mean_sops_hrf),                # SOPs
        'sops_lif_mean':  float(mean_sops_lif),                # SOPs
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid':           int(args.n_hid),
        'base_seed':       int(args.seed),
        'sequence_length': int(seq_length),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = (
        f"results_adiac_nhid{args.n_hid}_{conn_lif_str}_{conn_hrf_str}"
        f"_{args.readout_mode}"                                # NEW
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    results_path = os.path.join(args.results_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

'''
"""
Spiking RON on the Spiking Heidelberg Digits (SHD) dataset.

SHD: 20-class spoken digit classification using cochlear spike trains.
- 700 input channels, Train: 8332, Test: 2088

Supports SPARSE INPUT PROJECTIONS to handle high-dimensional input
without scaling the reservoir. The input weight matrix x2h (700→n_hid)
can be made sparse so each reservoir neuron receives input from only
a random subset of input channels. This is:
  - Biologically plausible (sparse afferent connectivity)
  - Energy-efficient (fewer input synaptic operations)
  - Theoretically grounded (Johnson-Lindenstrauss preserves distances)

Download:
    wget https://zenkelab.org/datasets/shd_train.h5.gz
    wget https://zenkelab.org/datasets/shd_test.h5.gz
    gunzip shd_train.h5.gz shd_test.h5.gz
    mkdir -p data/SHD && mv shd_train.h5 shd_test.h5 data/SHD/
"""

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
from ucr_data_utils import get_SHD_data

# Add matplotlib for visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    """
    Apply a sparse mask to the input weight matrix x2h.
    
    Each reservoir neuron receives input from only a random subset
    of input channels. The density parameter controls the fraction
    of non-zero connections.
    
    Args:
        model: spiking_coESN_rescaled_II instance
        input_density: fraction of input connections to keep (0-1)
                       e.g., 0.05 means each neuron sees ~5% of inputs
        n_inp: number of input channels
        n_hid: number of reservoir neurons
        device: torch device
    
    Returns:
        n_input_connections: number of non-zero input connections
    """
    if input_density >= 1.0:
        n_input_connections = n_inp * n_hid
        print(f"  Input projection: DENSE ({n_input_connections} connections)")
        return n_input_connections
    
    # Create sparse mask
    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    
    # Ensure every neuron gets at least one input (avoid dead neurons)
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            random_inp = torch.randint(0, n_inp, (1,))
            mask[random_inp, j] = 1.0
    
    # Ensure every input channel connects somewhere (no wasted channels)
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            random_hid = torch.randint(0, n_hid, (1,))
            mask[i, random_hid] = 1.0
    
    # Apply mask and rescale to preserve expected magnitude
    # Scale by 1/sqrt(density) to keep variance roughly constant
    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale
    
    n_input_connections = int(mask.sum().item())
    print(f"  Input projection: SPARSE density={input_density:.3f}, "
          f"{n_input_connections}/{n_inp * n_hid} connections "
          f"({n_input_connections / (n_inp * n_hid) * 100:.1f}%), "
          f"scale={scale:.2f}")
    print(f"  Avg inputs per neuron: {n_input_connections / n_hid:.1f}/{n_inp}")
    
    return n_input_connections


def _extract_trajectories(model, x, last_steps=None):
    """
    Run the model step-by-step to capture HRF trajectories over time.
    Works with any version of the model (no forward() changes needed).
    
    Args:
        model: spiking_coESN_rescaled_II instance
        x: input tensor (B, L, n_inp)
        last_steps: how many final timesteps to save (None = all)
    
    Returns:
        features: same as model.forward() output features (B, feat_dim)
        rate_dict: same as model.forward() output rates
        trajectories: dict with 'hy' key → (B, steps_saved, n_hid) tensor
    """
    B = x.size(0)
    L = x.size(1)
    n_hid = model.n_hid
    device = model.device
    
    # Initialize states (same as model.forward)
    hy = torch.zeros(B, n_hid, device=device)
    hz = torch.zeros(B, n_hid, device=device)
    ref_period = torch.zeros(B, n_hid, device=device)
    s = torch.zeros(B, n_hid, device=device)
    lif_v = torch.zeros(B, n_hid, device=device)
    
    # Accumulators for features (same as model.forward)
    hy_sum = torch.zeros(B, n_hid, device=device)
    hy_sq_sum = torch.zeros(B, n_hid, device=device)
    total_hrf_spikes = 0.0
    total_lif_spikes = 0.0
    
    # Trajectory storage — always save ALL steps from the beginning
    # (last_steps can be used by the visualization to zoom in)
    hy_trajectory = torch.zeros(B, L, n_hid, device=device)
    
    # Also track input activity per timestep (to show when input is present)
    input_spike_counts = torch.zeros(B, L, device=device)
    
    for t in range(L):
        # Record input activity BEFORE processing
        input_spike_counts[:, t] = x[:, t].sum(dim=-1)  # total spikes across channels
        
        hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
            x[:, t], hy, hz, lif_v, s, ref_period=ref_period
        )
        
        hy_sum += hy
        hy_sq_sum += hy ** 2
        total_hrf_spikes += s.sum()
        total_lif_spikes += lif_s.sum()
        
        hy_trajectory[:, t, :] = hy
    
    # Now call the real forward() to get the actual features used for classification
    features, rate_dict = model(x)
    
    trajectories = {
        'hy': hy_trajectory.detach(),
        'input_spikes': input_spike_counts.detach(),  # (B, L)
    }
    return features, rate_dict, trajectories

def extract_features(loader, model, device, save_trajectories=False, n_samples=50, last_steps=None):
    """Extract reservoir features from data
    
    Args:
        save_trajectories: If True, save HRF trajectories for visualization
        n_samples: Number of neurons to sample for visualization
        last_steps: Number of final timesteps to save (None = save all timesteps)
    """
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []
    
    # For trajectory visualization
    trajectories_list = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, ncols=80, desc="Extracting features")):
            x = x.to(device)
            
            if save_trajectories and batch_idx == 0:  # Only save first batch
                features, r, trajectories = _extract_trajectories(
                    model, x, last_steps=last_steps
                )
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
    
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    # Process trajectories if saved
    saved_trajectories = None
    if save_trajectories and len(trajectories_list) > 0:
        saved_trajectories = trajectories_list[0]  # First batch only

    return (
        feats,
        labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item(),
        saved_trajectories
    )


def visualize_hrf_trajectories(trajectories, labels=None, n_samples=50,
                               save_path='hrf_trajectories_SHD.png',
                               title_prefix='Train', dt_bin_ms=5.6):
    """
    Visualize HRF neuron trajectories over time for the SHD dataset.
    
    Generates 4 panels:
      1. Input spike activity over time (shows when sample has actual input)
      2. HRF trajectories zoomed to INPUT PERIOD ONLY (auto-detected)
      3. HRF heatmap zoomed to input period
      4. Single neuron across classes (zoomed to input period)
    
    Args:
        trajectories: dict with 'hy' (B, L, n_hid) and 'input_spikes' (B, L)
        labels: class labels for the batch
        n_samples: number of neurons to plot
        save_path: where to save the figure
        title_prefix: 'Train' or 'Test'
        dt_bin_ms: physical time per bin in ms
    """
    hy_traj = trajectories['hy']  # (B, L, n_hid)
    B, T, n_hid = hy_traj.shape
    
    # Get input spike counts if available
    has_input_info = 'input_spikes' in trajectories
    if has_input_info:
        input_spikes = trajectories['input_spikes']  # (B, L)
        # Detect end of input for sample 0: last bin with >0 spikes
        sample0_spikes = input_spikes[0].cpu().numpy()
        nonzero_bins = np.nonzero(sample0_spikes)[0]
        if len(nonzero_bins) > 0:
            last_input_bin = nonzero_bins[-1]
            # Add 10% margin after input ends
            zoom_end = min(int(last_input_bin * 1.15) + 5, T)
        else:
            zoom_end = T
    else:
        sample0_spikes = None
        zoom_end = T
    
    # Sample neurons
    if n_samples > n_hid:
        n_samples = n_hid
    neuron_indices = np.linspace(0, n_hid - 1, n_samples, dtype=int)
    
    # Time axes
    time_ms_full = np.arange(T) * dt_bin_ms
    time_ms_zoom = np.arange(zoom_end) * dt_bin_ms
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    
    # --- Panel 1: Input spike activity ---
    ax = axes[0]
    if has_input_info:
        # Show input for first few samples
        n_show = min(5, B)
        for i in range(n_show):
            spk = input_spikes[i].cpu().numpy()
            lbl = int(labels[i]) if labels is not None else '?'
            alpha = 1.0 if i == 0 else 0.3
            lw = 1.5 if i == 0 else 0.5
            ax.plot(time_ms_full, spk, alpha=alpha, linewidth=lw,
                    label=f'Sample {i} (class {lbl})')
        ax.axvline(x=zoom_end * dt_bin_ms, color='red', linestyle='--', alpha=0.7,
                   label=f'Zoom boundary ({zoom_end * dt_bin_ms:.0f}ms)')
        ax.set_ylabel('Spikes per bin', fontsize=11)
        ax.set_title(f'{title_prefix} — Input Spike Activity Over Time '
                     f'(red line = zoom boundary)', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_ms_full[-1]])
    else:
        ax.text(0.5, 0.5, 'Input spike data not available', 
                transform=ax.transAxes, ha='center', fontsize=14)
    ax.set_xlabel('Time (ms)', fontsize=11)
    
    # --- Panel 2: Neuron trajectories ZOOMED to input period ---
    ax = axes[1]
    colors = cm.viridis(np.linspace(0, 1, n_samples))
    sample_label = int(labels[0]) if labels is not None else '?'
    
    for idx, neuron_idx in enumerate(neuron_indices):
        trajectory = hy_traj[0, :zoom_end, neuron_idx].cpu().numpy()
        ax.plot(time_ms_zoom, trajectory, color=colors[idx], alpha=0.6, linewidth=0.6)
    
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('HRF State (hy)', fontsize=11)
    ax.set_title(f'{title_prefix} — {n_samples} Neuron Trajectories DURING INPUT '
                 f'(Sample 0, class={sample_label}, 0–{zoom_end * dt_bin_ms:.0f}ms)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_ms_zoom[-1]])
    
    # --- Panel 3: Heatmap ZOOMED to input period ---
    ax = axes[2]
    heatmap_data = hy_traj[0, :zoom_end, neuron_indices].cpu().numpy().T  # (n_samples, zoom_end)
    
    vmax = np.percentile(np.abs(heatmap_data), 98)
    if vmax < 1e-6:
        vmax = 1.0  # avoid zero range
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', origin='lower',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, time_ms_zoom[-1], 0, n_samples])
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Neuron Index', fontsize=11)
    ax.set_title(f'{title_prefix} — HRF State Heatmap DURING INPUT '
                 f'({n_samples} Neurons, 0–{zoom_end * dt_bin_ms:.0f}ms)', fontsize=13)
    
    ytick_positions = np.linspace(0, n_samples - 1, min(10, n_samples), dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([neuron_indices[i] for i in ytick_positions])
    plt.colorbar(im, ax=ax, label='hy value')
    
    # --- Panel 4: Single neuron across classes ZOOMED ---
    ax = axes[3]
    if labels is not None:
        # Pick neuron with highest std DURING INPUT PERIOD (not explosion)
        neuron_stds = hy_traj[0, :zoom_end, :].cpu().numpy().std(axis=0)
        best_neuron = neuron_stds.argmax()
        
        unique_classes = np.unique(labels[:min(B, 10)])
        class_colors = cm.tab10(np.linspace(0, 1, min(len(unique_classes), 10)))
        
        for cls_idx, cls in enumerate(unique_classes[:10]):
            sample_idx = np.where(labels == cls)[0]
            if len(sample_idx) > 0:
                idx = sample_idx[0]
                if idx < B:
                    traj = hy_traj[idx, :zoom_end, best_neuron].cpu().numpy()
                    ax.plot(time_ms_zoom, traj, color=class_colors[cls_idx],
                           alpha=0.8, linewidth=1.0, label=f'Class {int(cls)}')
        
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel(f'HRF State (Neuron {best_neuron})', fontsize=11)
        ax.set_title(f'{title_prefix} — Single Neuron Across Classes DURING INPUT '
                     f'(Neuron {best_neuron}, highest variance in input period)', fontsize=13)
        ax.legend(fontsize=8, ncol=5, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_ms_zoom[-1]])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ HRF trajectories saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on SHD Dataset')
    
    # Model architecture
    parser.add_argument('--n_hid', type=int, default=256,
                       help='Number of hidden units (reservoir size)')
    parser.add_argument('--batch', type=int, default=128,
                       help='Batch size for training')
    
    # Oscillator parameters
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=1.88)
    parser.add_argument('--epsilon', type=float, default=0.022)
    parser.add_argument('--gamma_range', type=float, default=2.64)
    parser.add_argument('--epsilon_range', type=float, default=0.068)
    
    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=1.76,
                       help='Input scaling factor')
    parser.add_argument('--rho', type=float, default=0.95,
                       help='Spectral radius')
    
    # LIF/HRF parameters
    parser.add_argument('--theta_lif', type=float, default=0.05)
    parser.add_argument('--theta_rf', type=float, default=0.005)
    parser.add_argument('--tau_filter', type=float, default=20.0)
    
    # === SPARSE CONNECTIVITY ===
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                       help="Fraction of LIF→HRF connections (0-1)")
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0,
                       help="Fraction of HRF→LIF recurrent connections (0-1)")
    
    # === SPARSE INPUT PROJECTION (new for high-dim inputs) ===
    parser.add_argument('--input_density', type=float, default=1.0,
                       help="Fraction of input connections per neuron (0-1). "
                            "1.0=dense, 0.05=each neuron sees ~5%% of 700 channels. "
                            "Biologically plausible sparse afferent connectivity.")
    
    # SHD-specific parameters
    parser.add_argument('--num_steps', type=int, default=250,
                       help='Number of time bins for spike binning')
    parser.add_argument('--max_time', type=float, default=1.4,
                       help='Max recording duration in seconds')
    
    # Training options
    parser.add_argument('--cpu', action="store_true")
    parser.add_argument('--use_test', action="store_true",
                       help="Evaluate on test set")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_trials', type=int, default=5,
                       help='Number of trials for mean/std')
    parser.add_argument('--data_dir', type=str, default='data/SHD')
    parser.add_argument('--force_reload', action="store_true")
    
    # Readout regularization
    parser.add_argument('--readout_C', type=float, default=1.0,
                       help="Inverse regularization strength for logistic regression. "
                            "Lower = stronger regularization. Try 0.01-0.1 for SHD.")
    
    # Visualization options
    parser.add_argument('--visualize_trajectories', action="store_true",
                       help="Save HRF trajectory visualizations")
    parser.add_argument('--viz_n_samples', type=int, default=50,
                       help="Number of neurons to visualize")
    parser.add_argument('--viz_last_steps', type=int, default=None,
                       help="Number of final timesteps to visualize (None = full sequence)")
    parser.add_argument('--visualize_test', action="store_true",
                       help="Also visualize test set trajectories (in addition to train)")
    
    # Results
    parser.add_argument('--results_dir', type=str, default='results_shd')
    
    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON SHD (SPIKING HEIDELBERG DIGITS) DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print(f"✅ Using device: {device}")
    
    n_inp = 700   # SHD has 700 cochlear channels
    n_out = 20    # 20 classes
    bs_test = 256
    
    gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    # Compute physical time per bin for visualization
    dt_bin_ms = (args.max_time / args.num_steps) * 1000  # ms per bin

    print("\n=== Loading SHD Dataset ===")
    train_loader, valid_loader, test_loader = get_SHD_data(
        batch_train=args.batch, batch_test=bs_test,
        data_dir=args.data_dir, num_steps=args.num_steps,
        max_time=args.max_time
    )
    print(f"✅ Loaded SHD dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Num time steps: {args.num_steps}, Input channels: {n_inp}")

    # Verify shapes
    sample_x, sample_y = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    n_features = sample_x.shape[2]
    print(f"   Sample shape: {sample_x.shape}  (batch, time, channels)")
    assert n_features == n_inp, f"Expected {n_inp} channels, got {n_features}"

    # Store results across trials
    all_test_accs, all_train_accs, all_energies = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []
    all_n_input_connections = []

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
        
        # Apply sparse input projection AFTER model creation
        n_input_connections = apply_sparse_input_projection(
            model, args.input_density, n_inp, args.n_hid, device
        )
        all_n_input_connections.append(n_input_connections)

        print(f"✅ Model created")

        # Save trajectories only on first trial and if requested
        save_traj = args.visualize_trajectories and trial == 0

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train, train_trajectories = extract_features(
            train_loader, model, device,
            save_trajectories=save_traj,
            n_samples=args.viz_n_samples,
            last_steps=args.viz_last_steps
        )
        print(f"✅ Training features: {train_feats.shape}")
        
        # Visualize TRAIN trajectories if requested
        if save_traj and train_trajectories is not None:
            print("\n=== Visualizing TRAIN HRF Trajectories ===")
            os.makedirs(args.results_dir, exist_ok=True)
            viz_path = os.path.join(args.results_dir,
                f'hrf_trajectories_SHD_TRAIN_nhid{args.n_hid}_trial{trial}.png')
            visualize_hrf_trajectories(
                train_trajectories,
                labels=train_labels[:args.batch],
                n_samples=args.viz_n_samples,
                save_path=viz_path,
                title_prefix='SHD Train',
                dt_bin_ms=dt_bin_ms
            )
        
        if args.use_test:
            save_test_traj = args.visualize_test and trial == 0
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, test_trajectories = extract_features(
                test_loader, model, device,
                save_trajectories=save_test_traj,
                n_samples=args.viz_n_samples,
                last_steps=args.viz_last_steps
            )
            print(f"✅ Test features: {test_feats.shape}")
            
            # Visualize TEST trajectories if requested
            if save_test_traj and test_trajectories is not None:
                print("\n=== Visualizing TEST HRF Trajectories ===")
                viz_path_test = os.path.join(args.results_dir,
                    f'hrf_trajectories_SHD_TEST_nhid{args.n_hid}_trial{trial}.png')
                visualize_hrf_trajectories(
                    test_trajectories,
                    labels=test_labels[:bs_test],
                    n_samples=args.viz_n_samples,
                    save_path=viz_path_test,
                    title_prefix='SHD Test',
                    dt_bin_ms=dt_bin_ms
                )
        else:
            test_feats, test_labels = train_feats, train_labels

        scaler = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)

        print("\n=== Training Logistic Regression Readout ===")
        clf = LogisticRegression(
            max_iter=2000, verbose=0, n_jobs=-1,
            multi_class='multinomial', solver='lbfgs',
            C=args.readout_C
        ).fit(train_feats, train_labels)
        
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
    print(f"Dataset: SHD (Spiking Heidelberg Digits)")
    print(f"Hidden units: {args.n_hid}, Trials: {args.test_trials}")
    print(f"Input density: {args.input_density}")
    print(f"Num steps: {args.num_steps}, Input channels: {n_inp}")
    print(f"Train accuracy:  {mean_train_acc:.2f}% ± {std_train_acc:.2f}%")
    print(f"Test accuracy:   {mean_test_acc:.2f}% ± {std_test_acc:.2f}%")
    print(f"Per-trial test:  {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy: {mean_energy:.3e} ± {std_energy:.3e} J")
    print(f"{'='*70}")

    results = {
        'dataset': 'SHD',
        'args': vars(args),
        'n_trials': args.test_trials,
        'n_inp': n_inp,
        'n_out': n_out,
        'num_steps': args.num_steps,
        'max_time': args.max_time,
        'input_density': float(args.input_density),
        'readout_C': float(args.readout_C),
        'n_input_connections_mean': float(np.mean(all_n_input_connections)),
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
    inp_str = f"inp{args.input_density:.2f}" if args.input_density < 1.0 else "inpDense"
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = (
        f"results_shd_nhid{args.n_hid}_steps{args.num_steps}"
        f"_{inp_str}_{conn_lif_str}_{conn_hrf_str}"
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    results_path = os.path.join(args.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
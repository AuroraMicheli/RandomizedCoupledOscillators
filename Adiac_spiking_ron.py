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

# Import the model from utils
from utils_aurora import estimate_snn_energy_sparse
from utils import get_Adiac_data

# Add matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class spiking_coESN_rescaled_II(nn.Module):
    """
    Spiking reservoir-only version (no trainable readout).
    Batch-first input (B, L, I)
    Adds customizable LIF/HRF thresholds and feature options, including filtered spikes.
    
    READOUT STRATEGY: Time-Pooled Statistics (RMS + Std + Final State)
    - RMS captures oscillation amplitude (energy)
    - Std captures temporal variability (dynamics)
    - Final state captures endpoint phase
    - Provides 3*n_hid features capturing temporal dynamics efficiently
    - Biological plausibility: mirrors rate and temporal coding
    - Minimal computational overhead: simple accumulation during forward pass
    
    ENERGY OPTIMIZATION: Sparse connectivity
    - Sparse LIF→HRF connectivity (reduces LIF-driven synaptic operations)
    - Sparse HRF→HRF recurrent connectivity (reduces recurrent operations)
    - Both reduce synaptic operations while maintaining representational capacity
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, 
                 theta_lif, theta_rf, tau_filter, count_lif_spikes=False, 
                 sparse_lif2hrf=True, connectivity_lif2hrf=0.1,
                 sparse_hrf2lif=True, connectivity_hrf2lif=0.1,
                 device='cpu', fading=False):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt
        self.theta_lif = theta_lif
        self.theta_rf = theta_rf
        self.tau_filter = tau_filter
        self.count_lif_spikes = count_lif_spikes
        self.sparse_lif2hrf = sparse_lif2hrf
        self.connectivity_lif2hrf = connectivity_lif2hrf
        self.sparse_hrf2lif= sparse_hrf2lif
        self.connectivity_hrf2lif = connectivity_hrf2lif

        # Parameters (same as before)
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = torch.rand(n_hid, device=device) * (gamma_max - gamma_min) + gamma_min
        else:
            self.gamma = torch.tensor(gamma, device=device)
            gamma_min = gamma_max = gamma

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = torch.tensor(epsilon, device=device)
            eps_min = eps_max = epsilon

        # ===== HRF→HRF Recurrent Weights (POTENTIALLY SPARSE) =====
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        
        if gamma_min == gamma_max and eps_min == eps_max and gamma_max == 1:
            leaky = dt**2
            I = torch.eye(n_hid)
            h2h = h2h * leaky + (I * (1 - leaky))
            h2h = spectral_norm_scaling(h2h, rho)
            h2h = (h2h + I * (leaky - 1)) * (1 / leaky)
        else:
            h2h = spectral_norm_scaling(h2h, rho)
        
        # Apply sparsity to HRF→HRF connections
        if sparse_hrf2lif:
            h2h = h2h.to(device)  # Move to device
            mask_hrf2lif = (torch.rand(n_hid, n_hid, device=device) < connectivity_hrf2lif).float()
            h2h = h2h * mask_hrf2lif
            n_connections_hrf2lif = mask_hrf2lif.sum().item()
            self.n_hrf2lif_connections = n_connections_hrf2lif
            print(f"HRF→LIF sparse recurrent connectivity: {n_connections_hrf2lif}/{n_hid**2} connections ({connectivity_hrf2lif*100:.1f}%)")
        else:
            h2h = h2h.to(device)  # Move to device
            self.n_hrf2lif_connections = n_hid ** 2
            print(f"HRF→LIF dense recurrent connectivity: {n_hid**2}/{n_hid**2} connections (100%)")
        
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # Input weights (always dense)
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        
        # Rescaled bias
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)
        
        # ===== LIF→HRF Synaptic Weights (POTENTIALLY SPARSE) =====
        if sparse_lif2hrf:
            lif2hrf_full = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            # Create sparse mask: only 'connectivity_lif2hrf' fraction of weights are non-zero
            mask_lif2hrf = (torch.rand(n_hid, n_hid, device=device) < connectivity_lif2hrf).float()
            lif2hrf = lif2hrf_full * mask_lif2hrf
            
            # Count actual connections for energy reporting
            n_connections_lif2hrf = mask_lif2hrf.sum().item()
            self.n_lif2hrf_connections = n_connections_lif2hrf
            print(f"LIF→HRF sparse connectivity: {n_connections_lif2hrf}/{n_hid**2} connections ({connectivity_lif2hrf*100:.1f}%)")
        else:
            # Dense connectivity (baseline)
            lif2hrf = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 0.2
            self.n_lif2hrf_connections = n_hid ** 2
            print(f"LIF→HRF dense connectivity: {n_hid**2}/{n_hid**2} connections (100%)")
            
        self.lif2hrf = nn.Parameter(lif2hrf, requires_grad=False)

        # Spike Gain
        self.spike_gain = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)


    def bio_cell(self, x, hy, hz, lif_v, s, ref_period=None):   
        dt = self.dt
        device = self.device
        theta_lif = self.theta_lif
        theta_rf = self.theta_rf
        
        # ==== LIF parameters ====
        lif_tau_m = 20.0
        lif_tau_ref = 1e9
        spike_gain = self.spike_gain

        # ==== HRF parameters ====
        alpha = 0.0
        beta = 0.0
        tau_ref = 0.25

        # ==== Input drive (includes sparse HRF→HRF recurrence) ====
        #input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias

        # ==== LIF membrane update ====
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s = (lif_v > theta_lif).float()
        lif_v = lif_v - lif_s * theta_lif
        
        # ==== HRF oscillator dynamics (with sparse LIF→HRF coupling) ====
        drive = torch.matmul(lif_s, self.lif2hrf)
        
        hz = hz + dt * (drive - self.gamma * hy - self.epsilon * hz)
        if self.fading:
            hz = hz - dt * hz

        #hz = torch.clamp(hz, -15, 15)  # ← ADD THIS

        hy = hy + dt * hz
        if self.fading:
            hy = hy - dt * hy

        #hy = torch.tanh(hy)
        # ==== HRF spike + reset + refractory ====
        if ref_period is None:
            ref_period = torch.zeros_like(hz)
            
        s = (hy - theta_rf - ref_period > 0).float()
        
        hy = hy * (1 - s * alpha)
        hz = hz * (1 - s * beta)

        ref_decay = torch.exp(-torch.as_tensor(dt / tau_ref, device=device))
        ref_period = ref_period * ref_decay + s
        
        return hy, hz, s, ref_period, lif_v, lif_s

    def forward(self, x, return_trajectories=False, last_steps=100):
        """
        Forward pass with time-pooled statistical features.
        
        Args:
            x: Input tensor (B, L, I)
            return_trajectories: If True, return HRF trajectories for visualization
            last_steps: Number of final timesteps to save for visualization
        
        Returns features of size (B, 3*n_hid) or (B, n_hid) depending on configuration
        """
        B = x.size(0)
        L = x.size(1)
        n_hid = self.n_hid
        device = self.device
        
        # Initialize states
        hy = torch.zeros(B, n_hid, device=device)
        hz = torch.zeros(B, n_hid, device=device)
        ref_period = torch.zeros(B, n_hid, device=device)
        s = torch.zeros(B, n_hid, device=device)
        lif_v = torch.zeros(B, n_hid, device=device)
        
        # Accumulators for temporal statistics
        hy_sum = torch.zeros(B, n_hid, device=device)
        hy_sq_sum = torch.zeros(B, n_hid, device=device)
        
        # Spike counting for energy analysis
        total_hrf_spikes = 0.0
        total_lif_spikes = 0.0
        
        # Trajectory storage if requested
        if return_trajectories:
            # Determine how many steps to save
            steps_to_save = min(last_steps, L)
            start_idx = L - steps_to_save
            hy_trajectory = torch.zeros(B, steps_to_save, n_hid, device=device)
        
        for t in range(L):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )
            
            # Accumulate statistics (minimal overhead)
            hy_sum += hy
            hy_sq_sum += hy ** 2
            
            # Count spikes for energy analysis
            total_hrf_spikes += s.sum()
            total_lif_spikes += lif_s.sum()
            
            # Save trajectory if requested
            if return_trajectories and t >= start_idx:
                hy_trajectory[:, t - start_idx, :] = hy
        
        # Compute temporal features
        hy_mean = hy_sum / L
        hy_rms = torch.sqrt(hy_sq_sum / L + 1e-8)
        hy_std = torch.sqrt(torch.clamp(hy_sq_sum / L - hy_mean ** 2, min=1e-8))
        hy_final = hy
        
        # Select features (currently using hy_final based on your code)
        features = hy_final
        # Change to:
        '''
        features = torch.cat([
            hy_rms,    # Oscillation amplitude
            hy_std,    # Temporal variability  
            hy_final   # Endpoint state
        ], dim=1)
    
        
        features = torch.cat([
            hy_rms,    # Oscillation amplitude
            hy_std,    # Temporal variability  
        ], dim=1)
        '''
        #features=hy_mean
        # Compute average firing rates for energy analysis
        r_hrf = total_hrf_spikes / (B * L * n_hid)
        r_lif = total_lif_spikes / (B * L * n_hid)
        r_total = (r_hrf + r_lif) if self.count_lif_spikes else r_hrf

        rate_dict = {
            "r_total": r_total.detach(),
            "r_hrf": r_hrf.detach(),
            "r_lif": r_lif.detach()
        }
        
        if return_trajectories:
            trajectory_dict = {
                'hy': hy_trajectory.detach()  # (B, last_steps, n_hid)
            }
            return features, rate_dict, trajectory_dict
        else:
            return features, rate_dict



def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            
            # Adiac data is already in format (batch, time, features)
            # No need to reshape like MNIST
            if save_trajectories and batch_idx == 0:  # Only save first batch
                # If last_steps is None, use full sequence length
                steps_to_save = last_steps if last_steps is not None else x.size(1)
                features, r, trajectories = model(x, return_trajectories=True, last_steps=steps_to_save)
                trajectories_list.append(trajectories)
            else:
                features, r = model(x)
            
            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(y)
    
    # Handle empty loader case
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


def visualize_hrf_trajectories(trajectories, n_samples=50, save_path='hrf_trajectories_Adiac.png', 
                               title_prefix='Train'):
    """
    Visualize HRF neuron trajectories over time for ONE sample.
    
    Args:
        trajectories: dict with keys 'hy' (B, last_steps, n_hid)
        n_samples: number of neurons to plot
        save_path: where to save the figure
        title_prefix: 'Train' or 'Test' to label the plot
    """
    hy_traj = trajectories['hy']  # (B, last_steps, n_hid)
    
    B, T, n_hid = hy_traj.shape
    
    # Sample neurons uniformly
    if n_samples > n_hid:
        n_samples = n_hid
    neuron_indices = np.linspace(0, n_hid-1, n_samples, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Multiple neurons from first batch sample
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_samples))
    
    for idx, neuron_idx in enumerate(neuron_indices):
        trajectory = hy_traj[0, :, neuron_idx].cpu().numpy()  # First sample from batch
        ax.plot(trajectory, color=colors[idx], alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('HRF State (hy)', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF Trajectories for {n_samples} Sampled Neurons (1 Sample, Full Sequence)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T-1])
    
    # Plot 2: Heatmap of all sampled neurons across time
    ax = axes[1]
    heatmap_data = hy_traj[0, :, neuron_indices].cpu().numpy().T  # (n_samples, T)
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(f'{title_prefix} - HRF State Heatmap ({n_samples} Neurons, 1 Sample, Full Sequence)', fontsize=14)
    
    # Set y-ticks to show actual neuron indices
    ytick_positions = np.linspace(0, n_samples-1, min(10, n_samples), dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([neuron_indices[i] for i in ytick_positions])
    
    plt.colorbar(im, ax=ax, label='HRF State Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ HRF trajectories saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Spiking RON on Adiac Dataset')
    
    # Model architecture
    parser.add_argument('--n_hid', type=int, default=100,
                       help='Number of hidden units (reservoir size)')
    parser.add_argument('--batch', type=int, default=120,
                       help='Batch size for training')
    
    # Oscillator parameters
    parser.add_argument('--dt', type=float, default=0.097,
                       help='Time step size')
    parser.add_argument('--gamma', type=float, default=3.077,
                       help='Gamma parameter (damping)')
    parser.add_argument('--epsilon', type=float, default=0.02,
                       help='Epsilon parameter (stiffness)')
    parser.add_argument('--gamma_range', type=float, default=3.28,
                       help='Range for gamma heterogeneity')
    parser.add_argument('--epsilon_range', type=float, default=0.20,
                       help='Range for epsilon heterogeneity')
    
    # Input/Reservoir parameters
    parser.add_argument('--inp_scaling', type=float, default=9.87,
                       help='Input scaling factor')
    parser.add_argument('--rho', type=float, default=0.87,
                       help='Spectral radius')
    
    # LIF/HRF parameters
    parser.add_argument('--theta_lif', type=float, default=0.05,
                       help='LIF threshold')
    parser.add_argument('--theta_rf', type=float, default=0.005,
                       help='HRF threshold')
    parser.add_argument('--tau_filter', type=float, default=20.0,
                       help='Filter time constant')
    
    # Sparse connectivity options
    parser.add_argument('--sparse_lif2hrf', action="store_true",
                       help="Use sparse LIF→HRF connectivity")
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0,
                       help="Fraction of LIF→HRF connections (0-1), 1.0 = dense")

    # HRF→LIF sparse recurrent connectivity options
    parser.add_argument('--sparse_hrf2lif', action="store_true",
                        help="Use sparse HRF→LIF recurrent connectivity")
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
    parser.add_argument('--results_dir', type=str, default='results_adiac',
                       help="Directory to save results")
    
    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON ADIAC DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print(f"✅ Using device: {device}")
    
    # Dataset parameters
    n_inp = 1  # Adiac is univariate
    n_out = 37  # 37 classes for Adiac
    bs_test = 30  # Using same batch size as in the original Adiac code
    
    # Oscillator parameter ranges
    gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

    # Determine if using sparse connectivity
    use_sparse_lif2hrf = args.sparse_lif2hrf
    use_sparse_hrf2lif = args.sparse_hrf2lif

    print("\n=== Loading Adiac Dataset ===")
    # Load with whole_train=True for multiple trials
    train_loader, valid_loader, test_loader = get_Adiac_data(args.batch, bs_test, whole_train=True)
    
    print(f"✅ Loaded Adiac dataset")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")

    # Get sequence length from a sample batch
    sample_x, _ = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    print(f"   Sequence length: {seq_length}")

    # Store results across trials
    all_test_accs = []
    all_train_accs = []
    all_energies = []
    all_r_hrf = []
    all_r_lif = []
    all_r_total = []

    # Run multiple trials
    for trial in range(args.test_trials):
        print("\n" + "=" * 70)
        print(f"TRIAL {trial + 1}/{args.test_trials}")
        print("=" * 70)
        
        print("\n=== Building Spiking RON ===")
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

        print(f"✅ Model created")
        if use_sparse_lif2hrf:
            print(f"   LIF→HRF connectivity: SPARSE ({args.connectivity_lif2hrf*100:.1f}%)")
        else:
            print(f"   LIF→HRF connectivity: DENSE (100%)")

        if use_sparse_hrf2lif:
            print(f"   HRF→LIF connectivity: SPARSE ({args.connectivity_hrf2lif*100:.1f}%)")
        else:
            print(f"   HRF→LIF connectivity: DENSE (100%)")

        # Extract features
        print("\n=== Extracting Reservoir Features ===")
        
        # Save trajectories only on first trial and if requested
        save_traj = args.visualize_trajectories and trial == 0
        
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train, train_trajectories = extract_features(
            train_loader, model, device, 
            save_trajectories=save_traj,
            n_samples=args.viz_n_samples,
            last_steps=args.viz_last_steps
        )
        print(f"✅ Training features extracted: {train_feats.shape}")
        
        # Visualize TRAIN trajectories if requested
        if save_traj and train_trajectories is not None:
            print("\n=== Visualizing TRAIN HRF Trajectories ===")
            os.makedirs(args.results_dir, exist_ok=True)
            viz_path = os.path.join(args.results_dir, 
                                   f'hrf_trajectories_TRAIN_trial{trial}_nhid{args.n_hid}.png')
            visualize_hrf_trajectories(
                train_trajectories, 
                n_samples=args.viz_n_samples,
                save_path=viz_path,
                title_prefix='TRAIN'
            )
        
        if args.use_test:
            # Extract test features (and optionally trajectories)
            save_test_traj = args.visualize_test and trial == 0
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, test_trajectories = extract_features(
                test_loader, model, device,
                save_trajectories=save_test_traj,
                n_samples=args.viz_n_samples,
                last_steps=args.viz_last_steps
            )
            print(f"✅ Test features extracted: {test_feats.shape}")
            
            # Visualize TEST trajectories if requested
            if save_test_traj and test_trajectories is not None:
                print("\n=== Visualizing TEST HRF Trajectories ===")
                viz_path_test = os.path.join(args.results_dir, 
                                       f'hrf_trajectories_TEST_trial{trial}_nhid{args.n_hid}.png')
                visualize_hrf_trajectories(
                    test_trajectories, 
                    n_samples=args.viz_n_samples,
                    save_path=viz_path_test,
                    title_prefix='TEST'
                )
        else:
            test_feats, test_labels = train_feats, train_labels
            r_tot_test, r_hrf_test, r_lif_test = r_tot_train, r_hrf_train, r_lif_train

        # Standardize features
        print(f"\n=== RAW Feature Statistics (BEFORE scaling) ===")
        print(f"RAW Train: mean={train_feats.mean():.4f}, std={train_feats.std():.4f}, min={train_feats.min():.4f}, max={train_feats.max():.4f}")
        print(f"RAW Test: mean={test_feats.mean():.4f}, std={test_feats.std():.4f}, min={test_feats.min():.4f}, max={test_feats.max():.4f}")


        # After extracting train_feats, BEFORE scaling:
        print(f"\n=== Per-Neuron Analysis ===")
        train_neuron_means = train_feats.mean(axis=0)
        train_neuron_stds = train_feats.std(axis=0)

        print(f"Neurons with std < 0.01 (dead): {(train_neuron_stds < 0.01).sum()}/100")
        print(f"Neurons with std > 0.5 (extreme): {(train_neuron_stds > 0.5).sum()}/100")
        print(f"Mean std per neuron: {train_neuron_stds.mean():.4f}")
        print(f"Std of stds (variance in neuron activity): {train_neuron_stds.std():.4f}")

        # Find the most extreme neurons
        extreme_neurons = np.where(train_neuron_stds > 0.5)[0]
        if len(extreme_neurons) > 0:
            print(f"\nExtreme neurons: {extreme_neurons[:10]}")  # Show first 10
            for idx in extreme_neurons[:3]:
                print(f"  Neuron {idx}: mean={train_neuron_means[idx]:.3f}, std={train_neuron_stds[idx]:.3f}")
                print(f"    Train range: [{train_feats[:, idx].min():.3f}, {train_feats[:, idx].max():.3f}]")
                print(f"    Test range: [{test_feats[:, idx].min():.3f}, {test_feats[:, idx].max():.3f}]")



        print("\n=== Standardizing Features ===")
        scaler = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        print("✅ Features standardized")

        print(f"\n=== Feature Statistics ===")
        print(f"Train features shape: {train_feats.shape}")
        print(f"Train: mean={train_feats.mean():.4f}, std={train_feats.std():.4f}, min={train_feats.min():.4f}, max={train_feats.max():.4f}")
        print(f"Test features shape: {test_feats.shape}")
        print(f"Test: mean={test_feats.mean():.4f}, std={test_feats.std():.4f}, min={test_feats.min():.4f}, max={test_feats.max():.4f}")

        # Train readout classifier
        print("\n=== Training Logistic Regression Readout ===")
        clf = LogisticRegression(max_iter=1000, verbose=0, n_jobs=-1).fit(train_feats, train_labels)


        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc = clf.score(test_feats, test_labels) * 100
        
        print(f"✅ Training accuracy: {train_acc:.2f}%")
        print(f"✅ Test accuracy: {test_acc:.2f}%")

        # Firing rate statistics
        print(f"\n=== Firing Rate Statistics ===")
        print(f"Average firing rate r_hrf (train): {r_hrf_train:.4f}")
        print(f"Average firing rate r_lif (train): {r_lif_train:.4f}")
        print(f"Average firing rate r_total (train): {r_tot_train:.4f}")

        # Energy estimation
        T = seq_length
        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_hrf_train,
            r_lif=r_lif_train,
            n_hid=args.n_hid,
            T=T,
            lif2hrf_connections=model.n_lif2hrf_connections,
            include_lif=True
        )
        
        print(f"\n=== Energy Estimation ===")
        print(f"Total SOPs: {snn_energy['SOPs']:.3e}")
        print(f"Energy (J): {snn_energy['Energy_J']:.3e}")

        # Store results
        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    # Compute statistics across trials
    mean_test_acc = np.mean(all_test_accs)
    std_test_acc = np.std(all_test_accs)
    mean_train_acc = np.mean(all_train_accs)
    std_train_acc = np.std(all_train_accs)
    mean_energy = np.mean(all_energies)
    std_energy = np.std(all_energies)
    mean_r_hrf = np.mean(all_r_hrf)
    mean_r_lif = np.mean(all_r_lif)
    mean_r_total = np.mean(all_r_total)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY (ACROSS ALL TRIALS)")
    print("=" * 70)
    print(f"Dataset: Adiac")
    print(f"Model: Spiking RON (Reservoir + Oscillators)")
    print(f"Hidden units: {args.n_hid}")
    print(f"Connectivity: {'DENSE LIF→HRF (100%)' if not use_sparse_lif2hrf else f'SPARSE LIF→HRF ({args.connectivity_lif2hrf*100:.1f}%)'}")
    print(f"Connectivity: {'DENSE HRF→LIF (100%)' if not use_sparse_hrf2lif else f'SPARSE HRF→LIF ({args.connectivity_hrf2lif*100:.1f}%)'}")
    print(f"Number of trials: {args.test_trials}")
    print(f"Base seed: {args.seed}")
    print(f"Sequence length: {seq_length}")
    print(f"-" * 70)
    print(f"Training accuracy:   {mean_train_acc:.2f}% ± {std_train_acc:.2f}%")
    print(f"Test accuracy:       {mean_test_acc:.2f}% ± {std_test_acc:.2f}%")
    print(f"-" * 70)
    print(f"Test accuracies per trial: {[f'{acc:.2f}' for acc in all_test_accs]}")
    print(f"-" * 70)
    print(f"HRF firing rate:  {mean_r_hrf:.4f}")
    print(f"LIF firing rate:  {mean_r_lif:.4f}")
    print(f"Total firing rate: {mean_r_total:.4f}")
    print(f"-" * 70)
    print(f"Energy efficiency: {mean_energy:.3e} ± {std_energy:.3e} J")
    print("=" * 70)

    # Save aggregated results
    print("\n=== Saving Aggregated Results ===")
    results = {
        'dataset': 'Adiac',
        'args': vars(args),
        'n_trials': args.test_trials,
        'train_acc_mean': float(mean_train_acc),
        'train_acc_std': float(std_train_acc),
        'test_acc_mean': float(mean_test_acc),
        'test_acc_std': float(std_test_acc),
        'test_accs_all': [float(x) for x in all_test_accs],
        'r_hrf_mean': float(mean_r_hrf),
        'r_lif_mean': float(mean_r_lif),
        'r_tot_mean': float(mean_r_total),
        'energy_J_mean': float(mean_energy),
        'energy_J_std': float(std_energy),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid': int(args.n_hid),
        'base_seed': int(args.seed),
        'sequence_length': int(seq_length)
    }

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save with descriptive filename
    conn_lif2hrf_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif2hrf{args.connectivity_lif2hrf:.1f}"
    conn_hrf2lif_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf2lif{args.connectivity_hrf2lif:.1f}"
    results_filename = f"results_adiac_nhid{args.n_hid}_{conn_lif2hrf_str}_{conn_hrf2lif_str}_trials{args.test_trials}_seed{args.seed}.json"
    results_path = os.path.join(args.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
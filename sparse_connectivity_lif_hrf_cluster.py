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


# --- RESCALED SPIKING coESN (Reservoir only) ---
class spiking_coESN_rescaled(nn.Module):
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
    
    ENERGY OPTIMIZATION: Sparse LIF→HRF connectivity
    - Only 10% of connections are active (biologically plausible)
    - Reduces synaptic operations by 90% while maintaining representational capacity
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, 
                 theta_lif, theta_rf, tau_filter, count_lif_spikes=False, 
                 sparse_lif2hrf=True, connectivity=0.1, device='cpu', fading=False):
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
        self.connectivity = connectivity

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

        # Recurrent and input weights (same as before)
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        if gamma_min == gamma_max and eps_min == eps_max and gamma_max == 1:
            leaky = dt**2
            I = torch.eye(n_hid)
            h2h = h2h * leaky + (I * (1 - leaky))
            h2h = spectral_norm_scaling(h2h, rho)
            self.h2h = (h2h + I * (leaky - 1)) * (1 / leaky)
        else:
            h2h = spectral_norm_scaling(h2h, rho)
            self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        
        # Rescaled bias
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)
        
        # LIF -> HRF Synaptic Weights (SPARSE for energy efficiency)
        if sparse_lif2hrf:
            lif2hrf_full = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            # Create sparse mask: only 'connectivity' fraction of weights are non-zero
            mask = (torch.rand(n_hid, n_hid, device=device) < connectivity).float()
            lif2hrf = lif2hrf_full * mask
            
            # Count actual connections for energy reporting
            n_connections = mask.sum().item()
            self.n_lif2hrf_connections = n_connections
            print(f"LIF→HRF sparse connectivity: {n_connections}/{n_hid**2} connections ({connectivity*100:.1f}%)")
        else:
            # Dense connectivity (baseline)
            lif2hrf = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            self.n_lif2hrf_connections = n_hid ** 2
            
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

        # ==== Input drive ====
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias
        
        # ==== LIF membrane update ====
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s = (lif_v > theta_lif).float()
        lif_v = lif_v - lif_s * theta_lif
        
        # ==== HRF oscillator dynamics ====
        # Sparse LIF→HRF coupling (most connections are zero for efficiency)
        drive = torch.matmul(lif_s, self.lif2hrf)
        
        hz = hz + dt * (drive - self.gamma * hy - self.epsilon * hz)
        if self.fading:
            hz = hz - dt * hz
        hy = hy + dt * hz
        if self.fading:
            hy = hy - dt * hy

        # ==== HRF spike + reset + refractory ====
        if ref_period is None:
            ref_period = torch.zeros_like(hz)
            
        s = (hy - theta_rf - ref_period > 0).float()
        
        hy = hy * (1 - s * alpha)
        hz = hz * (1 - s * beta)

        ref_decay = torch.exp(-torch.as_tensor(dt / tau_ref, device=device))
        ref_period = ref_period * ref_decay + s
        
        return hy, hz, s, ref_period, lif_v, lif_s

    def forward(self, x):
        """
        Forward pass with time-pooled statistical features.
        Returns features of size (B, 3*n_hid):
        - hy_rms: root mean square (oscillation amplitude/energy)
        - hy_std: temporal standard deviation (variability/dynamics)
        - hy_final: final HRF state (sequence endpoint phase)
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
        

        # Compute temporal features
        hy_mean = hy_sum / L
        hy_rms = torch.sqrt(hy_sq_sum / L + 1e-8)  # Root mean square (oscillation amplitude)
        hy_std = torch.sqrt(torch.clamp(hy_sq_sum / L - hy_mean ** 2, min=1e-8))  # Temporal variability
        hy_final = hy  # Final state (phase information)
        
        # Concatenate features: 3*n_hid dimensional
        '''
        features = torch.cat([
            hy_rms,    # RMS captures oscillation amplitude (always positive, informative)
            hy_std,    # Std captures dynamics/variability
            hy_final   # Final state captures endpoint phase
        ], dim=1)
        '''
      
        #features = hy_final
        features = hy_mean

        # Compute average firing rates for energy analysis
        r_hrf = total_hrf_spikes / (B * L * n_hid)
        r_lif = total_lif_spikes / (B * L * n_hid)
        r_total = (r_hrf + r_lif) if self.count_lif_spikes else r_hrf

        return features, {
            "r_total": r_total.detach(),
            "r_hrf": r_hrf.detach(),
            "r_lif": r_lif.detach()
        }


########## MAIN with Rescaled Model and Time-Pooled Features ##########

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Spiking coESN with Time-Pooled Readout Features')
    parser.add_argument('--n_hid', type=int, default=800)
    parser.add_argument('--batch', type=int, default=256)  
    parser.add_argument('--dt', type=float, default=0.042)
    parser.add_argument('--gamma', type=float, default=2.7)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--gamma_range', type=float, default=2)
    parser.add_argument('--epsilon_range', type=float, default=1)

    parser.add_argument('--inp_scaling', type=float, default=2.0)   
    parser.add_argument('--theta_lif', type=float, default=0.05)
    parser.add_argument('--theta_rf', type=float, default=0.005)
    parser.add_argument('--tau_filter', type=float, default=20.0)

    parser.add_argument('--rho', type=float, default=0.99)
    parser.add_argument('--cpu', action="store_true")
    parser.add_argument('--use_test', action="store_true")

    # Energy efficiency options
    parser.add_argument('--sparse_lif2hrf', action="store_true", 
                        help="Use sparse LIF→HRF connectivity (default: dense)")
    parser.add_argument('--connectivity', type=float, default=1.0,
                        help="Fraction of LIF→HRF connections (0-1), 1.0 = dense")

    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Results directory
    parser.add_argument('--results_dir', type=str, default='results_sparse_connectivity',
                        help="Directory to save results")

    args = parser.parse_args()

    print("=" * 70)
    print("SPARSE CONNECTIVITY EXPERIMENT")
    print("=" * 70)
    print(args)
    print("=" * 70)

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"\n✅ Set random seed to {args.seed}")

    # --- setup ---
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print(f"✅ Using device: {device}")
    
    n_inp = 1
    n_out = 10
    bs_test = 100
    gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

    # Determine if using sparse connectivity
    use_sparse = args.connectivity < 1.0

    # --- model ---
    model = spiking_coESN_rescaled( 
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
        sparse_lif2hrf=use_sparse,
        connectivity=args.connectivity,
        device=device
    ).to(device)

    train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

    print("\n=== Generating time-pooled statistical features ===")
    print(f"Feature extraction strategy: Final State Only")
    print(f"Feature dimensionality: {args.n_hid}")
    if use_sparse:
        print(f"Using SPARSE LIF→HRF connectivity: {args.connectivity*100:.1f}%")
    else:
        print("Using DENSE LIF→HRF connectivity (baseline)")

    def extract_features(loader, model, device):
        model.eval()
        feats, labels_all = [], []
        r_tot, r_hrf, r_lif = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, ncols=80, desc="Extracting features"):
                images = images.to(device)
                # Reshape MNIST image to a sequence of 784 1D inputs
                images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
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

    print("\n=== Extracting features ===")
    train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = extract_features(train_loader, model, device)
    print(f"✅ Extracted training feature vector size: {train_feats.shape[1]}")

    valid_feats, valid_labels, r_tot_valid, r_hrf_valid, r_lif_valid = extract_features(valid_loader, model, device)
    
    if args.use_test:
        test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test = extract_features(test_loader, model, device)
    else:
        test_feats, test_labels = valid_feats, valid_labels  # Use validation as test
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

    # ===== Theoretical SNN Energy (including sparse connectivity) =====
    T = 784  # sMNIST timesteps

    snn_energy = estimate_snn_energy_sparse(
        r_hrf=r_hrf_train,
        r_lif=r_lif_train,
        n_hid=args.n_hid,
        T=T,
        lif2hrf_connections=model.n_lif2hrf_connections,
        include_lif=True
    )

    print("\n=== Theoretical SNN Energy ===")
    print(f"Total SOPs: {snn_energy['SOPs']:.3e}")
    print(f"Energy (J): {snn_energy['Energy_J']:.3e}")

    # ===== Save Results =====
    results = {
        'args': vars(args),
        'train_acc': float(train_acc),
        'valid_acc': float(valid_acc),
        'test_acc': float(test_acc),
        'r_hrf_train': float(r_hrf_train),
        'r_lif_train': float(r_lif_train),
        'r_tot_train': float(r_tot_train),
        'energy_J': float(snn_energy['Energy_J']),
        'SOPs': float(snn_energy['SOPs']),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'connectivity': float(args.connectivity),
        'n_hid': int(args.n_hid),
        'seed': int(args.seed)
    }

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save results with descriptive filename
    conn_str = "dense" if args.connectivity == 1.0 else f"conn{args.connectivity:.1f}"
    results_filename = f"results_nhid{args.n_hid}_{conn_str}_seed{args.seed}.json"
    results_path = os.path.join(args.results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")

    print("\n=== Final Results Summary ===")
    print(f"Model: Spiking coESN with Time-Pooled Readout")
    print(f"Hidden units: {args.n_hid}")
    print(f"Connectivity: {'DENSE (100%)' if not use_sparse else f'SPARSE ({args.connectivity*100:.1f}%)'}")
    print(f"Seed: {args.seed}")
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Validation accuracy: {valid_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"HRF firing rate: {r_hrf_train:.4f}")
    print(f"Energy efficiency: {snn_energy['Energy_J']:.3e} J")
    print("=" * 70)


if __name__ == "__main__":
    main()
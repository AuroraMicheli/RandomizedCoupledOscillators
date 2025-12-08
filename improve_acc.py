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

# --- RESCALED SPIKING coESN (Reservoir only) ---
class spiking_coESN_rescaled(nn.Module):
    """
    Spiking reservoir-only version (no trainable readout).
    Batch-first input (B, L, I)
    Adds customizable LIF/HRF thresholds and feature options, including filtered spikes.
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, 
                 theta_lif, theta_rf, tau_filter, device='cpu', fading=False): # Added tau_filter
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt
        self.theta_lif = theta_lif
        self.theta_rf = theta_rf
        self.tau_filter = tau_filter # Filter time constant

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
        bias = ((torch.rand(n_hid) * 2 - 1) * 0.01 ) + self.theta_lif 
        self.bias = nn.Parameter(bias, requires_grad=False)
        
        # Spike Gain
        self.spike_gain = nn.Parameter(torch.tensor(3.0, device=device), requires_grad=False)


    # bio_cell remains the same as it outputs 's' (spikes) which the forward pass filters.
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
        # Recurrent connection is still via s (spikes), as constrained
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias
        
        # ==== LIF membrane update ====
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s = (lif_v > theta_lif).float()

        # Soft reset (subtraction by threshold)
        lif_v = lif_v - lif_s * theta_lif
        
        # ==== HRF oscillator dynamics ====
        drive = spike_gain * lif_s

        hz = hz + dt * (drive - self.gamma * hy - self.epsilon * hz)
        if self.fading:
            hz = hz - dt * hz
        hy = hy + dt * hz
        if self.fading:
            hy = hy - dt * hy

        # ==== HRF spike + refractory ====
        if ref_period is None:
            ref_period = torch.zeros_like(hz)
            
        s = (hy - theta_rf - ref_period > 0).float()
        
        hy = hy * (1 - s * alpha)
        hz = hz * (1 - s * beta)

        ref_decay = torch.exp(-torch.as_tensor(dt / tau_ref, device=device))
        ref_period = ref_period * ref_decay + s
        
        return hy, hz, s, ref_period, lif_v, lif_s

    def forward(self, x):
        B = x.size(0)
        L = x.size(1) # Sequence length
        n_hid = self.n_hid
        device = self.device
        
        hy = torch.zeros(B, n_hid, device=device)
        hz = torch.zeros(B, n_hid, device=device)
        ref_period = torch.zeros(B, n_hid, device=device)
        s = torch.zeros(B, n_hid, device=device)
        
        lif_v = torch.zeros(B, n_hid, device=device)
        spike_counts = torch.zeros(B, n_hid, device=device)
        
        # Variables for advanced features
        half_spike_counts = torch.zeros(B, n_hid, device=device)
        final_hy = None
        
        # New state variable for the filtered trace
        filtered_trace = torch.zeros(B, n_hid, device=device)
        filtered_trace_sum = torch.zeros(B, n_hid, device=device)
        decay_factor = torch.exp(-torch.as_tensor(self.dt / self.tau_filter, device=device))
        
        for t in range(L):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )
            spike_counts += s # Count HRF spikes (for mean rate)
            
            # Update Filtered Trace (Exponential Decay + New Spike)
            filtered_trace = filtered_trace * decay_factor + s
            filtered_trace_sum += filtered_trace

            # Feature Option 2: Spike Count for the first half of the sequence
            if t < L // 2:
                half_spike_counts += s 
            
            # Feature Option 4: Capture final HRF potential
            if t == L - 1:
                final_hy = hy

        # --- FEATURE OPTIONS: UNCOMMENT THE ONE YOU WANT TO USE ---
        
        # 1. Mean Firing Rate (Baseline) - Feature size: n_hid
        # mean_rates = spike_counts / L
        # return mean_rates

        # 2. Concatenated Spike Counts (First Half, Second Half) - Feature size: 2 * n_hid
        # second_half_spike_counts = spike_counts - half_spike_counts
        # half_mean_rates = half_spike_counts / (L // 2)
        # second_half_mean_rates = second_half_spike_counts / (L - L // 2)
        # return torch.cat((half_mean_rates, second_half_mean_rates), dim=1)

        # 3. Filtered Spikes Feature (Mean Trace + Final Trace) - Feature size: 2 * n_hid
        # This is your "fine-grained" feature option.
        mean_filtered_trace = filtered_trace_sum / L
        final_filtered_trace = filtered_trace
        return torch.cat((mean_filtered_trace, final_filtered_trace), dim=1)


        # 4. Concatenated Spike Rate and Final HRF Potential - Feature size: 2 * n_hid
        # mean_rates = spike_counts / L
        # return torch.cat((mean_rates, final_hy), dim=1)


########## MAIN with Rescaled Model and New Filter Parameter ##########

parser = argparse.ArgumentParser(description='Spiking coESN Option A (firing-rate readout)')
parser.add_argument('--n_hid', type=int, default=256)
parser.add_argument('--batch', type=int, default=256)  
parser.add_argument('--dt', type=float, default=0.042)
parser.add_argument('--gamma', type=float, default=2.7)
parser.add_argument('--epsilon', type=float, default=4.7)
parser.add_argument('--gamma_range', type=float, default=2.7)
parser.add_argument('--epsilon_range', type=float, default=4.7)

# RESCALED DEFAULTS: Use modest scaling and low thresholds
parser.add_argument('--inp_scaling', type=float, default=1.0)   
parser.add_argument('--theta_lif', type=float, default=0.005) # Low Threshold
parser.add_argument('--theta_rf', type=float, default=0.005)  # Low Threshold
parser.add_argument('--tau_filter', type=float, default=20.0) # New Filter Time Constant (ms equivalent)

parser.add_argument('--rho', type=float, default=0.99)
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--use_test', action="store_true")
args = parser.parse_args()

print(args)

# --- setup ---
device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
n_inp = 1
n_out = 10
bs_test = 100
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

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
    tau_filter=args.tau_filter, # Pass new parameter
    device=device
).to(device)

train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

print("\n=== Generating spike-rate features ===")

#############################
visualize_dynamics_and_spikes_middle(model, train_loader, device, n_neurons=100, n_timesteps=150)

def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, ncols=80):
            images = images.to(device)
            # Reshape MNIST image to a sequence of 784 1D inputs
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            rates = model(images)   # (B, n_feats)
            feats.append(rates.cpu())
            labels_all.append(labels)
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return feats, labels_all

# Note: The feature vector size changes based on your choice in model.forward()
train_feats, train_labels = extract_features(train_loader, model, device)
print(f"Feature vector size: {train_feats.shape[1]}") # Print feature size

valid_feats, valid_labels = extract_features(valid_loader, model, device)
if args.use_test:
    test_feats, test_labels = extract_features(test_loader, model, device)
else:
    test_feats, test_labels = None, None

# --- standardize ---
scaler = preprocessing.StandardScaler().fit(train_feats)
train_feats = scaler.transform(train_feats)
valid_feats = scaler.transform(valid_feats)
if test_feats is not None:
    test_feats = scaler.transform(test_feats)

# --- logistic regression readout ---
print("\n=== Training logistic regression readout ===")
clf = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1).fit(train_feats, train_labels)

valid_acc = clf.score(valid_feats, valid_labels) * 100
print(f"Validation accuracy: {valid_acc:.2f}%")

if test_feats is not None:
    test_acc = clf.score(test_feats, test_labels) * 100
    print(f"Test accuracy: {test_acc:.2f}%")
else:
    test_acc = 0.0

# --- unified logging ---
print(f"\n=== Done. Results ===")
print(f"Hidden units: {args.n_hid}")
print(f"Validation accuracy: {valid_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")
print("=== End of run ===")

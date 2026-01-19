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
                 theta_lif, theta_rf, tau_filter, count_lif_spikes=False, device='cpu', fading=False): # Added tau_filter
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt
        self.theta_lif = theta_lif
        self.theta_rf = theta_rf
        self.tau_filter = tau_filter # Filter time constant
        self.count_lif_spikes = count_lif_spikes

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
        #bias = ((torch.rand(n_hid) * 2 - 1) * 0.01 ) + self.theta_lif 
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling  #original one, centered in 0
        self.bias = nn.Parameter(bias, requires_grad=False)
        
        # NEW: LIF -> HRF Synaptic Weights (Simulating Tanh Bounding and Diversity)
        # Initialize weights uniformly in a bounded range, e.g., [-1.0, 1.0].
        # When a spike (1) is multiplied by this matrix, the input to the HRF
        # will effectively be bounded between -1.0 and 1.0, mimicking tanh(input)
        # where the input leads to saturation.
        lif2hrf = (torch.rand(n_hid, n_hid) * 2 - 1) * 2.0 
        
        # We can apply spectral norm scaling here to control the stability, similar to h2h,
        # ensuring the maximum possible drive doesn't cause instability.
        #lif2hrf = spectral_norm_scaling(lif2hrf, 1.0) # Scaling by 1.0 ensures max singular value is 1.0
        
        self.lif2hrf = nn.Parameter(lif2hrf, requires_grad=False)


        # Spike Gain
        self.spike_gain = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)


    # bio_cell remains the same as it outputs 's' (spikes) which the forward pass filters.
    def bio_cell(self, x, hy, hz, lif_v, s, ref_period=None):   
        dt = self.dt
        device = self.device
        theta_lif = self.theta_lif
        theta_rf = self.theta_rf
        
        # ==== LIF parameters ====
        lif_tau_m = 20.0 #20.0
        lif_tau_ref = 1e9
        spike_gain = self.spike_gain

        # ==== HRF parameters ====
        alpha = 0.0
        beta = 0.0
        tau_ref = 0.25  #to be tuned

        # ==== Input drive ====
        # Recurrent connection is still via s (spikes), as constrained
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias
        
        # ==== LIF membrane update ====
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s = (lif_v > theta_lif).float()
        #drive = lif_v
      
        # Soft reset (subtraction by threshold)
        lif_v = lif_v - lif_s * theta_lif
        
        # ==== HRF oscillator dynamics ====

        #======== EXPERIMENTS WITH DIFFERENT DRIVES===================
        #drive = (spike_gain * lif_s) #+ (spike_gain * s)
        #drive = torch.matmul(lif_s, self.lif2hrf)  #the one that works best, spikes*matrix
        #drive = (torch.tanh(lif_v))

        #lif_v_hp = lif_v - lif_v.mean(dim=1, keepdim=True)
        #drive = lif_v
        #drive = torch.tanh(input_current)


        hz = hz + dt * (drive - self.gamma * hy - self.epsilon * hz)
        if self.fading: #fading comes from original ron 
            hz = hz - dt * hz
        hy = hy + dt * hz
        if self.fading:
            hy = hy - dt * hy

        # ==== HRF spike + reset (alfa-beta) + refractory ====
        if ref_period is None:
            ref_period = torch.zeros_like(hz)
            
        s = (hy - theta_rf - ref_period > 0).float()
        
        hy = hy * (1 - s * alpha)  #special soft reset (from balanced rf)
        hz = hz * (1 - s * beta)   #special soft reset

        ref_decay = torch.exp(-torch.as_tensor(dt / tau_ref, device=device))
        ref_period = ref_period * ref_decay + s  #refractory period introduced to prevent excessive spiking
        
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

        hrf_spikes_feat = torch.zeros(B, n_hid, device=device)

        lif_v = torch.zeros(B, n_hid, device=device)
        #spike_counts = torch.zeros(B, n_hid, device=device)
        
        # Variables for advanced features
        half_spike_counts = torch.zeros(B, n_hid, device=device)
        final_hy = None
       
        total_hrf_spikes = 0.0
        total_lif_spikes = 0.0
    
        # New state variable for the filtered trace
        filtered_trace = torch.zeros(B, n_hid, device=device)
        filtered_trace_sum = torch.zeros(B, n_hid, device=device)
        decay_factor = torch.exp(-torch.as_tensor(self.dt / self.tau_filter, device=device))

        #to comppute sttistics of the different types of drive
        # --- Statistics accumulators ---
        lif_v_sum = 0.0
        lif_v_sq_sum = 0.0

        lif2hrf_sum = 0.0
        lif2hrf_sq_sum = 0.0

        stat_count = 0  # counts total elements accumulated

        
        for t in range(L):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )
            
            total_hrf_spikes += s.sum()
            total_lif_spikes += lif_s.sum()

            hrf_spikes_feat += s  #to use mean firing rate per neuron per sample as a feature
            # Update Filtered Trace (Exponential Decay + New Spike)
            filtered_trace = filtered_trace * decay_factor + s
            filtered_trace_sum += filtered_trace

            # Feature Option 2: Spike Count for the first half of the sequence
            if t < L // 2:
                half_spike_counts += s 
            
            # Feature Option 4: Capture final HRF potential
            if t == L - 1:
                final_hy = hy

            #compute statistics-diagnostic drive
            # --- lif_v statistics ---
            lif_v_sum += lif_v.sum()
            lif_v_sq_sum += (lif_v ** 2).sum()

            # --- lif_s -> HRF drive statistics ---
            lif2hrf_drive = torch.matmul(lif_s, self.lif2hrf)
            lif2hrf_sum += lif2hrf_drive.sum()
            lif2hrf_sq_sum += (lif2hrf_drive ** 2).sum()

            # total number of elements accumulated this step
            stat_count += lif_v.numel()

            

        # --- FEATURE OPTIONS: UNCOMMENT THE ONE YOU WANT TO USE ---
        
        # 1. Mean Firing Rate (Baseline) - Feature size: n_hid
        #mean_rates = total_hrf_spikes / L / B
        mean_rates = hrf_spikes_feat / L
        #return mean_rates

        # 2. Concatenated Spike Counts (First Half, Second Half) - Feature size: 2 * n_hid
        # second_half_spike_counts = spike_counts - half_spike_counts
        # half_mean_rates = half_spike_counts / (L // 2)
        # second_half_mean_rates = second_half_spike_counts / (L - L // 2)
        # return torch.cat((half_mean_rates, second_half_mean_rates), dim=1)

        # 3. Filtered Spikes Feature (Mean Trace + Final Trace) - Feature size: 2 * n_hid
        # This is your "fine-grained" feature option.
        mean_filtered_trace = filtered_trace_sum / L
        final_filtered_trace = filtered_trace
        #return torch.cat((mean_filtered_trace, final_filtered_trace), dim=1)


        # 4. Concatenated Spike Rate and Final HRF Potential - Feature size: 2 * n_hid
        #mean_rates = total_hrf_spikes / L 
       
        #return torch.cat((mean_rates, final_hy), dim=1)
        
        #r = total_spikes / (B * L * n_hid)
        
        r_hrf = total_hrf_spikes / (B * L * n_hid)
        r_lif = total_lif_spikes / (B * L * n_hid)

        if self.count_lif_spikes:
            r_total = r_hrf + r_lif
        else:
            r_total = r_hrf

        #compute statistics-diagnostic drive
        # --- Compute statistics ---
        lif_v_mean = lif_v_sum / stat_count
        lif_v_std = torch.sqrt(lif_v_sq_sum / stat_count - lif_v_mean ** 2)

        lif2hrf_mean = lif2hrf_sum / stat_count
        lif2hrf_std = torch.sqrt(lif2hrf_sq_sum / stat_count - lif2hrf_mean ** 2)

       
        features = torch.cat((mean_rates, final_hy), dim=1)
        return final_hy, {   #final_hy to return membrane potentials only
            "r_total": r_total.detach(),
            "r_hrf": r_hrf.detach(),
            "r_lif": r_lif.detach(),
            "lif_v_mean": lif_v_mean.detach(),
            "lif_v_std": lif_v_std.detach(),
            "lif2hrf_mean": lif2hrf_mean.detach(),
            "lif2hrf_std": lif2hrf_std.detach(),
        }


########## MAIN with Rescaled Model and New Filter Parameter ##########

parser = argparse.ArgumentParser(description='Spiking coESN Option A (firing-rate readout)')
parser.add_argument('--n_hid', type=int, default=256) #256 default
parser.add_argument('--batch', type=int, default=256)  
parser.add_argument('--dt', type=float, default=0.042)  #0.042 original
parser.add_argument('--gamma', type=float, default=2.7) #2.7 default
parser.add_argument('--epsilon', type=float, default=0.08) #0.51 from paper for smnist
parser.add_argument('--gamma_range', type=float, default=2)
parser.add_argument('--epsilon_range', type=float, default=1)

# RESCALED DEFAULTS: Use modest scaling and low thresholds
parser.add_argument('--inp_scaling', type=float, default=2.0)   
parser.add_argument('--theta_lif', type=float, default=5.0) # Low Threshold #default=0.05
parser.add_argument('--theta_rf', type=float, default=0.005)  # Low Threshold #standard 0.005
parser.add_argument('--tau_filter', type=float, default=20.0) # New Filter Time Constant (ms equivalent)

parser.add_argument('--rho', type=float, default=0.99)
parser.add_argument('--count_lif_spikes', action="store_true")
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
    count_lif_spikes=args.count_lif_spikes,
    device=device
).to(device)

train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

print("\n=== Generating spike-rate features ===")

#############################
visualize_dynamics_and_spikes_middle(model, train_loader, device, n_neurons=100, n_timesteps=200)

def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []

    r_tot, r_hrf, r_lif = [], [], []
    lif_v_mean_all, lif_v_std_all = [], []
    lif2hrf_mean_all, lif2hrf_std_all = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, ncols=80):
            images = images.to(device)
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)

            rates, r = model(images)

            feats.append(rates.cpu())
            labels_all.append(labels)

            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])

            lif_v_mean_all.append(r["lif_v_mean"])
            lif_v_std_all.append(r["lif_v_std"])
            lif2hrf_mean_all.append(r["lif2hrf_mean"])
            lif2hrf_std_all.append(r["lif2hrf_std"])

    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()

    return (
        feats,
        labels_all,
        torch.stack(r_tot).mean().item(),
        torch.stack(r_hrf).mean().item(),
        torch.stack(r_lif).mean().item(),
        torch.stack(lif_v_mean_all).mean().item(),
        torch.stack(lif_v_std_all).mean().item(),
        torch.stack(lif2hrf_mean_all).mean().item(),
        torch.stack(lif2hrf_std_all).mean().item(),
    )

# Note: The feature vector size changes based on your choice in model.forward()
#train_feats, train_labels, train_r = extract_features(train_loader, model, device)
(
    train_feats,
    train_labels,
    r_tot_train,
    r_hrf_train,
    r_lif_train,
    lif_v_mean_train,
    lif_v_std_train,
    lif2hrf_mean_train,
    lif2hrf_std_train,
) = extract_features(train_loader, model, device)

print(f"Feature vector size: {train_feats.shape[1]}") # Print feature size

(
    valid_feats,
    valid_labels,
    r_tot_valid,
    r_hrf_valid,
    r_lif_valid,
    lif_v_mean_valid,
    lif_v_std_valid,
    lif2hrf_mean_valid,
    lif2hrf_std_valid,
) = extract_features(valid_loader, model, device)

if args.use_test:
    test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, *_ = extract_features(test_loader, model, device)
else:
    test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test, *_ = None, None

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

print(f"Average firing rate r hrf (train): {r_hrf_train:.4f}")
print(f"Average firing rate r lif (train): {r_lif_train:.4f}")


print(
    f"LIF v: mean={lif_v_mean_train:.4e}, std={lif_v_std_train:.4e} | "
    f"LIF→HRF drive: mean={lif2hrf_mean_train:.4e}, std={lif2hrf_std_train:.4e}"
)


'''
# --- unified logging ---
print(f"\n=== Done. Results ===")
print(f"Hidden units: {args.n_hid}")
print(f"Validation accuracy: {valid_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")
print(f"Average firing rate r (train): {train_r:.4f}")
print(f"Average firing rate r (valid): {valid_r:.4f}")


print(f"HRF firing rate (train): {r_hrf_train:.4f}")
print(f"HRF firing rate (valid): {r_hrf_valid:.4f}")
print(f"LIF firing rate (train): {r_lif_train:.4f}")
print(f"LIF firing rate (valid): {r_lif_valid:.4f}")
print(f"TOTAL firing rate (train): {r_tot_train:.4f}")
print(f"TOTAL firing rate (valid): {r_tot_valid:.4f}")
'''
# ===== Theoretical SNN Energy =====
T = 784  # sMNIST timesteps

snn_energy = estimate_snn_energy(
    r_hrf=r_hrf_train,
    r_lif=r_lif_train,
    n_hid=args.n_hid,
    T=T,
    include_lif=args.count_lif_spikes
)

plot_lif_membrane_traces(
    model,
    train_loader,
    device,
    n_neurons=20,
    t_window=250,
    save_path="lif_membrane_traces_middle.png",
)

plot_hrf_membrane_traces(
    model,
    train_loader,
    device,
    n_neurons=20,
    t_window=250,
    save_path="hrf_membrane_traces_middle.png"
)

print("\n=== Theoretical SNN Energy ===")
print(f"Total SOPs: {snn_energy['SOPs']:.3e}")
print(f"Energy (J): {snn_energy['Energy_J']:.3e}")
print(f"(include LIF spikes: {args.count_lif_spikes})")
print("=== End of run ===")

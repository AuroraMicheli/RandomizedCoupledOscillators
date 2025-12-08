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



########## SPIKING coESN (Reservoir only) ##########
#spike rates per neuron are used as reservoir features to then compute regression 

class spiking_coESN(nn.Module):
    """
    Spiking reservoir-only version (no trainable readout).
    Batch-first input (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, device='cpu', fading=False):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt 

        # Parameters
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

        # Recurrent and input weights
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
        #bias = (torch.rand(n_hid) * 2 - 1) * input_scaling   #bias sampled from [-input_scaling, +input_scaling] (original)
        #bias = torch.rand(n_hid) * input_scaling  #try bias only positive, so sampled from [0, input_scaling]
        bias = ((torch.rand(n_hid) * 2 - 1) * 0.2 ) + 0.05 #bias sampled from [-0.2 + theta_lif, +0.2 + theta_lif]
        #bias = torch.rand(n_hid) * 3 - 1 #bias sampled from [-1, +2]
        self.bias = nn.Parameter(bias, requires_grad=False)

    def bio_cell(self, x, hy, hz, lif_v, s, theta_lif=0.05, theta_rf=0.05, ref_period=None):   #theta lif was 0.05
        """
        LIF-driven HRF (Harmonic Balanced Resonate-and-Fire) dynamics.
        - LIF neurons receive input current from external + recurrent drive (version 1) implements recurrent drive via potentials, version 2) implements recurrent drive via spikes)
        and produce spikes via thresholding with soft reset (subtractive).
        - Their spikes drive the HRF oscillators.
        - All additional parameters are currently neutralized for simplicity.
        """
        
        
        #used at first:
        #theta_lif=0.05, theta_rf=0.05 
        dt = self.dt
        device = self.device

        # ==== LIF parameters ====
        lif_tau_m = 20.0 #20.0       # membrane time constant (ms equivalent)
        lif_tau_ref = 1e9      # refractory time constant (huge = inactive)
        #lif_input_scale = 1.0  # input current scaling (there's already an input scaling param in the init)
        lif_ref_inh = 0.0      # no refractory inhibition
        spike_gain = 35 # LIF→HRF coupling gain

        # ==== HRF parameters ==== (tu be tuned, now random)
        alpha = 0.0 #0.3            # HRF reset rate
        beta = 0.0 #0.4             # HRF velocity damping
        tau_ref = 0.25         # HRF refractory constant

        # ==== Input drive ====
        #version 1) xternal input + HRF potentials
        
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias
        
        '''
        #version 2) xternal input + HRF spikes
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias   
        '''
        #print("Max value:", input_current.max().item()) #with input scaling = 2: max value ~ 4.3
        #print("Min value:", input_current.min().item()) #with input scaling = 2: min value ~ -2.5
        # ==== LIF membrane update ====
        # Leaky integration: dv/dt = -v / tau_m + input
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        #print("Max lif v:", lif_v.max().item())
        #print("min lif v:", lif_v.min().item())
        # Spike generation
        lif_s = (lif_v > theta_lif).float()

        # Soft reset (subtraction by threshold)
        lif_v = lif_v - lif_s * theta_lif

        # Optional refractory inhibition (currently neutralized)
        '''
        lif_ref_decay = torch.exp(-torch.as_tensor(dt / lif_tau_ref, device=device))
        lif_inhibition = lif_ref_inh * (1.0 - lif_ref_decay)
        lif_s = lif_s * (1.0 - lif_inhibition)
        '''
        
        # ==== HRF oscillator dynamics ====
        drive = spike_gain * lif_s   # + s to also have recurrent connection hrf-hrf

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
        
        hy = hy * (1 - s * alpha)  # soft reset for HRF
        hz = hz * (1 - s * beta)   # damp oscillation velocity

        ref_decay = torch.exp(-torch.as_tensor(dt / tau_ref, device=device))
        ref_period = ref_period * ref_decay + s
        
        return hy, hz, s, ref_period, lif_v, lif_s

    def forward(self, x):
        B = x.size(0)
        hy = torch.zeros(B, self.n_hid, device=self.device)
        hz = torch.zeros(B, self.n_hid, device=self.device)
        ref_period = torch.zeros(B, self.n_hid, device=self.device)
        s = torch.zeros(B, self.n_hid, device=self.device)
        theta_lif = torch.zeros(B, self.n_hid, device=self.device)

        lif_v = torch.zeros(B, self.n_hid, device=self.device)  # LIF membrane potential
        spike_counts = torch.zeros(B, self.n_hid, device=self.device)

        for t in range(x.size(1)):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )
            #spike_counts += s
            spike_counts += s  #use lif_s to check whether the hrf neurons are doing anything or not
            
        mean_rates = spike_counts / x.size(1)
        return mean_rates
        #return hy #if want to do classification based on membrane potentials



########## MAIN ##########

parser = argparse.ArgumentParser(description='Spiking coESN Option A (firing-rate readout)')
parser.add_argument('--n_hid', type=int, default=256)
parser.add_argument('--batch', type=int, default=256)  
parser.add_argument('--dt', type=float, default=0.042) #default=0.042
parser.add_argument('--gamma', type=float, default=2.7) #default=2.7
parser.add_argument('--epsilon', type=float, default=4.7) #default=4.7
parser.add_argument('--gamma_range', type=float, default=2.7) #default=2.7
parser.add_argument('--epsilon_range', type=float, default=4.7) #default=4.7
parser.add_argument('--inp_scaling', type=float, default=7.0)   #default=1.0
parser.add_argument('--rho', type=float, default=0.99)  #default 0.99
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
model = spiking_coESN(
    n_inp=n_inp,
    n_hid=args.n_hid,
    dt=args.dt,
    gamma=gamma,
    epsilon=epsilon,
    rho=args.rho,
    input_scaling=args.inp_scaling,
    device=device
).to(device)

train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

print("\n=== Generating spike-rate features ===")



#visualize_dynamics(model, train_loader, device, n_neurons=100, n_timesteps=150)
visualize_dynamics_and_spikes_middle(model, train_loader, device, n_neurons=100, n_timesteps=150)

#############################


def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, ncols=80):
            images = images.to(device)
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            rates = model(images)   # (B, n_hid)
            feats.append(rates.cpu())
            labels_all.append(labels)
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return feats, labels_all

feats, _ = extract_features(train_loader, model, device)
print(f"spiking rates before normalization: mean={feats.mean():.2f}, std={feats.std():.2f}") #spikes per neuron per time step
#to have an idea: spikes per neuron per image ~ mean*T = mean*784 ~ 23.5



train_feats, train_labels = extract_features(train_loader, model, device)
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


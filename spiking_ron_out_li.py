#!/usr/bin/env python3
import torch
from torch import nn
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from esn import spectral_norm_scaling
from utils import get_mnist_data
import argparse


###############################################
#          SPIKING coESN (Option B)           #
#     Offline logistic regression readout     #
###############################################

class spiking_coESN(nn.Module):
    """
    Spiking version of coESN with leaky integration
    producing fixed features for offline classification.
    Batch-first input shape: (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling,
                 device='cpu', fading=False):
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
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = torch.tensor(epsilon, device=device)

        # Weights
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    # -------------------------------------------------------
    # CELL: spiking coESN neuron update
    # -------------------------------------------------------
    def cell(self, x, hy, hz, ref_period, theta=0.1):
        hz = hz + self.dt * (
            torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
            - self.gamma * hy - self.epsilon * hz
        )
        if self.fading:
            hz = hz - self.dt * hz
        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy

        s = (hy - theta - ref_period > 0).float()
        #ref_period = ref_period * 0.9 + s
        return hy, hz, s, ref_period

    # -------------------------------------------------------
    # Forward pass: compute LI-filtered spike features
    # -------------------------------------------------------
    def forward_li_features(self, x, tau_mem=20.0):
        """
        Integrates spikes with an exponential decay (tau_mem)
        to produce per-sample feature vectors for logistic regression.
        Returns: (B, n_hid) = final LI-filtered activity
        """
        B, T = x.size(0), x.size(1)
        hy = torch.zeros(B, self.n_hid, device=self.device)
        hz = torch.zeros(B, self.n_hid, device=self.device)
        ref_period = torch.zeros(B, self.n_hid, device=self.device)
        u = torch.zeros(B, self.n_hid, device=self.device)

        alpha = torch.exp(-1.0 / torch.tensor(tau_mem, device=self.device))
        total_spikes = 0

        for t in range(T):
            hy, hz, s, ref_period = self.cell(x[:, t], hy, hz, ref_period)
            total_spikes += int(s.sum().item())
            u = u * alpha + s * (1 - alpha)  # leaky integration of spikes

        avg_firing_rate = total_spikes / (B * T * self.n_hid)
        return u.detach().cpu(), avg_firing_rate


###############################################
#                 MAIN SCRIPT                 #
###############################################

parser = argparse.ArgumentParser(description='Spiking coESN (Option B) — offline logistic regression')
parser.add_argument('--n_hid', type=int, default=120)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--dt', type=float, default=0.042)
parser.add_argument('--gamma', type=float, default=2.7)
parser.add_argument('--epsilon', type=float, default=4.7)
parser.add_argument('--gamma_range', type=float, default=2.7)
parser.add_argument('--epsilon_range', type=float, default=4.7)
parser.add_argument('--inp_scaling', type=float, default=1.0)
parser.add_argument('--rho', type=float, default=0.99)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--use_test', action='store_true')
args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

# --- dataset ---
train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test=100)

# --- model ---
n_inp = 1
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

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

###############################################
#         FEATURE EXTRACTION (reservoir)      #
###############################################

def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    total_rate = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, ncols=80, desc="Extracting features"):
            images = images.to(device)
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            u, rate = model.forward_li_features(images)
            feats.append(u)
            labels_all.append(labels)
            total_rate += rate
            count += 1
    avg_rate = total_rate / count
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return feats, labels_all, avg_rate

feats, _, _ = extract_features(train_loader, model, device)
print(f"membrane potentils before normalization: mean={feats.mean():.2f}, std={feats.std():.2f}")

# --- extract features ---
train_feats, train_labels, train_rate = extract_features(train_loader, model, device)
valid_feats, valid_labels, valid_rate = extract_features(valid_loader, model, device)
if args.use_test:
    test_feats, test_labels, test_rate = extract_features(test_loader, model, device)
else:
    test_feats, test_labels, test_rate = None, None, None

# --- standardize ---
scaler = preprocessing.StandardScaler().fit(train_feats)
train_feats = scaler.transform(train_feats)
valid_feats = scaler.transform(valid_feats)
if test_feats is not None:
    test_feats = scaler.transform(test_feats)

###############################################
#           OFFLINE LOGISTIC REGRESSION       #
###############################################

clf = LogisticRegression(max_iter=1000)
clf.fit(train_feats, train_labels)

valid_acc = clf.score(valid_feats, valid_labels) * 100.0
test_acc = clf.score(test_feats, test_labels) * 100.0 if args.use_test else 0.0

print(f"\nValidation accuracy: {valid_acc:.2f}%")
if args.use_test:
    print(f"Test accuracy: {test_acc:.2f}%")
print(f"Average reservoir firing rate: {train_rate*100:.2f}%")

# --- log results ---
Path("result").mkdir(parents=True, exist_ok=True)
with open("result/sMNIST_spiking_coESN_offline.txt", "a") as f:
    f.write(
        f"n_hid={args.n_hid}, Valid acc={valid_acc:.2f}, Test acc={test_acc:.2f}, "
        f"Rate={train_rate*100:.2f}%\n"
    )

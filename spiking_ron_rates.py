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

from esn import spectral_norm_scaling
from utils import get_mnist_data


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

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = torch.tensor(epsilon, device=device)

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
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz, theta=0.1, ref_period=None):
        hz = hz + self.dt * (
            torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
            - self.gamma * hy - self.epsilon * hz
        )
        if self.fading:
            hz = hz - self.dt * hz
        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy

        if ref_period is None:
            ref_period = torch.zeros_like(hz)
        s = (hy - theta - ref_period > 0).float()   #tried using hy as mem potential and better accuracy, it's probably higher so more spikes. check values
        ref_period = ref_period.mul(0.9) + s
        #to implement proper HRF neurons add smooth reset/ref period like in paper

        return hy, hz, s, ref_period

    def bio_cell(self, x, hy, hz, theta=0.1, ref_period=None):
        #Here also implemented version of soft reset and refractory period
        hz = hz + self.dt * (
            torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
            - self.gamma * hy - self.epsilon * hz
        )
        if self.fading:
            hz = hz - self.dt * hz
        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy

        if ref_period is None:
            ref_period = torch.zeros_like(hz)

        # Spike detection (refractory increases threshold)
        s = (hy - theta - ref_period > 0).float()

        # Smooth reset and refractory decay
        alpha = 0.3  # reset rate (increase for less spikes)
        beta = 0.4 # velocity damping (increase for less spikes)
        tau_ref = 0.25  # refractory time constant

        hy = hy * (1 - s * alpha)         # soft reset
        hz = hz * (1 - s * beta)          # damp velocity
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

from esn import spectral_norm_scaling
from utils import get_mnist_data


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

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = torch.tensor(epsilon, device=device)

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
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz, theta=0.1, ref_period=None):
        hz = hz + self.dt * (
            torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
            - self.gamma * hy - self.epsilon * hz
        )
        if self.fading:
            hz = hz - self.dt * hz
        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy

        if ref_period is None:
            ref_period = torch.zeros_like(hz)
        s = (hy - theta - ref_period > 0).float()   #tried using hy as mem potential and better accuracy, it's probably higher so more spikes. check values
        ref_period = ref_period.mul(0.9) + s
        #to implement proper HRF neurons add smooth reset/ref period like in paper

        return hy, hz, s, ref_period

    def bio_cell(self, x, hy, hz, theta=0.1, ref_period=None):
        #Here also implemented version of soft reset and refractory period
        hz = hz + self.dt * (
            torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
            - self.gamma * hy - self.epsilon * hz
        )
        if self.fading:
            hz = hz - self.dt * hz
        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy

        if ref_period is None:
            ref_period = torch.zeros_like(hz)

        # Spike detection (refractory increases threshold)
        s = (hy - theta - ref_period > 0).float()

        # Smooth reset and refractory decay
        alpha = 0.3  # reset rate (increase for less spikes)
        beta = 0.4 # velocity damping (increase for less spikes)
        tau_ref = 0.25  # refractory time constant

        hy = hy * (1 - s * alpha)         # soft reset
        hz = hz * (1 - s * beta)          # damp velocity
        ref_decay = torch.exp(-torch.as_tensor(self.dt / tau_ref, device=self.device))

        ref_period = ref_period * ref_decay + s

        return hy, hz, s, ref_period

    def forward(self, x):
        B = x.size(0)
        hy = torch.zeros(B, self.n_hid, device=self.device)
        hz = torch.zeros(B, self.n_hid, device=self.device)
        ref_period = torch.zeros(B, self.n_hid, device=self.device)

        spike_counts = torch.zeros(B, self.n_hid, device=self.device)

        for t in range(x.size(1)):
            hy, hz, s, ref_period = self.bio_cell(x[:, t], hy, hz, ref_period=ref_period)
            spike_counts += s

        mean_rates = spike_counts / x.size(1)  # (B, n_hid)
        return mean_rates


########## MAIN ##########

parser = argparse.ArgumentParser(description='Spiking coESN Option A (firing-rate readout)')
parser.add_argument('--n_hid', type=int, default=256)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--dt', type=float, default=0.042)
parser.add_argument('--gamma', type=float, default=2.7)
parser.add_argument('--epsilon', type=float, default=4.7)
parser.add_argument('--gamma_range', type=float, default=2.7)
parser.add_argument('--epsilon_range', type=float, default=4.7)
parser.add_argument('--inp_scaling', type=float, default=1.0)
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
print(f"spiking rates before normalization: mean={feats.mean():.2f}, std={feats.std():.2f}")


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

# --- log results ---
Path("result").mkdir(parents=True, exist_ok=True)
with open("result/sMNIST_spiking_coESN_optionA.txt", "a") as f:
    f.write(f"n_hid={args.n_hid}, Valid acc={valid_acc:.2f}, Test acc={test_acc:.2f}\n")

print("\n=== Done. Features saved and accuracies logged. ===")


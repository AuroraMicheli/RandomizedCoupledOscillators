from scipy.integrate import odeint
import numpy as np
import torch
from aeon.datasets import load_classification
import os
import torchvision
import torchvision.transforms as transforms
from torch import nn
from sklearn.model_selection import train_test_split
from esn import spectral_norm_scaling
from utils import *
import argparse


from torch import nn, optim
from tqdm import tqdm
from utils import get_mnist_data
from sklearn import preprocessing
from pathlib import Path


    ####### SPIKING RON ##########


class spiking_coESN(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, device='cpu',
                 fading=False):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = epsilon

        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        if gamma_max == gamma_min and eps_min == eps_max and gamma_max == 1:
            leaky = dt**2
            I = torch.eye(self.n_hid)
            h2h = h2h * leaky + (I * (1 - leaky))
            h2h = spectral_norm_scaling(h2h, rho)
            self.h2h = (h2h + I * (leaky - 1)) * (1 / leaky)
        else:
            h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # x2h = 2 * (2 * torch.rand(n_inp, self.n_hid) - 1) * input_scaling
        # alternative init
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz, theta: float = 0.1, ref_period=torch.Tensor):
        hz = hz + self.dt * (torch.tanh(
            torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
                             - self.gamma * hy - self.epsilon * hz)
        if self.fading:
            hz = hz - self.dt * hz

        hy = hy + self.dt * hz
        if self.fading:
            hy = hy - self.dt * hy
        
        # spiking non-linearity
        with torch.no_grad():
            s = (hz - theta - ref_period > 0).float()
        #ref_period = ref_period.mul(0.9) + s       #increase refractory period (decay). Surrogate of a soft reset

        # reset membrane potential NOW MISSING
     
        return hy, hz, s, ref_period

        #return hy, hz

    def forward(self, x):
        ## initialize hidden states
        hy = torch.zeros(x.size(0),self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0),self.n_hid).to(self.device)
        ref_period = torch.zeros(x.size(0),self.n_hid).to(self.device)

        all_states = []
        all_spikes = []
        
        for t in range(x.size(1)):
            hy, hz, s, ref_period = self.cell(x[:, t],hy,hz, ref_period=ref_period)
            all_states.append(hy)
            all_spikes.append(s)

        all_states = torch.stack(all_states, dim=1)  # (B, T, n_hid)
        all_spikes = torch.stack(all_spikes, dim=1)  # (B, T, n_hid)
        #return torch.stack(all_states, dim=1)    (batch, time, n_hidden)
        #return [hy] (batch, n_hidden), it contains only the final state
        #return torch.stack(all_states, dim=1), [hy]  # list to be compatible with ESN implementation

        return all_states, all_spikes, [hy]  # list to be compatible with ESN implementation
    

    def init_li_layer(self, n_classes: int, tau_mem: float = 20.0):
        """
        Initialize a trainable LI output layer that maps reservoir spikes to class activations.
        """
        self.n_classes = n_classes
        self.tau_mem = tau_mem

        # Linear mapping from reservoir -> output
        self.li_linear = torch.nn.Linear(
            in_features=self.n_hid,
            out_features=n_classes,
            bias=True
        ).to(self.device)

        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.li_linear.weight)
        torch.nn.init.constant_(self.li_linear.bias, 0.0)

        # Create a buffer for tau-based deca y
        tau_mem_tensor = torch.tensor(tau_mem, device=self.device)
        alpha = torch.exp(-1.0 / tau_mem_tensor)
        self.register_buffer("li_alpha", alpha)


    #LI layer linear operation fully vectorized
    def li_layer(self, spikes):
        """
        spikes: (B, T, n_hid)
        returns:
            u_all: (B, T, n_classes)
            u_final: (B, n_classes)
        """
        B, T, _ = spikes.shape

        # Vectorize linear layer
        spikes_flat = spikes.reshape(B*T, self.n_hid)            # (B*T, n_hid)
        lin_flat = self.li_linear(spikes_flat)                  # (B*T, n_classes)
        lin = lin_flat.view(B, T, self.n_classes)              # (B, T, n_classes)

        # Leaky integration (still a small Python loop)
        u = torch.zeros(B, self.n_classes, device=spikes.device)
        u_all = torch.zeros(B, T, self.n_classes, device=spikes.device)

        for t in range(T):
            u = u * self.li_alpha + lin[:, t] * (1 - self.li_alpha)
            u_all[:, t] = u

        return u_all, u

    
     # ---------- NEW FUNCTION BELOW ----------
    def forward_streaming_li(self, x):
        """
        Streaming leaky integration (Option B).
        Does not store (B,T,N) spikes, integrates output online.

        Returns:
            u_final: (B, n_classes)
            stats: dict with total_spikes and avg_firing_rate
        """
        B, T = x.size(0), x.size(1)
        hy = torch.zeros(B, self.n_hid, device=self.device)
        hz = torch.zeros(B, self.n_hid, device=self.device)
        ref_period = torch.zeros(B, self.n_hid, device=self.device)
        u = torch.zeros(B, self.n_classes, device=self.device)

        total_spikes = 0

        for t in range(T):
            hy, hz, s, ref_period = self.cell(x[:, t], hy, hz, ref_period=ref_period)
            total_spikes += int((s != 0).sum().item())

            # dense readout (can be replaced by sparse version later)
            lin = self.li_linear(s)
            u = u * self.li_alpha + lin * (1.0 - self.li_alpha)

        avg_firing_rate = total_spikes / (B * T * self.n_hid)

        stats = {
            "total_spikes": total_spikes,
            "avg_firing_rate": avg_firing_rate
        }

        return u, stats



    ############# TRAINING ON sMNIST #################

import torch
from torch import nn, optim

from utils import get_mnist_data
from sklearn import preprocessing
from pathlib import Path

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, #default=256, #original defualt is 256. decrease to increase speed
                    default=120,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, #default=120, #original defualt is 120. decrease to increase speed
                    default=20,
                    help='max epochs')
parser.add_argument('--batch', type=int, #default=120, #original defualt is 120. increase to increase speed
                    default=256,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0021,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.042,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--check', action="store_true")
parser.add_argument('--no_friction', action="store_true", help="remove friction term inside non-linearity")
parser.add_argument('--esn', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--lstm', action="store_true")
parser.add_argument('--use_test', action="store_true")


main_folder = 'result'
args = parser.parse_args()
print(args)



# --- setup ---
device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
#device = torch.device("cpu")

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

model.init_li_layer(n_classes=n_out, tau_mem=20.0)  # attach LI output layer

# Only optimize LI layer parameters
optimizer = optim.Adam(model.li_linear.parameters(), lr=args.lr)
objective = nn.CrossEntropyLoss()

train_loader, valid_loader, test_loader = get_mnist_data(args.batch, bs_test)

# --- training ---

# --- training ---
dense_MACs = None  # we'll compute this once based on model dims

for epoch in range(args.epochs):
    print(f"\nEpoch {epoch + 1}/{args.epochs}")
    model.train()
    total_loss = 0.0
    total_spikes_epoch = 0
    total_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        # reshape to (B, T, 1)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)

        optimizer.zero_grad()

        # forward with streaming LI integration
        u_final, stats = model.forward_streaming_li(images)
        loss = objective(u_final, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_spikes_epoch += stats["total_spikes"]
        total_batches += 1

        avg_loss = total_loss / total_batches
        avg_rate = stats["avg_firing_rate"]

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "rate": f"{avg_rate*100:.2f}%"
        })

    # ---- epoch summary ----
    avg_loss = total_loss / len(train_loader)
    avg_rate_epoch = total_spikes_epoch / (len(train_loader.dataset) * model.n_hid)
    print(f"Train loss: {avg_loss:.4f} | Avg firing rate: {avg_rate_epoch*100:.2f}%")

    # --- validation ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.reshape(bs_test, 1, 784).permute(0, 2, 1)

            u_final, _ = model.forward_streaming_li(images)
            preds = u_final.argmax(dim=1)
            correct += (preds == labels).sum().item()

    valid_acc = 100.0 * correct / len(valid_loader.dataset)
    print(f"Validation accuracy: {valid_acc:.2f}%")

    # --- MACs estimation ---
    if dense_MACs is None:
        B = args.batch
        T = 784
        N_hid = model.n_hid
        N_cls = model.n_classes
        dense_MACs = T * B * N_hid * N_cls  # baseline dense cost
    sparse_MACs = total_spikes_epoch * model.n_classes
    mac_reduction = (1 - sparse_MACs / dense_MACs) * 100

    print(f"Estimated MACs reduction: {mac_reduction:.2f}%")

    # --- Save logs ---
    Path(main_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{main_folder}/sMNIST_log_spiking_coESN.txt', 'a') as f:
        f.write(
            f"Epoch {epoch + 1}, "
            f"Loss: {avg_loss:.4f}, "
            f"Valid acc: {valid_acc:.2f}, "
            f"Avg rate: {avg_rate_epoch*100:.2f}%, "
            f"MAC reduction: {mac_reduction:.2f}%\n"
        )


'''
for epoch in range(args.epochs):
    print(f"\nEpoch {epoch}")
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)

    for images, labels in progress_bar: #tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        # reshape to (B, T, 1)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)

        optimizer.zero_grad()

        # reservoir forward
      
        states, spikes, _ = model(images)  #states nd spikes [batch, time, hidden units]
        

        #print("states:", states[0, :, 50])
        num_spikes = (spikes != 0).sum().item()
        print(f"Number of spikes: {num_spikes}")

        # LI output layer forward
        #u_all, u_final = model.li_layer(spikes)
        u_final, stats = model.forward_streaming_li(images) #function for online

        # loss on final membrane potential (logits)
        loss = objective(u_final, labels)
        loss.backward()
        optimizer.step()

        total_spikes = stats["total_spikes"]
        avg_rate = stats["avg_firing_rate"]

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_loss:.4f}")

    # --- validation ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.reshape(bs_test, 1, 784).permute(0, 2, 1)

            _, spikes, _ = model(images)
            _, u_final = model.li_layer(spikes)
            preds = u_final.argmax(dim=1)
            correct += (preds == labels).sum().item()

    valid_acc = 100.0 * correct / len(valid_loader.dataset)
    print(f"Validation accuracy: {valid_acc:.2f}%")

    # Save logs
    Path(main_folder).mkdir(parents=True, exist_ok=True)
    f = open(f'{main_folder}/sMNIST_log_spiking_coESN.txt', 'a')
    f.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Valid acc: {valid_acc:.2f}\n")
    f.close()

    if (epoch + 1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
'''
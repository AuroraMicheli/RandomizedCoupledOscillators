import torch
import torch.nn as nn
from esn import spectral_norm_scaling


class spiking_coESN(nn.Module):
    """
    Minimal spiking coESN model for Optuna optimisation.
    Only the parameters tuned by Optuna are external.
    All other parameters are fixed internally.
    """

    def __init__(self,
                 n_inp,
                 n_hid,
                 dt,
                 gamma,
                 epsilon,
                 rho,
                 input_scaling,
                 lif_tau_m,
                 spike_gain,
                 theta_lif,
                 theta_rf,
                 device='cpu'):

        super().__init__()
        self.n_hid = n_hid
        self.device = device

        # === OPTUNA-TUNED PARAMETERS ===
        self.dt = dt
        self.gamma = torch.tensor(gamma, device=device)
        self.epsilon = torch.tensor(epsilon, device=device)
        self.rho = rho
        self.theta_lif = theta_lif
        self.theta_rf = theta_rf
        self.lif_tau_m = lif_tau_m
        self.spike_gain = spike_gain

        # === FIXED PARAMETERS (NOT tuned by Optuna) ===
        self.fading = False
        self.alpha = 0.0      # HRF reset rate
        self.beta = 0.0       # HRF velocity damping
        self.tau_ref = 0.25   # HRF refractory constant

        # === Input weights ===
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)

        # === Recurrent weights ===
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # === Bias ===
        bias = (torch.rand(n_hid) * 2 - 1) * 0.2 + 0.05
        self.bias = nn.Parameter(bias, requires_grad=False)

    # ----------------------------------------------------------------------
    #                      Single-Timestep Dynamics
    # ----------------------------------------------------------------------

    def bio_cell(self, x, hy, hz, lif_v, s, ref_period):

        dt = self.dt
        device = self.device

        # LIF update
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias

        lif_v = lif_v + dt * (-lif_v / self.lif_tau_m + input_current)

        lif_s = (lif_v > self.theta_lif).float()
        lif_v = lif_v - lif_s * self.theta_lif

        # HRF update
        drive = self.spike_gain * lif_s

        hz = hz + dt * (drive - self.gamma * hy - self.epsilon * hz)
        hy = hy + dt * hz

        s = (hy - self.theta_rf - ref_period > 0).float()

        hy = hy * (1 - s * self.alpha)
        hz = hz * (1 - s * self.beta)

        ref_decay = torch.exp(-torch.as_tensor(dt / self.tau_ref, device=device))
        ref_period = ref_period * ref_decay + s

        return hy, hz, s, ref_period, lif_v, lif_s

    # ----------------------------------------------------------------------
    #                             FORWARD PASS
    # ----------------------------------------------------------------------

    def forward(self, x):
        """
        x: shape (B, T, input_dim)
        returns: mean spike rate (B, n_hid)
        """

        B = x.size(0)
        n_hid = self.n_hid

        hy = torch.zeros(B, n_hid, device=self.device)
        hz = torch.zeros(B, n_hid, device=self.device)
        ref_period = torch.zeros(B, n_hid, device=self.device)
        s = torch.zeros(B, n_hid, device=self.device)
        lif_v = torch.zeros(B, n_hid, device=self.device)

        spike_counts = torch.zeros(B, n_hid, device=self.device)

        for t in range(x.size(1)):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period
            )
            spike_counts += s

        return spike_counts / x.size(1)

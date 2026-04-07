import matplotlib.pyplot as plt
import numpy as np
import random
import torch

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse
from pathlib import Path
from esn import spectral_norm_scaling

class spiking_LIF_reservoir(nn.Module):
    """
    LIF-only ablation of spiking_coESN_rescaled_II.

    Architecture (identical topology to s-RON):
      - Layer 1: encoder LIF neurons  (fixed, identical to s-RON)
                 receive external input x, emit spikes lif_s
      - Layer 2: reservoir LIF neurons (replaces HRF in s-RON)
                 receive encoder spikes via lif2res (was lif2hrf)
                 recurrent connections via h2h (was hrf->hrf)
                 emit spikes res_s

    Heterogeneous reservoir LIF parameters (per-neuron, drawn from distributions):
      - tau_m   : membrane time constant, log-uniform in
                  (tau_m   - tau_m_range/2,   tau_m   + tau_m_range/2)
      - theta_res: firing threshold, log-uniform in
                  (theta_res - theta_res_range/2, theta_res + theta_res_range/2)

    All weight matrices, sparsity, readout modes, and energy tracking are
    identical to spiking_coESN_rescaled_II so results are directly comparable.

    Naming note:
      - lif2hrf  -> lif2res  (encoder -> reservoir)
      - hrf2lif  -> res2enc  (reservoir -> encoder, controls h2h sparsity)
      - theta_rf -> theta_res (reservoir threshold, now heterogeneous)
      - gamma/epsilon -> tau_m/theta_res (reservoir neuron params)
    """

    def __init__(self, n_inp, n_hid, dt, tau_m, tau_m_range,
                 theta_res, theta_res_range,
                 rho, input_scaling,
                 theta_lif,           # encoder LIF threshold (fixed scalar)
                 tau_filter,
                 count_lif_spikes=False,
                 sparse_lif2res=True, connectivity_lif2res=0.1,
                 sparse_res2enc=True, connectivity_res2enc=0.1,
                 device='cpu', fading=False,
                 readout_mode='final'):
        super().__init__()

        self.n_hid      = n_hid
        self.device     = device
        self.fading     = fading
        self.dt         = dt
        self.theta_lif  = theta_lif   # encoder threshold — fixed scalar
        self.tau_filter = tau_filter
        self.count_lif_spikes  = count_lif_spikes
        self.sparse_lif2res    = sparse_lif2res
        self.connectivity_lif2res = connectivity_lif2res
        self.sparse_res2enc    = sparse_res2enc
        self.connectivity_res2enc = connectivity_res2enc

        _valid_modes = ("final", "mean", "rms_std_final")
        assert readout_mode in _valid_modes, \
            f"readout_mode must be one of {_valid_modes}, got '{readout_mode}'"
        self.readout_mode = readout_mode

        # ── Heterogeneous reservoir LIF parameters ────────────────────────────
        # tau_m: log-uniform sampling so equal density across scales
        # Clamp to strictly positive values (analogous to gamma/epsilon clamp)
        tau_lo  = max(tau_m  - tau_m_range  / 2., 1e-3)
        tau_hi  = tau_m  + tau_m_range  / 2.
        tres_lo = max(theta_res - theta_res_range / 2., 1e-6)
        tres_hi = theta_res + theta_res_range / 2.

        # Log-uniform: sample uniformly in log space then exponentiate
        self.tau_m_vec    = torch.exp(
            torch.FloatTensor(n_hid).uniform_(np.log(tau_lo), np.log(tau_hi))
        ).to(device)
        self.theta_res_vec = torch.exp(
            torch.FloatTensor(n_hid).uniform_(np.log(tres_lo), np.log(tres_hi))
        ).to(device)

        # ── Recurrent weight matrix h2h (reservoir LIF -> reservoir LIF) ─────
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
        h2h = spectral_norm_scaling(h2h, rho)

        if sparse_res2enc:
            h2h  = h2h.to(device)
            mask = (torch.rand(n_hid, n_hid, device=device) < connectivity_res2enc).float()
            h2h  = h2h * mask
            self.n_res2enc_connections = int(mask.sum().item())
            print(f"Res->Enc sparse recurrent: "
                  f"{self.n_res2enc_connections}/{n_hid**2} "
                  f"({connectivity_res2enc*100:.1f}%)")
        else:
            h2h = h2h.to(device)
            self.n_res2enc_connections = n_hid ** 2
            print(f"Res->Enc dense recurrent: {n_hid**2}/{n_hid**2} (100%)")

        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # ── Input weights (always dense, same as s-RON) ───────────────────────
        x2h  = torch.rand(n_inp, n_hid) * input_scaling
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.x2h  = nn.Parameter(x2h,  requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

        # ── Encoder -> Reservoir weights (lif2res, potentially sparse) ────────
        if sparse_lif2res:
            lif2res_full = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            mask_l2r     = (torch.rand(n_hid, n_hid, device=device) < connectivity_lif2res).float()
            lif2res      = lif2res_full * mask_l2r
            self.n_lif2res_connections = int(mask_l2r.sum().item())
            print(f"Enc->Res sparse: "
                  f"{self.n_lif2res_connections}/{n_hid**2} "
                  f"({connectivity_lif2res*100:.1f}%)")
        else:
            lif2res = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            self.n_lif2res_connections = n_hid ** 2
            print(f"Enc->Res dense: {n_hid**2}/{n_hid**2} (100%)")

        self.lif2res = nn.Parameter(lif2res, requires_grad=False)

        # kept for API compatibility with energy estimator
        self.n_lif2hrf_connections = self.n_lif2res_connections
        self.n_hrf2lif_connections = self.n_res2enc_connections

        self.spike_gain = nn.Parameter(torch.tensor(1.0, device=device),
                                       requires_grad=False)

    # ── Single timestep update ────────────────────────────────────────────────
    def bio_cell(self, x, res_v, res_s, lif_v, ref_period=None):
        """
        x       : (B, n_inp)   — input at this timestep
        res_v   : (B, n_hid)   — reservoir LIF membrane voltage
        res_s   : (B, n_hid)   — reservoir LIF spikes (previous step)
        lif_v   : (B, n_hid)   — encoder LIF membrane voltage
        ref_period: (B, n_hid) — refractory period tracker (unused for LIF,
                                  kept for signature compatibility)
        """
        dt        = self.dt
        device    = self.device
        theta_lif = self.theta_lif   # encoder: fixed scalar

        # ── Encoder LIF (identical to s-RON) ─────────────────────────────────
        lif_tau_m    = 20.0           # encoder time constant fixed
        input_current = (torch.matmul(x, self.x2h)
                         + torch.matmul(res_s, self.h2h)
                         + self.bias)

        lif_v  = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s  = (lif_v > theta_lif).float()
        lif_v  = lif_v - lif_s * theta_lif   # soft reset

        # ── Reservoir LIF (heterogeneous tau_m and theta_res) ─────────────────
        # drive from encoder spikes through lif2res
        drive  = torch.matmul(lif_s, self.lif2res)

        # per-neuron leak: -v / tau_m_vec
        res_v  = res_v + dt * (-res_v / self.tau_m_vec + drive)

        # per-neuron threshold
        res_s  = (res_v > self.theta_res_vec).float()
        res_v  = res_v - res_s * self.theta_res_vec   # soft reset

        if self.fading:
            res_v = res_v - dt * res_v

        return res_v, res_s, lif_v, lif_s

    # ── Full forward pass ─────────────────────────────────────────────────────
    def forward(self, x):
        """
        Output shape matches spiking_coESN_rescaled_II:
          "final"         -> (B, n_hid)
          "mean"          -> (B, n_hid)
          "rms_std_final" -> (B, 3*n_hid)
        """
        B, L, _ = x.shape
        n_hid   = self.n_hid
        device  = self.device

        res_v      = torch.zeros(B, n_hid, device=device)
        res_s      = torch.zeros(B, n_hid, device=device)
        lif_v      = torch.zeros(B, n_hid, device=device)

        need_stats = self.readout_mode in ("mean", "rms_std_final")
        if need_stats:
            res_sum    = torch.zeros(B, n_hid, device=device)
            res_sq_sum = torch.zeros(B, n_hid, device=device)

        total_res_spikes = 0.0
        total_lif_spikes = 0.0

        for t in range(L):
            res_v, res_s, lif_v, lif_s = self.bio_cell(
                x[:, t], res_v, res_s, lif_v
            )
            if need_stats:
                res_sum    += res_v
                res_sq_sum += res_v ** 2

            total_res_spikes += res_s.sum()
            total_lif_spikes += lif_s.sum()

        # ── Readout features (res_v plays the role of hy in s-RON) ───────────
        if self.readout_mode == "final":
            features = res_v

        elif self.readout_mode == "mean":
            features = res_sum / L

        elif self.readout_mode == "rms_std_final":
            res_mean = res_sum / L
            res_rms  = torch.sqrt(res_sq_sum / L + 1e-8)
            res_std  = torch.sqrt(torch.clamp(
                res_sq_sum / L - res_mean ** 2, min=1e-8))
            features = torch.cat([res_rms, res_std, res_v], dim=1)

        # ── Firing rates (compatible with energy estimator API) ───────────────
        r_res   = total_res_spikes / (B * L * n_hid)
        r_lif   = total_lif_spikes / (B * L * n_hid)
        r_total = (r_res + r_lif) if self.count_lif_spikes else r_res

        return features, {
            "r_total": r_total.detach(),
            "r_hrf":   r_res.detach(),    # kept as r_hrf for API compatibility
            "r_lif":   r_lif.detach(),
        }



        
class spiking_coESN_rescaled_II(nn.Module):
    """
    Spiking reservoir-only version (no trainable readout).
    Batch-first input (B, L, I)
    Adds customizable LIF/HRF thresholds and feature options, including filtered spikes.
    
    READOUT STRATEGY: controlled by readout_mode argument
    -------------------------------------------------------
    "final"         - hy at the last time step only          -> (B, n_hid)
    "mean"          - temporal mean of hy                    -> (B, n_hid)
    "rms_std_final" - concatenation of RMS, Std, Final state -> (B, 3*n_hid)
                        . RMS   captures oscillation amplitude (energy)
                        . Std   captures temporal variability (dynamics)
                        . Final captures endpoint phase

    Default: "final"

    ENERGY OPTIMIZATION: Sparse connectivity
    - Sparse LIF->HRF connectivity (reduces LIF-driven synaptic operations)
    - Sparse HRF->HRF recurrent connectivity (reduces recurrent operations)
    - Both reduce synaptic operations while maintaining representational capacity
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling,
                 theta_lif, theta_rf, tau_filter, count_lif_spikes=False,
                 sparse_lif2hrf=True, connectivity_lif2hrf=0.1,
                 sparse_hrf2lif=True, connectivity_hrf2lif=0.1,
                 device='cpu', fading=False,
                 readout_mode='final'):          # NEW: controls what forward() returns
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
        self.sparse_hrf2lif = sparse_hrf2lif
        self.connectivity_hrf2lif = connectivity_hrf2lif

        # NEW: validate and store readout mode
        _valid_modes = ("final", "mean", "rms_std_final")
        assert readout_mode in _valid_modes, \
            f"readout_mode must be one of {_valid_modes}, got '{readout_mode}'"
        self.readout_mode = readout_mode

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

        # ===== HRF->HRF Recurrent Weights (POTENTIALLY SPARSE) =====
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)

        if gamma_min == gamma_max and eps_min == eps_max and gamma_max == 1:
            leaky = dt**2
            I = torch.eye(n_hid)
            h2h = h2h * leaky + (I * (1 - leaky))
            h2h = spectral_norm_scaling(h2h, rho)
            h2h = (h2h + I * (leaky - 1)) * (1 / leaky)
        else:
            h2h = spectral_norm_scaling(h2h, rho)

        if sparse_hrf2lif:
            h2h = h2h.to(device)
            mask_hrf2lif = (torch.rand(n_hid, n_hid, device=device) < connectivity_hrf2lif).float()
            h2h = h2h * mask_hrf2lif
            n_connections_hrf2lif = mask_hrf2lif.sum().item()
            self.n_hrf2lif_connections = n_connections_hrf2lif
            print(f"HRF->LIF sparse recurrent connectivity: {n_connections_hrf2lif}/{n_hid**2} connections ({connectivity_hrf2lif*100:.1f}%)")
        else:
            h2h = h2h.to(device)
            self.n_hrf2lif_connections = n_hid ** 2
            print(f"HRF->LIF dense recurrent connectivity: {n_hid**2}/{n_hid**2} connections (100%)")

        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # Input weights (always dense)
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)

        # Rescaled bias
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

        # ===== LIF->HRF Synaptic Weights (POTENTIALLY SPARSE) =====
        if sparse_lif2hrf:
            lif2hrf_full = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            mask_lif2hrf = (torch.rand(n_hid, n_hid, device=device) < connectivity_lif2hrf).float()
            lif2hrf = lif2hrf_full * mask_lif2hrf
            n_connections_lif2hrf = mask_lif2hrf.sum().item()
            self.n_lif2hrf_connections = n_connections_lif2hrf
            print(f"LIF->HRF sparse connectivity: {n_connections_lif2hrf}/{n_hid**2} connections ({connectivity_lif2hrf*100:.1f}%)")
        else:
            lif2hrf = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
            self.n_lif2hrf_connections = n_hid ** 2
            print(f"LIF->HRF dense connectivity: {n_hid**2}/{n_hid**2} connections (100%)")

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

        # ==== Input drive (includes sparse HRF->HRF recurrence) ====
        input_current = torch.matmul(x, self.x2h) + torch.matmul(s, self.h2h) + self.bias

        # ==== LIF membrane update ====
        lif_v = lif_v + dt * (-lif_v / lif_tau_m + input_current)
        lif_s = (lif_v > theta_lif).float()
        lif_v = lif_v - lif_s * theta_lif

        # ==== HRF oscillator dynamics (with sparse LIF->HRF coupling) ====
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
        Forward pass. Output feature size depends on readout_mode:
          "final"         -> (B, n_hid)    hy at final time step
          "mean"          -> (B, n_hid)    temporal mean of hy
          "rms_std_final" -> (B, 3*n_hid)  [RMS | Std | Final]
        """
        B = x.size(0)
        L = x.size(1)
        n_hid = self.n_hid
        device = self.device

        # Initialize states
        hy        = torch.zeros(B, n_hid, device=device)
        hz        = torch.zeros(B, n_hid, device=device)
        ref_period= torch.zeros(B, n_hid, device=device)
        s         = torch.zeros(B, n_hid, device=device)
        lif_v     = torch.zeros(B, n_hid, device=device)

        # NEW: only allocate accumulators when the readout mode actually needs them
        need_stats = self.readout_mode in ("mean", "rms_std_final")
        if need_stats:
            hy_sum    = torch.zeros(B, n_hid, device=device)
            hy_sq_sum = torch.zeros(B, n_hid, device=device)

        # Spike counting for energy analysis
        total_hrf_spikes = 0.0
        total_lif_spikes = 0.0

        for t in range(L):
            hy, hz, s, ref_period, lif_v, lif_s = self.bio_cell(
                x[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )

            # NEW: only accumulate when needed
            if need_stats:
                hy_sum    += hy
                hy_sq_sum += hy ** 2

            total_hrf_spikes += s.sum()
            total_lif_spikes += lif_s.sum()

        # NEW: build features according to readout_mode
        if self.readout_mode == "final":
            features = hy                                            # (B, n_hid)

        elif self.readout_mode == "mean":
            features = hy_sum / L                                    # (B, n_hid)

        elif self.readout_mode == "rms_std_final":
            hy_mean  = hy_sum / L
            hy_rms   = torch.sqrt(hy_sq_sum / L + 1e-8)
            hy_std   = torch.sqrt(torch.clamp(hy_sq_sum / L - hy_mean ** 2, min=1e-8))
            features = torch.cat([hy_rms, hy_std, hy], dim=1)       # (B, 3*n_hid)

        # Compute average firing rates for energy analysis
        r_hrf   = total_hrf_spikes / (B * L * n_hid)
        r_lif   = total_lif_spikes / (B * L * n_hid)
        r_total = (r_hrf + r_lif) if self.count_lif_spikes else r_hrf

        return features, {
            "r_total": r_total.detach(),
            "r_hrf":   r_hrf.detach(),
            "r_lif":   r_lif.detach()
        }

        
'''
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
            lif2hrf = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
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
        
        features = torch.cat([
            hy_rms,    # RMS captures oscillation amplitude (always positive, informative)
            hy_std,    # Std captures dynamics/variability
            hy_final   # Final state captures endpoint phase
        ], dim=1)
        
        #features = hy_final
        #features = hy_mean
        # Compute average firing rates for energy analysis
        r_hrf = total_hrf_spikes / (B * L * n_hid)
        r_lif = total_lif_spikes / (B * L * n_hid)
        r_total = (r_hrf + r_lif) if self.count_lif_spikes else r_hrf

        return features, {
            "r_total": r_total.detach(),
            "r_hrf": r_hrf.detach(),
            "r_lif": r_lif.detach()
        }



'''
        
# --- RESCALED SPIKING coESN (Reservoir only) ---
class spiking_coESN_rescaled_I(nn.Module):
    """
    Spiking reservoir-only version (no trainable readout).
    Batch-first input (B, L, I)
    Adds customizable LIF/HRF thresholds and feature options, including filtered spikes.
    
    READOUT STRATEGY: Time-Pooled Statistics (RMS + Std + Final State)
    - RMS captures oscillation amplitude (energy)
    - Std captures temporal variability (dynamics)
    - Final state captures endpoint phase
    - Provides n_hid features capturing temporal dynamics efficiently (option to extend to 3*n_hid features)
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
        features = hy_final

        # Compute average firing rates for energy analysis
        r_hrf = total_hrf_spikes / (B * L * n_hid)
        r_lif = total_lif_spikes / (B * L * n_hid)
        r_total = (r_hrf + r_lif) if self.count_lif_spikes else r_hrf

        return features, {
            "r_total": r_total.detach(),
            "r_hrf": r_hrf.detach(),
            "r_lif": r_lif.detach()
        }





#=================================== PLOTTING FUNCTIONS ========================================
def plot_hrf_membrane_traces(
    model,
    loader,
    device,
    n_neurons=30,
    t_window=200,
    save_path="hrf_membrane_traces_middle.png",
):
    model.eval()

    # Take one batch, one sample
    images, _ = next(iter(loader))
    images = images.to(device)
    images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)

    B, T, _ = images.shape
    mid = T // 2
    t0 = mid - t_window // 2
    t1 = mid + t_window // 2

    # States
    hy = torch.zeros(B, model.n_hid, device=device)
    hz = torch.zeros_like(hy)
    ref = torch.zeros_like(hy)
    s = torch.zeros_like(hy)
    lif_v = torch.zeros_like(hy)

    # Sample neurons
    idx = torch.randperm(model.n_hid)[:n_neurons]

    traces = []

    with torch.no_grad():
        for t in range(T):
            hy, hz, s, ref, lif_v, lif_s = model.bio_cell(
                images[:, t], hy, hz, lif_v, s, ref
            )

            if t0 <= t < t1:
                traces.append(hy[0, idx].cpu().numpy())  # <- HRF membrane potentials

    traces = np.stack(traces)  # (time, neurons)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))

    for i in range(traces.shape[1]):
        plt.plot(traces[:, i], lw=1, label=f"n{i}")

    plt.axhline(model.theta_rf, color="k", linestyle="--", alpha=0.5, label="θ_rf")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)

    plt.title("HRF membrane potentials (raw, middle time window)")
    plt.xlabel("Time step")
    plt.ylabel("Membrane potential")
    plt.tight_layout()

    # Optional: legend only if few neurons
    if n_neurons <= 10:
        plt.legend(loc="best", fontsize=8)

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved HRF membrane plot to: {save_path}")



def plot_lif_membrane_traces(
    model,
    loader,
    device,
    n_neurons=30,
    t_window=200,
    save_path="lif_membrane_traces_middle.png",
):
    model.eval()

    # Take one batch, one sample
    images, _ = next(iter(loader))
    images = images.to(device)
    images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)

    B, T, _ = images.shape
    mid = T // 2
    t0 = mid - t_window // 2
    t1 = mid + t_window // 2

    # States
    hy = torch.zeros(B, model.n_hid, device=device)
    hz = torch.zeros_like(hy)
    ref = torch.zeros_like(hy)
    s = torch.zeros_like(hy)
    lif_v = torch.zeros_like(hy)

    # Sample neurons
    idx = torch.randperm(model.n_hid)[:n_neurons]

    traces = []

    with torch.no_grad():
        for t in range(T):
            hy, hz, s, ref, lif_v, lif_s = model.bio_cell(
                images[:, t], hy, hz, lif_v, s, ref
            )

            if t0 <= t < t1:
                traces.append(lif_v[0, idx].cpu().numpy())

    traces = np.stack(traces)  # (time, neurons)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))

    for i in range(traces.shape[1]):
        plt.plot(traces[:, i], lw=1, label=f"n{i}")

    plt.axhline(model.theta_lif, color="k", linestyle="--", alpha=0.5, label="θ_lif")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)

    plt.title("LIF membrane potentials (raw, middle time window)")
    plt.xlabel("Time step")
    plt.ylabel("Membrane potential")
    plt.tight_layout()

    # Optional: legend only if few neurons
    if n_neurons <= 10:
        plt.legend(loc="best", fontsize=8)

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved LIF membrane plot to: {save_path}")


#=================================== ESTIMATE ENERGY FUNCTIONS ========================================

def estimate_ann_energy(n_inp, n_hid, T):
    """
    Theoretical energy for non-spiking coESN
    Following Appendix B (MAC-based energy)
    """
    E_MAC = 4.6e-12  # Joules per MAC (from paper)

    # x2h + h2h
    macs_per_timestep = (
        n_inp * n_hid +        # input to hidden
        n_hid * n_hid          # recurrent
    )
    '''
    #option without taking into account x2h:
    macs_per_timestep = n_hid * n_hid          # recurrent
    
    '''
    total_macs = T * macs_per_timestep
    energy = total_macs * E_MAC


    return {
        "MACs": total_macs,
        "Energy_J": energy
    }


def estimate_snn_energy_sparse(
    r_hrf,
    r_lif,
    n_hid,
    T,
    lif2hrf_connections,
    include_lif=True,
    E_SOP=0.9e-12
):
    """
    Energy estimator compatible with sparse LIF→HRF connectivity
    """

    # --- HRF spikes ---
    hrf_spikes = r_hrf * n_hid * T
    hrf_sops = hrf_spikes * n_hid  # dense HRF→LIF

    total_sops = hrf_sops

    # --- LIF spikes ---
    if include_lif:
        lif_spikes = r_lif * n_hid * T

        # average fanout per LIF neuron
        lif_fanout = lif2hrf_connections / n_hid

        lif_sops = lif_spikes * lif_fanout
        total_sops += lif_sops
    else:
        lif_sops = 0.0

    energy = total_sops * E_SOP

    return {
        "SOPs": total_sops,
        "Energy_J": energy,
        "HRF_SOPs": hrf_sops,
        "LIF_SOPs": lif_sops
    }



    
def estimate_snn_energy(
    r_hrf,
    r_lif,
    n_hid,
    T,
    include_lif=True
):
    """
    Theoretical SNN energy (Appendix B style)
    r_* are average firing rates
    """
    E_SOP = 0.9e-12  # Joules per SOP

    if include_lif:
        r_total = r_hrf + r_lif
    else:
        r_total = r_hrf

    # total spikes per sample
    total_spikes = r_total * n_hid * T

    # each spike triggers n_hid synaptic ops
    total_sops = total_spikes * n_hid

    energy = total_sops * E_SOP

    return {
        "SOPs": total_sops,
        "Energy_J": energy
    }

#=================================== VISUALIZE DYNAMICS AND SPIKES FUNCTIONS ========================================

def visualize_dynamics_and_spikes(
    model, loader, device, n_neurons=100, n_timesteps=150, save_prefix="spiking_coesn"
):
    """
    Visualizes both membrane potentials and spike rasters for LIF and HRF neurons
    over the last n_timesteps for a random subset of neurons.
    Generates 4 figures:
      - LIF membrane potentials (heatmap)
      - HRF potentials (heatmap)
      - LIF spike raster
      - HRF spike raster
    """
    model.eval()
    with torch.no_grad():
        # --- Get one batch ---
        images, labels = next(iter(loader))
        images = images.to(device)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
        B, T, _ = images.shape

        # --- Initialize states ---
        hy = torch.zeros(B, model.n_hid, device=device)
        hz = torch.zeros(B, model.n_hid, device=device)
        ref_period = torch.zeros(B, model.n_hid, device=device)
        s = torch.zeros(B, model.n_hid, device=device)
        lif_v = torch.zeros(B, model.n_hid, device=device)

        # --- Recordings ---
        lif_vs, hrf_ys = [], []
        lif_spikes, hrf_spikes = [], []

        # --- Run simulation ---
        for t in range(T):
            hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
                images[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )
            if t >= T - n_timesteps:
                lif_vs.append(lif_v[0].detach().cpu().numpy())
                hrf_ys.append(hy[0].detach().cpu().numpy())
                lif_spikes.append(lif_s[0].detach().cpu().numpy())
                hrf_spikes.append(s[0].detach().cpu().numpy())

        # --- Convert to arrays ---
        lif_vs = np.stack(lif_vs, axis=0)   # (time, neurons)
        hrf_ys = np.stack(hrf_ys, axis=0)
        lif_spikes = np.stack(lif_spikes, axis=0)
        hrf_spikes = np.stack(hrf_spikes, axis=0)

        # --- Select subset of neurons ---
        n_total = lif_vs.shape[1]
        sel_idx = random.sample(range(n_total), min(n_neurons, n_total))

        lif_vs = lif_vs[:, sel_idx].T
        hrf_ys = hrf_ys[:, sel_idx].T
        lif_spikes = lif_spikes[:, sel_idx].T
        hrf_spikes = hrf_spikes[:, sel_idx].T

        # ==============================================================
        # 1) LIF membrane potentials (comparison normalized and not)
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(lif_vs, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("LIF membrane potentials (raw)")
        axes[0].set_xlabel("Time step (last 150)")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow((lif_vs - lif_vs.mean(axis=1, keepdims=True)) / (lif_vs.std(axis=1, keepdims=True) + 1e-9), aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2)
        axes[1].set_title("LIF membrane potentials (z-scored)")
        axes[1].set_xlabel("Time step (last 150)")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("LIF neuron activity comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 2) HRF potentials (comparison normalized and not)
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(hrf_ys, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("HRF membrane potentials (raw)")
        axes[0].set_xlabel("Time step (last 150)")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow((hrf_ys - hrf_ys.mean(axis=1, keepdims=True)) / (hrf_ys.std(axis=1, keepdims=True) + 1e-9), aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2)
        axes[1].set_title("HRF membrane potentials (z-scored)")
        axes[1].set_xlabel("Time step (last 150)")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("HRF neuron activity comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 3) LIF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(lif_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Time step (last %d)" % n_timesteps)
        plt.ylabel("Neuron index")
        plt.title("LIF neuron spike raster (sample)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_spike_raster.png", dpi=300)
        plt.close()

        # ==============================================================
        # 4) HRF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(hrf_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Time step (last %d)" % n_timesteps)
        plt.ylabel("Neuron index")
        plt.title("HRF neuron spike raster (sample)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_spike_raster.png", dpi=300)
        plt.close()

        print(
            f"Saved visualizations:\n"
            f"  {save_prefix}_LIF_membrane_heatmap.png\n"
            f"  {save_prefix}_HRF_potential_heatmap.png\n"
            f"  {save_prefix}_LIF_spike_raster.png\n"
            f"  {save_prefix}_HRF_spike_raster.png"
        )



def visualize_dynamics_and_spikes_first(
    model, loader, device, n_neurons=100, n_timesteps=150, save_prefix="spiking_coesn"
):
    """
    Visualizes membrane potentials and spike rasters for LIF and HRF neurons
    over the FIRST n_timesteps for a random subset of neurons.
    """

    model.eval()
    with torch.no_grad():
        # --- Get one batch ---
        images, labels = next(iter(loader))
        images = images.to(device)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
        B, T, _ = images.shape

        # Restrict to the FIRST n_timesteps available
        Tmax = min(n_timesteps, T)

        # --- Initialize states ---
        hy = torch.zeros(B, model.n_hid, device=device)
        hz = torch.zeros(B, model.n_hid, device=device)
        ref_period = torch.zeros(B, model.n_hid, device=device)
        s = torch.zeros(B, model.n_hid, device=device)
        lif_v = torch.zeros(B, model.n_hid, device=device)

        # --- Recordings (first Tmax steps) ---
        lif_vs, hrf_ys = [], []
        lif_spikes, hrf_spikes = [], []

        # --- Run simulation ---
        for t in range(Tmax):     # <-- FIRST n_timesteps instead of whole sequence
            hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
                images[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )

            lif_vs.append(lif_v[0].detach().cpu().numpy())
            hrf_ys.append(hy[0].detach().cpu().numpy())
            lif_spikes.append(lif_s[0].detach().cpu().numpy())
            hrf_spikes.append(s[0].detach().cpu().numpy())

        # --- Convert to arrays ---
        lif_vs = np.stack(lif_vs, axis=0)
        hrf_ys = np.stack(hrf_ys, axis=0)
        lif_spikes = np.stack(lif_spikes, axis=0)
        hrf_spikes = np.stack(hrf_spikes, axis=0)

        # --- Select subset of neurons ---
        n_total = lif_vs.shape[1]
        sel_idx = random.sample(range(n_total), min(n_neurons, n_total))

        lif_vs = lif_vs[:, sel_idx].T
        hrf_ys = hrf_ys[:, sel_idx].T
        lif_spikes = lif_spikes[:, sel_idx].T
        hrf_spikes = hrf_spikes[:, sel_idx].T

        # ==============================================================
        # 1) LIF membrane potentials – raw + z-scored
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(lif_vs, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("LIF membrane potentials (raw)")
        axes[0].set_xlabel("Time step (first 150)")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow(
            (lif_vs - lif_vs.mean(axis=1, keepdims=True)) /
            (lif_vs.std(axis=1, keepdims=True) + 1e-9),
            aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2
        )
        axes[1].set_title("LIF membrane potentials (z-scored)")
        axes[1].set_xlabel("Time step (first 150)")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("LIF neuron activity comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 2) HRF membrane potentials – raw + z-scored
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(hrf_ys, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("HRF membrane potentials (raw)")
        axes[0].set_xlabel("Time step (first 150)")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow(
            (hrf_ys - hrf_ys.mean(axis=1, keepdims=True)) /
            (hrf_ys.std(axis=1, keepdims=True) + 1e-9),
            aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2
        )
        axes[1].set_title("HRF membrane potentials (z-scored)")
        axes[1].set_xlabel("Time step (first 150)")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("HRF neuron activity comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 3) LIF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(lif_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Time step (first 150)")
        plt.ylabel("Neuron index")
        plt.title("LIF neuron spike raster (sample)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_spike_raster.png", dpi=300)
        plt.close()

        # ==============================================================
        # 4) HRF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(hrf_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Time step (first 150)")
        plt.ylabel("Neuron index")
        plt.title("HRF neuron spike raster (sample)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_spike_raster.png", dpi=300)
        plt.close()

        print(
            f"Saved visualizations:\n"
            f"  {save_prefix}_LIF_membrane_comparison.png\n"
            f"  {save_prefix}_HRF_membrane_comparison.png\n"
            f"  {save_prefix}_LIF_spike_raster.png\n"
            f"  {save_prefix}_HRF_spike_raster.png"
        )




def visualize_dynamics_and_spikes_middle(
    model, loader, device, n_neurons=100, n_timesteps=200, save_prefix="spiking_coesn"
):
    """
    Visualizes membrane potentials and spike rasters for LIF and HRF neurons
    over the MIDDLE n_timesteps of the sequence.
    """

    model.eval()
    with torch.no_grad():
        # --- Get one batch ---
        images, labels = next(iter(loader))
        images = images.to(device)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
        B, T, _ = images.shape

        # --- Compute middle slice ---
        n_timesteps = min(n_timesteps, T)
        start = max(0, (T - n_timesteps) // 2)
        end = start + n_timesteps
        print(f"Middle slice: t={start} to t={end} out of T={T}")

        # --- Initialize states ---
        hy = torch.zeros(B, model.n_hid, device=device)
        hz = torch.zeros(B, model.n_hid, device=device)
        ref_period = torch.zeros(B, model.n_hid, device=device)
        s = torch.zeros(B, model.n_hid, device=device)
        lif_v = torch.zeros(B, model.n_hid, device=device)
        theta_lif = torch.zeros(B, model.n_hid, device=device)

        # --- Recordings ---
        lif_vs, hrf_ys = [], []
        lif_spikes, hrf_spikes = [], []

        # --- Run **full** simulation, but store only middle timesteps ---
        for t in range(T):
            hy, hz, s, ref_period, lif_v, lif_s = model.bio_cell(
                images[:, t], hy, hz, lif_v, s, ref_period=ref_period
            )

            if start <= t < end:
                lif_vs.append(lif_v[0].detach().cpu().numpy())
                hrf_ys.append(hy[0].detach().cpu().numpy())
                lif_spikes.append(lif_s[0].detach().cpu().numpy())
                hrf_spikes.append(s[0].detach().cpu().numpy())

        # --- Convert to arrays ---
        lif_vs = np.stack(lif_vs, axis=0)
        hrf_ys = np.stack(hrf_ys, axis=0)
        lif_spikes = np.stack(lif_spikes, axis=0)
        hrf_spikes = np.stack(hrf_spikes, axis=0)

        # --- Select subset of neurons ---
        n_total = lif_vs.shape[1]
        sel_idx = random.sample(range(n_total), min(n_neurons, n_total))

        lif_vs = lif_vs[:, sel_idx].T
        hrf_ys = hrf_ys[:, sel_idx].T
        lif_spikes = lif_spikes[:, sel_idx].T
        hrf_spikes = hrf_spikes[:, sel_idx].T

        # ==============================================================
        # 1) LIF membrane potentials – raw + z-scored
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(lif_vs, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("LIF membrane potentials (raw)")
        axes[0].set_xlabel("Middle time steps")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow(
            (lif_vs - lif_vs.mean(axis=1, keepdims=True)) /
            (lif_vs.std(axis=1, keepdims=True) + 1e-9),
            aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2
        )
        axes[1].set_title("LIF membrane potentials (z-scored)")
        axes[1].set_xlabel("Middle time steps")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("LIF neuron activity comparison (middle segment)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 2) HRF membrane potentials – raw + z-scored
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        im0 = axes[0].imshow(hrf_ys, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("HRF membrane potentials (raw)")
        axes[0].set_xlabel("Middle time steps")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="Membrane potential (raw)")

        im1 = axes[1].imshow(
            (hrf_ys - hrf_ys.mean(axis=1, keepdims=True)) /
            (hrf_ys.std(axis=1, keepdims=True) + 1e-9),
            aspect='auto', cmap='viridis', origin='lower', vmin=-2, vmax=2
        )
        axes[1].set_title("HRF membrane potentials (z-scored)")
        axes[1].set_xlabel("Middle time steps")
        fig.colorbar(im1, ax=axes[1], label="Membrane potential (z)")

        plt.suptitle("HRF neuron activity comparison (middle segment)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_membrane_comparison.png", dpi=300)
        plt.close()

        # ==============================================================
        # 3) LIF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(lif_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Middle time steps")
        plt.ylabel("Neuron index")
        plt.title("LIF neuron spike raster (middle segment)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_spike_raster.png", dpi=300)
        plt.close()

        # ==============================================================
        # 4) HRF spike raster
        # ==============================================================
        plt.figure(figsize=(10, 6))
        for i, neuron_spikes in enumerate(hrf_spikes):
            spike_times = np.where(neuron_spikes > 0)[0]
            plt.vlines(spike_times, i + 0.5, i + 1.5, color="black", linewidth=0.7)
        plt.xlabel("Middle time steps")
        plt.ylabel("Neuron index")
        plt.title("HRF neuron spike raster (middle segment)")
        plt.ylim(0.5, len(sel_idx) + 0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_spike_raster.png", dpi=300)
        plt.close()

        print(
            f"Saved visualizations:\n"
            f"  {save_prefix}_LIF_membrane_comparison.png\n"
            f"  {save_prefix}_HRF_membrane_comparison.png\n"
            f"  {save_prefix}_LIF_spike_raster.png\n"
            f"  {save_prefix}_HRF_spike_raster.png"
        )


def visualize_coesn_hy(
    model,
    loader,
    device,
    n_neurons=100,
    save_path="hy_ron.png"
):
    """
    Visualize hy dynamics for coESN.
    
    Produces ONE figure with 2 subplots:
      - Raw hy activity (heatmap)
      - Z-scored hy activity (heatmap)
    """

    model.eval()
    with torch.no_grad():

        # ---- Get one batch ----
        images, labels = next(iter(loader))
        images = images.to(device)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
        B, T, _ = images.shape

        # ---- Initialize states ----
        hy = torch.zeros(B, model.n_hid, device=device)
        hz = torch.zeros(B, model.n_hid, device=device)

        # ---- Record hy over time ----
        hy_list = []

        for t in range(T):
            hy, hz = model.cell(images[:, t], hy, hz)
            hy_list.append(hy[0].detach().cpu().numpy())  # take sample 0

        hy_array = np.stack(hy_list, axis=0)  # shape (T, neurons)

        # ---- Select subset of neurons ----
        n_total = hy_array.shape[1]
        sel_idx = random.sample(range(n_total), min(n_neurons, n_total))
        hy_sel = hy_array[:, sel_idx].T  # shape (neurons, time)

        # ---- Z-score normalization per neuron ----
        hy_norm = (hy_sel - hy_sel.mean(axis=1, keepdims=True)) / (
            hy_sel.std(axis=1, keepdims=True) + 1e-9
        )

        # ---- Plot raw + z-scored ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        im0 = axes[0].imshow(hy_sel, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("hy dynamics (raw)")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="hy")

        im1 = axes[1].imshow(hy_norm, aspect='auto', cmap='viridis',
                             origin='lower', vmin=-2, vmax=2)
        axes[1].set_title("hy dynamics (z-scored)")
        axes[1].set_xlabel("Time step")
        fig.colorbar(im1, ax=axes[1], label="z-scored hy")

        plt.suptitle("coESN hy state dynamics", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved hy visualization → {save_path}")


def visualize_coesn_hy_middle(
    model,
    loader,
    device,
    n_neurons=100,
    n_timesteps=200,
    save_path="hy_middle.png"
):
    """
    Visualize hy dynamics for coESN over the MIDDLE n_timesteps.
    
    Produces ONE figure with 2 subplots:
      - Raw hy activity (heatmap)
      - Z-scored hy activity (heatmap)
    """

    model.eval()
    with torch.no_grad():

        # ---- Get one batch ----
        images, labels = next(iter(loader))
        images = images.to(device)
        images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
        B, T, _ = images.shape

        # ---- Define middle segment ----
        n_timesteps = min(n_timesteps, T)
        start = max(0, (T - n_timesteps) // 2)
        end = start + n_timesteps
        print(f"[coESN hy middle] middle slice: t={start} to t={end} out of T={T}")

        # ---- Initialize states ----
        hy = torch.zeros(B, model.n_hid, device=device)
        hz = torch.zeros(B, model.n_hid, device=device)

        # ---- Record hy only in the middle slice ----
        hy_list = []

        for t in range(T):
            hy, hz = model.cell(images[:, t], hy, hz)

            if start <= t < end:
                hy_list.append(hy[0].detach().cpu().numpy())  # sample 0

        # ---- Stack into array ----
        hy_array = np.stack(hy_list, axis=0)  # shape (middle_T, neurons)

        # ---- Select subset of neurons ----
        n_total = hy_array.shape[1]
        sel_idx = random.sample(range(n_total), min(n_neurons, n_total))
        hy_sel = hy_array[:, sel_idx].T  # shape (neurons, time)

        # ---- Z-score normalization ----
        hy_norm = (hy_sel - hy_sel.mean(axis=1, keepdims=True)) / (
            hy_sel.std(axis=1, keepdims=True) + 1e-9
        )

        # ---- Plot raw + z-scored ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        im0 = axes[0].imshow(hy_sel, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("hy dynamics (raw, middle slice)")
        axes[0].set_xlabel("Middle time steps")
        axes[0].set_ylabel("Neuron index")
        fig.colorbar(im0, ax=axes[0], label="hy")

        im1 = axes[1].imshow(
            hy_norm, aspect='auto', cmap='viridis', origin='lower',
            vmin=-2, vmax=2
        )
        axes[1].set_title("hy dynamics (z-scored, middle slice)")
        axes[1].set_xlabel("Middle time steps")
        fig.colorbar(im1, ax=axes[1], label="z-scored hy")

        plt.suptitle("coESN hy state dynamics (middle segment)", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved middle-slice hy visualization → {save_path}")

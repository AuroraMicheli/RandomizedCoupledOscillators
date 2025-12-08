import matplotlib.pyplot as plt
import numpy as np
import random
import torch

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
        # 1) LIF membrane potentials
        # ==============================================================
        '''
        plt.figure(figsize=(10, 6))
        #plt.imshow(lif_vs, aspect='auto', cmap='viridis', origin='lower')
        plt.imshow((lif_vs - lif_vs.mean(axis=1, keepdims=True)) / (lif_vs.std(axis=1, keepdims=True) + 1e-9),
           aspect='auto', cmap='viridis', origin='lower') #normalized per neuron to better visualize
           #z-score normalization: for each neuron, subtract neuron's mean and divide by neuron's std. replaces original voltage scale
           #with a dimensionless scale centered around 0

        plt.colorbar(label="Membrane potential (LIF)")
        plt.xlabel("Time step (last %d)" % n_timesteps)
        plt.ylabel("Neuron index")
        plt.title("LIF membrane potentials (sample)")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_LIF_membrane_heatmap.png", dpi=300)
        plt.close()
        '''
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

        '''
        # ==============================================================
        # 2) HRF potentials
        # ==============================================================
        plt.figure(figsize=(10, 6))
        #plt.imshow(hrf_ys, aspect='auto', cmap='viridis', origin='lower')
        plt.imshow((hrf_ys - hrf_ys.mean(axis=1, keepdims=True)) / (hrf_ys.std(axis=1, keepdims=True) + 1e-9),
           aspect='auto', cmap='viridis', origin='lower') #normalized per neuron to better visualize
        plt.colorbar(label="Potential (HRF)")
        plt.xlabel("Time step (last %d)" % n_timesteps)
        plt.ylabel("Neuron index")
        plt.title("HRF potentials (sample)")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_HRF_potential_heatmap.png", dpi=300)
        plt.close()
        '''
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

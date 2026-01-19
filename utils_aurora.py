import matplotlib.pyplot as plt
import numpy as np
import random
import torch

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
    hrf_sops = hrf_spikes * n_hid  # dense HRF→HRF

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

"""
Analyze SHD temporal structure to inform hyperparameter selection.
Computes: spike rates, inter-spike intervals, frequency content,
active channels per timestep, etc.
"""
import numpy as np
import h5py
import json

# Try to load SHD - if not available, we'll use known properties
DATA_DIR = "data/SHD"

try:
    with h5py.File(f"{DATA_DIR}/shd_train.h5", 'r') as f:
        spike_times_all = f['spikes']['times']
        spike_units_all = f['spikes']['units']
        labels = f['labels'][()]
        
        n_samples = len(labels)
        print(f"SHD train: {n_samples} samples, {len(set(labels))} classes")
        
        # Analyze first 500 samples for speed
        N_ANALYZE = min(500, n_samples)
        
        all_durations = []
        all_n_spikes = []
        all_isi = []  # inter-spike intervals per channel
        all_rates = []  # spikes per second per channel
        channel_activity = np.zeros(700)
        spikes_per_step = []  # at different binning resolutions
        
        for i in range(N_ANALYZE):
            times = spike_times_all[i][()]
            units = spike_units_all[i][()]
            
            if len(times) == 0:
                continue
            
            duration = times.max() - times.min()
            all_durations.append(times.max())
            all_n_spikes.append(len(times))
            
            # Per-channel analysis
            for ch in np.unique(units):
                ch_times = np.sort(times[units == ch])
                channel_activity[int(ch)] += 1
                
                if len(ch_times) > 1:
                    isis = np.diff(ch_times)
                    all_isi.extend(isis.tolist())
                    
                    # Firing rate for this channel in this sample
                    rate = len(ch_times) / (times.max() + 1e-6)
                    all_rates.append(rate)
            
            # Spikes per time bin (at different resolutions)
            for num_steps in [50, 100, 250, 500]:
                bins = np.floor(times / 1.4 * num_steps).astype(int)
                bins = np.clip(bins, 0, num_steps - 1)
                counts_per_step = np.bincount(bins, minlength=num_steps)
                spikes_per_step.append({
                    'num_steps': num_steps,
                    'mean': counts_per_step.mean(),
                    'std': counts_per_step.std(),
                    'max': counts_per_step.max(),
                    'pct_zero': (counts_per_step == 0).mean() * 100
                })
        
        all_isi = np.array(all_isi)
        all_rates = np.array(all_rates)
        
        print(f"\n=== TEMPORAL STATISTICS (first {N_ANALYZE} samples) ===")
        print(f"Sample durations: {np.mean(all_durations):.3f} ± {np.std(all_durations):.3f} s")
        print(f"  min={np.min(all_durations):.3f}, max={np.max(all_durations):.3f} s")
        print(f"Spikes per sample: {np.mean(all_n_spikes):.1f} ± {np.std(all_n_spikes):.1f}")
        print(f"  min={np.min(all_n_spikes)}, max={np.max(all_n_spikes)}")
        
        print(f"\n=== INTER-SPIKE INTERVALS ===")
        print(f"ISI: mean={np.mean(all_isi)*1000:.2f} ms, median={np.median(all_isi)*1000:.2f} ms")
        print(f"  std={np.std(all_isi)*1000:.2f} ms")
        print(f"  5th pct={np.percentile(all_isi, 5)*1000:.2f} ms")
        print(f"  25th pct={np.percentile(all_isi, 25)*1000:.2f} ms")
        print(f"  75th pct={np.percentile(all_isi, 75)*1000:.2f} ms")
        print(f"  95th pct={np.percentile(all_isi, 95)*1000:.2f} ms")
        
        # Convert ISI to frequency
        isi_freq = 1.0 / (all_isi + 1e-8)
        print(f"\n=== IMPLIED FIRING FREQUENCIES ===")
        print(f"Frequency from ISI: mean={np.mean(isi_freq):.1f} Hz, median={np.median(isi_freq):.1f} Hz")
        print(f"  5th-95th pct: {np.percentile(isi_freq, 5):.1f} - {np.percentile(isi_freq, 95):.1f} Hz")
        
        print(f"\n=== PER-CHANNEL FIRING RATES (Hz) ===")
        print(f"Rate: mean={np.mean(all_rates):.1f} Hz, median={np.median(all_rates):.1f} Hz")
        print(f"  std={np.std(all_rates):.1f} Hz")
        print(f"  5th-95th pct: {np.percentile(all_rates, 5):.1f} - {np.percentile(all_rates, 95):.1f} Hz")
        
        print(f"\n=== CHANNEL ACTIVITY ===")
        active_channels = (channel_activity > 0).sum()
        print(f"Active channels (in {N_ANALYZE} samples): {active_channels}/700")
        print(f"Mean samples per channel: {channel_activity[channel_activity>0].mean():.1f}")
        
        # How many channels active per sample?
        channels_per_sample = []
        for i in range(N_ANALYZE):
            units = spike_units_all[i][()]
            channels_per_sample.append(len(np.unique(units)))
        print(f"Active channels per sample: {np.mean(channels_per_sample):.1f} ± {np.std(channels_per_sample):.1f}")
        print(f"  min={np.min(channels_per_sample)}, max={np.max(channels_per_sample)}")
        
        print(f"\n=== SPIKES PER TIME BIN (input sparsity) ===")
        for ns in [50, 100, 250, 500]:
            entries = [s for s in spikes_per_step if s['num_steps'] == ns]
            means = [s['mean'] for s in entries]
            maxs = [s['max'] for s in entries]
            zeros = [s['pct_zero'] for s in entries]
            dt_ms = 1400.0 / ns
            print(f"  {ns} steps (dt={dt_ms:.1f}ms): "
                  f"mean={np.mean(means):.1f} spikes/bin, "
                  f"max={np.mean(maxs):.0f}, "
                  f"{np.mean(zeros):.1f}% empty bins")
        
        # Key frequency: what's the dominant timescale of the INPUT signal?
        # The cochlear model produces spikes that encode audio frequencies
        # SHD uses the Heidelberg auditory model with 700 frequency channels
        # spanning roughly 20 Hz to 20 kHz (cochlear tonotopy)
        # But the SPIKE PATTERNS have much slower dynamics than the carrier frequencies
        
        # Estimate effective input bandwidth from autocorrelation
        print(f"\n=== EFFECTIVE INPUT DYNAMICS ===")
        # Bin at 250 steps, look at temporal correlation of total spike count
        autocorrs = []
        for i in range(min(100, N_ANALYZE)):
            times = spike_times_all[i][()]
            if len(times) == 0:
                continue
            bins = np.floor(times / 1.4 * 250).astype(int)
            bins = np.clip(bins, 0, 249)
            counts = np.bincount(bins, minlength=250).astype(float)
            # Normalize
            counts = (counts - counts.mean()) / (counts.std() + 1e-8)
            # Autocorrelation
            ac = np.correlate(counts, counts, mode='full')
            ac = ac[len(ac)//2:]  # keep positive lags
            ac = ac / ac[0]
            autocorrs.append(ac[:50])  # first 50 lags
        
        mean_ac = np.mean(autocorrs, axis=0)
        # Find where autocorrelation drops to 1/e
        tau_idx = np.argmax(mean_ac < 1/np.e)
        tau_ms = tau_idx * (1400.0 / 250)
        print(f"Autocorrelation time constant: ~{tau_idx} bins = {tau_ms:.1f} ms")
        print(f"  → dominant frequency: ~{1000/tau_ms:.1f} Hz (if periodic)")
        print(f"  → This is the timescale the reservoir should match!")
        
        # Print autocorrelation values
        print(f"\nAutocorrelation (first 20 lags, dt=5.6ms):")
        for lag in range(0, 20, 2):
            print(f"  lag {lag} ({lag*5.6:.0f}ms): {mean_ac[lag]:.3f}")

except FileNotFoundError:
    print("SHD data not found - using known properties from literature")
    print("Run this on your machine with data/SHD/ available")
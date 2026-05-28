"""
plot_readout_ablation.py

Produces a publication-quality grouped bar chart for the readout strategy
ablation of [NAME]. Saves the figure as both PDF and PNG.

Usage:
    python plot_readout_ablation.py

Output:
    readout_ablation.pdf   (for LaTeX inclusion)
    readout_ablation.png   (for quick preview)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────
# Format: (mean, std) per (dataset, readout_mode)
# Readout modes: Final, Mean, RMS, Std, Combined, Spikes(mean)

data = {
    "sMNIST": {
        "Final":    (93.64, 0.25),
        "Mean":     (93.54, 0.28),
        "RMS":      (93.56, 0.08),
        "Std":      (93.16, 0.27),
        "Combined": (93.92, 0.21),
        "Spikes":   (91.09, 0.28),
    },
    "FordA": {
        "Final":    (69.72, 0.68),
        "Mean":     (73.71, 1.12),
        "RMS":      (75.05, 0.19),
        "Std":      (73.40, 0.36),
        "Combined": (74.22, 0.86),
        "Spikes":   (74.02, 0.84),
    },
    "SHD": {
        "Final":    (83.48, 0.44),
        "Mean":     (83.16, 0.40),
        "RMS":      (85.59, 0.65),
        "Std":      (85.70, 0.39),
        "Combined": (89.78, 0.35),
        "Spikes":   (81.08, 0.43),
    },
    "DVS Gesture": {
        "Final":    (71.81, 1.59),
        "Mean":     (77.78, 0.94),
        "RMS":      (76.77, 2.40),
        "Std":      (74.87, 0.36),
        "Combined": (78.54, 0.90),
        "Spikes":   (73.74, 1.39),
    },
}

datasets   = list(data.keys())
modes      = ["Final", "Mean", "RMS", "Std", "Combined", "Spikes"]
n_datasets = len(datasets)
n_modes    = len(modes)

# ── Colours ───────────────────────────────────────────────────────────────────
# First 5 modes: shades of blue/teal (membrane-based)
# Last mode (Spikes): orange, visually separated
membrane_colors = [
    "#1f4e79",   # Final      — dark navy
    "#2e75b6",   # Mean       — mid blue
    "#5ba3d9",   # RMS        — light blue
    "#9dc3e6",   # Std        — pale blue
    "#70ad47",   # Combined   — green (best performer, stands out)
]
spike_color = "#c55a11"  # orange

colors = membrane_colors + [spike_color]

# ── Layout ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        19,
    "axes.titlesize":   19,
    "axes.labelsize":   17,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  20,
    "figure.dpi":       150,
    "pdf.fonttype":     42,   # embeds fonts for PDF
    "ps.fonttype":      42,
})

fig, axes = plt.subplots(1, n_datasets, figsize=(20, 3.5), sharey=False)
fig.subplots_adjust(wspace=0.25, left=0.04, right=0.99, top=0.93, bottom=0.18)

bar_width   = 0.13
group_gap   = 0.08   # extra gap between membrane group and spikes bar
x_base      = np.arange(n_modes)

for ax_idx, (dataset, ax) in enumerate(zip(datasets, axes)):
    means = [data[dataset][m][0] for m in modes]
    stds  = [data[dataset][m][1] for m in modes]

    # Shift spike bar slightly to the right to create visual separation
    x_positions = list(range(n_modes - 1)) + [n_modes - 1 + group_gap / bar_width]
    x_positions = np.array(x_positions, dtype=float) * bar_width

    # Centre the whole group
    x_positions -= x_positions.mean()

    bars = ax.bar(
        x_positions,
        means,
        width=bar_width * 0.88,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Error bars
    ax.errorbar(
        x_positions,
        means,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    # Highlight best bar with a bold outline
    best_idx = int(np.argmax(means))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)

    # Add a thin vertical separator line between membrane and spikes groups
    sep_x = (x_positions[-2] + x_positions[-1]) / 2.0
    ax.axvline(sep_x, color="gray", linewidth=0.6, linestyle="--", zorder=2)

    # Axis formatting
    y_min = min(means) - max(stds) - 1.5
    y_max = max(means) + max(stds) + 1.5
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_title(dataset, fontweight="bold", pad=4)
    ax.yaxis.grid(True, linewidth=0.4, color="lightgray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax_idx == 0:
        ax.set_ylabel("Test Accuracy (%)", labelpad=4)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_labels = ["Final", "Mean", "RMS", "Std", "Combined (RMS+Std+Final)", "Spikes (mean rate)"]
patches = [
    mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=l)
    for c, l in zip(colors, legend_labels)
]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
    handlelength=1.2,
    columnspacing=1.0,
)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fname = f"readout_ablation.{ext}"
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    print(f"Saved: {fname}")

plt.close(fig)
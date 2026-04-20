"""
Plot Pareto curves: RON vs s-RON energy-accuracy tradeoff.

Each N_hid value gets a distinct shade within the model's color family
(light = small N_hid, dark = large N_hid). Two colorbars show the gradient
for RON and s-RON respectively.

Reads: pareto_results/pareto_summary.json
Saves: pareto_results/pareto_curve.pdf / .png

Usage:
    python plot_pareto.py
    python plot_pareto.py --results_dir /path/to/pareto_results
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Style
# =============================================================================

def set_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':           9,
        'axes.titlesize':      10,
        'axes.labelsize':      9,
        'xtick.labelsize':     8,
        'ytick.labelsize':     8,
        'legend.fontsize':     8,
        'axes.linewidth':      0.8,
        'xtick.major.width':   0.8,
        'ytick.major.width':   0.8,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'savefig.dpi':         300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches':  0.05,
    })


# =============================================================================
# Config
# =============================================================================

DATASETS = ['sMNIST', 'FordA', 'Adiac']

RON_LIGHT  = '#FDDBC7'   # very pale red-orange
RON_DARK   = '#D6604D'   # original RON red-orange
SRON_LIGHT = '#DEEBF7'   # very pale blue
SRON_DARK  = '#2166AC'   # original s-RON blue


# =============================================================================
# Helpers
# =============================================================================

def make_shade_palette(light_hex, dark_hex, n):
    light = np.array(mcolors.to_rgb(light_hex))
    dark  = np.array(mcolors.to_rgb(dark_hex))
    shades = []
    for i in range(n):
        t = i / max(n - 1, 1)
        shades.append(tuple(np.clip((1 - t) * light + t * dark, 0, 1)))
    return shades


def load_results(results_dir):
    path = os.path.join(results_dir, 'pareto_summary.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"pareto_summary.json not found in {results_dir}.")
    with open(path) as f:
        return json.load(f)


def extract_points(data, dataset, model):
    model_data = data.get(dataset, {}).get(model, {})
    points = []
    for n_hid_str, result in model_data.items():
        if result.get('n_trials_ok', 0) == 0:
            continue
        points.append({
            'n_hid':       int(n_hid_str),
            'energy_mean': result['energy_J_mean'],
            'energy_std':  result['energy_J_std'],
            'acc_mean':    result['test_acc_mean'],
            'acc_std':     result['test_acc_std'],
        })
    points.sort(key=lambda x: x['energy_mean'])
    return points


# =============================================================================
# Plot
# =============================================================================

def plot_pareto(data, out_dir):
    set_style()

    # Collect all N_hid values
    all_n_hids = set()
    for ds in DATASETS:
        for model in ['ron', 'sron']:
            for p in extract_points(data, ds, model):
                all_n_hids.add(p['n_hid'])
    all_n_hids  = sorted(all_n_hids)
    n_sizes     = len(all_n_hids)
    nhid_to_idx = {n: i for i, n in enumerate(all_n_hids)}

    ron_shades  = make_shade_palette(RON_LIGHT,  RON_DARK,  n_sizes)
    sron_shades = make_shade_palette(SRON_LIGHT, SRON_DARK, n_sizes)

    # Figure: 3 data panels + 1 narrow colorbar panel
    fig = plt.figure(figsize=(10.5, 3.2))
    # width_ratios: 3 equal data panels + 1 narrow colorbar column
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.10],
        wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cb = fig.add_subplot(gs[0, 3])

    # ── Data panels ──────────────────────────────────────────────────────────
    for ax, dataset in zip(axes, DATASETS):
        ron_pts  = extract_points(data, dataset, 'ron')
        sron_pts = extract_points(data, dataset, 'sron')

        for pts, shades, dark, marker in [
            (ron_pts,  ron_shades,  RON_DARK,  'o'),
            (sron_pts, sron_shades, SRON_DARK, 's'),
        ]:
            if not pts:
                continue

            energies = np.array([p['energy_mean'] for p in pts])
            accs     = np.array([p['acc_mean']    for p in pts])
            e_stds   = np.array([p['energy_std']  for p in pts])
            a_stds   = np.array([p['acc_std']     for p in pts])
            n_hids   = [p['n_hid'] for p in pts]

            # Thin connecting line
            ax.plot(energies, accs, '-', color=dark,
                    linewidth=1.0, alpha=0.30, zorder=2)

            # Shaded markers
            for e, a, es, as_, n in zip(energies, accs, e_stds, a_stds, n_hids):
                ax.errorbar(
                    e, a, xerr=es, yerr=as_,
                    fmt=marker, color=shades[nhid_to_idx[n]],
                    markersize=7,
                    markeredgewidth=0.8, markeredgecolor=dark,
                    elinewidth=0.7, capsize=2, capthick=0.7,
                    alpha=1.0, zorder=3, linestyle='none',
                )

        ax.set_xscale('log')
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel('Energy (J)', fontsize=9)
        ax.grid(True, which='both',  alpha=0.10, linewidth=0.4)
        ax.grid(True, which='major', alpha=0.20, linewidth=0.6)

        all_accs = ([p['acc_mean'] for p in ron_pts] +
                    [p['acc_mean'] for p in sron_pts])
        all_stds = ([p['acc_std']  for p in ron_pts] +
                    [p['acc_std']  for p in sron_pts])
        if all_accs:
            ax.set_ylim(
                min(a - s for a, s in zip(all_accs, all_stds)) - 2,
                max(a + s for a, s in zip(all_accs, all_stds)) + 3,
            )

        if ax is axes[0]:
            ax.set_ylabel('Test accuracy (%)', fontsize=9)

    # ── Model marker legend (top-left of first panel) ─────────────────────────
    model_legend = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=RON_DARK, markeredgecolor=RON_DARK,
               markeredgewidth=0.8, markersize=7, label='RON'),
        Line2D([0], [0], marker='s', color='none',
               markerfacecolor=SRON_DARK, markeredgecolor=SRON_DARK,
               markeredgewidth=0.8, markersize=7, label='s-RON'),
    ]
    axes[0].legend(handles=model_legend, loc='upper left',
                   frameon=False, handlelength=0.8,
                   handletextpad=0.4, labelspacing=0.3)

    # ── Colorbar panel ────────────────────────────────────────────────────────
    # Draw two vertical colorbars stacked in ax_cb:
    # top half = RON gradient, bottom half = s-RON gradient

    ax_cb.set_axis_off()

    n_hid_min = all_n_hids[0]
    n_hid_max = all_n_hids[-1]

    for pos, light, dark, label, marker in [
        ([0.52, 0.55, 0.35, 0.38], RON_LIGHT,  RON_DARK,  'RON',   'o'),
        ([0.52, 0.05, 0.35, 0.38], SRON_LIGHT, SRON_DARK, 's-RON', 's'),
    ]:
        cax = ax_cb.inset_axes(pos)
        cmap = LinearSegmentedColormap.from_list(
            '', [light, dark], N=256)
        norm = plt.Normalize(vmin=n_hid_min, vmax=n_hid_max)
        cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                          orientation='vertical')
        cb.set_ticks([n_hid_min, n_hid_max])
        cb.set_ticklabels([str(n_hid_min), str(n_hid_max)])
        cb.ax.tick_params(labelsize=7, length=2, pad=2)
        cb.outline.set_linewidth(0.5)

        # Model label above each bar
        cax.set_title(label, fontsize=7.5, pad=3)

    fig.savefig(os.path.join(out_dir, 'pareto_curve_lif02.pdf'))
    fig.savefig(os.path.join(out_dir, 'pareto_curve._lif02.png'))
    plt.close()
    print(f"Saved: {os.path.join(out_dir, 'pareto_curve.pdf/.png')}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='pareto_results_lif02')
    parser.add_argument('--out_dir',     type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    data = load_results(args.results_dir)
    plot_pareto(data, out_dir)
    print("Done.")


if __name__ == '__main__':
    main()
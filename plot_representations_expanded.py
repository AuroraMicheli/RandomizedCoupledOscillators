"""
Generate publication-quality figures for HRF-Res paper.

Produces:
  figures/bandpass_summary.{pdf,png}
  figures/repr_analysis_combined.{pdf,png}

Usage:
    python plot_figures.py
    python plot_figures.py --results_dir /path/to/analysis_results_extended
    python plot_figures.py --style slides
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# Style — unified font sizes for both figures
# =============================================================================

def set_style(style='paper'):
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':           10 if style == 'paper' else 13,
        'axes.titlesize':      11 if style == 'paper' else 14,
        'axes.labelsize':      10 if style == 'paper' else 13,
        'xtick.labelsize':     9  if style == 'paper' else 12,
        'ytick.labelsize':     9  if style == 'paper' else 12,
        'legend.fontsize':     9  if style == 'paper' else 12,
        'axes.linewidth':      0.8,
        'xtick.major.width':   0.8,
        'ytick.major.width':   0.8,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'savefig.dpi':         300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches':  0.02,
    })


# =============================================================================
# Config
# =============================================================================

HRF_COLOR = '#1A78C2'
LIF_COLOR = '#E8402A'

HRF_LABEL = 'HRF-Res'
LIF_LABEL = 'LIF-Res'

DATASETS = ['sMNIST', 'fordA', 'shd', 'dvs_gesture']

DATASET_LABELS = {
    'sMNIST':      'sMNIST',
    'fordA':       'FordA',
    'shd':         'SHD',
    'dvs_gesture': 'DVS\nGesture',
}

DATASET_LABELS_SHORT = {
    'sMNIST':      'sMNIST',
    'fordA':       'FordA',
    'shd':         'SHD',
    'dvs_gesture': 'DVS Gesture',
}


# =============================================================================
# Load results
# =============================================================================

def load_results(results_dir, cka_feature='mean'):
    data = {}
    for ds in DATASETS:
        path_new = os.path.join(results_dir, ds,
                                f'analysis_summary_{ds}_{cka_feature}.json')
        path_old = os.path.join(results_dir, ds,
                                f'analysis_summary_{ds}.json')
        if os.path.exists(path_new):
            with open(path_new) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_new}")
        elif os.path.exists(path_old):
            with open(path_old) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_old}  (old filename)")
        else:
            print(f"  WARNING [{ds}]: not found, skipping")
    return data


# =============================================================================
# Helper
# =============================================================================

def _shared_freq_range(data, ds_present):
    f_mins, f_maxs = [], []
    for ds in ds_present:
        r = data[ds]['freq_selectivity']
        if 'f_min' in r and 'f_max' in r:
            f_mins.append(r['f_min'])
            f_maxs.append(r['f_max'])
        else:
            vals = np.array(r['pref_freq_hrf_array'] + r['pref_freq_lif_array'])
            f_mins.append(np.percentile(vals, 1) * 0.5)
            f_maxs.append(np.percentile(vals, 99) * 2.0)
    return min(f_mins), max(f_maxs), min(f_mins) / 3.0, max(f_maxs)


# =============================================================================
# Figure 1: bandpass_summary
# Both panels same proportions as repr_analysis_combined.
# Legend placed BELOW the axes to avoid overlap with bars.
# Y-axis panel A capped at 100.
# =============================================================================

def plot_bandpass_summary(data, out_dir, style='paper'):
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'freq_selectivity' in data[ds]]
    if not ds_present:
        print("  WARNING: no freq_selectivity data, skipping bandpass_summary")
        return

    # Same height as repr_analysis_combined; wide
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.5, 2.4) if style == 'paper' else (11, 3.4))
    # Extra bottom margin to fit below-axes legend
    plt.subplots_adjust(wspace=0.35, left=0.10, right=0.97,
                        top=0.87, bottom=0.30)

    x     = np.arange(len(ds_present))
    width = 0.35

    frac_h = [data[ds]['freq_selectivity']['frac_resonant_hrf'] * 100
              for ds in ds_present]
    frac_l = [data[ds]['freq_selectivity']['frac_resonant_lif'] * 100
              for ds in ds_present]
    q_h    = [data[ds]['freq_selectivity']['q_hrf_mean_resonant']
              for ds in ds_present]
    q_l    = [data[ds]['freq_selectivity']['q_lif_mean_resonant']
              for ds in ds_present]

    # Panel A
    b1 = axes[0].bar(x - width/2, frac_h, width,
                     color=HRF_COLOR, edgecolor='white', linewidth=0.5)
    b2 = axes[0].bar(x + width/2, frac_l, width,
                     color=LIF_COLOR, edgecolor='white', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[0].set_ylabel('Band-pass neurons (%)', fontsize=10)
    axes[0].yaxis.set_label_coords(-0.18, 0.40)
    axes[0].set_title('Fraction of resonant neurons')
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].text(-0.16, 1.06, 'A', transform=axes[0].transAxes,
                 fontsize=13, fontweight='bold')
    # Legend below the axes, centred
    axes[0].legend([b1, b2], [HRF_LABEL, LIF_LABEL],
                   frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, -0.22), ncol=2)

    # Panel B
    b3 = axes[1].bar(x - width/2, q_h, width,
                     color=HRF_COLOR, edgecolor='white', linewidth=0.5)
    b4 = axes[1].bar(x + width/2, q_l, width,
                     color=LIF_COLOR, edgecolor='white', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[1].set_ylabel('Mean Q-factor')
    axes[1].set_title('Band-pass sharpness')
    axes[1].set_ylim(0, max(max(q_h), max(q_l)) * 1.30)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].text(-0.16, 1.06, 'B', transform=axes[1].transAxes,
                 fontsize=13, fontweight='bold')
    axes[1].legend([b3, b4], [HRF_LABEL, LIF_LABEL],
                   frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, -0.22), ncol=2)

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'bandpass_summary.{ext}'))
    plt.close()
    print("  Saved: bandpass_summary.pdf/png")


# =============================================================================
# Figure 2: repr_analysis_combined
# Panel A: CKA. Panel B: freq histograms (legend only on DVS Gesture).
# Same height and font proportions as bandpass_summary.
# Reduced gap between A and B.
# =============================================================================

def plot_cka(data, style='paper', ax=None):
    ds_present = [ds for ds in DATASETS if ds in data and 'cka' in data[ds]]
    if not ds_present or ax is None:
        return

    x = np.arange(len(ds_present))
    width = 0.35
    cka_hrf = [data[ds]['cka']['cka_hrf'] for ds in ds_present]
    cka_lif = [data[ds]['cka']['cka_lif'] for ds in ds_present]

    b1 = ax.bar(x - width/2, cka_hrf, width,
                color=HRF_COLOR, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + width/2, cka_lif, width,
                color=LIF_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    ax.set_ylabel('CKA score')
    ax.set_title('Class-label alignment')
    ax.set_ylim(0, max(max(cka_hrf), max(cka_lif)) * 1.30)
    ax.grid(axis='y', alpha=0.25)
    ax.text(-0.22, 1.06, 'A', transform=ax.transAxes,
            fontsize=13, fontweight='bold')
    ax.legend([b1, b2], [HRF_LABEL, LIF_LABEL],
              frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=2)


def _draw_freq_panels(axes, data, ds_list, bins, f_min_plot, f_max_plot):
    n = len(axes)
    for i, (ax, ds) in enumerate(zip(axes, ds_list)):
        if ds not in data or 'freq_selectivity' not in data[ds]:
            ax.set_visible(False)
            continue
        r   = data[ds]['freq_selectivity']
        hrf = np.array(r['pref_freq_hrf_array'])
        lif = np.array(r['pref_freq_lif_array'])
        w_h = np.ones_like(hrf) / len(hrf)
        w_l = np.ones_like(lif) / len(lif)

        ax.hist(hrf, bins=bins, weights=w_h, color=HRF_COLOR, alpha=0.45)
        ax.hist(lif, bins=bins, weights=w_l, color=LIF_COLOR, alpha=0.45)
        ax.hist(hrf, bins=bins, weights=w_h, histtype='step',
                color=HRF_COLOR, linewidth=1.0)
        ax.hist(lif, bins=bins, weights=w_l, histtype='step',
                color=LIF_COLOR, linewidth=1.0)
        ax.set_xscale('log')
        ax.set_xlim(f_min_plot, f_max_plot)
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.set_yticks([])

        if i == 0:
            ax.set_ylabel('Fraction of neurons')
            ax.text(-0.42, 1.06, 'B', transform=ax.transAxes,
                    fontsize=13, fontweight='bold')

        # Legend only on the rightmost panel (DVS Gesture)
        if i == n - 1:
            legend_els = [
                Patch(facecolor=HRF_COLOR, alpha=0.7, label=HRF_LABEL),
                Patch(facecolor=LIF_COLOR, alpha=0.7, label=LIF_LABEL),
            ]
            ax.legend(handles=legend_els, frameon=False,
                      loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)


def plot_combined(data, out_dir, style='paper'):
    ds_present_freq = [ds for ds in DATASETS
                       if ds in data and 'freq_selectivity' in data[ds]]

    # Same height as bandpass_summary
    fig = plt.figure(figsize=(9.5, 2.9) if style == 'paper' else (14, 4.0))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.8],
                           wspace=0.14,          # reduced gap A↔B
                           left=0.08, right=0.98,
                           top=0.87, bottom=0.30)
    axA = fig.add_subplot(gs[0])
    plot_cka(data, style=style, ax=axA)

    gsB  = gs[1].subgridspec(1, 4, wspace=0.10)
    axes = [fig.add_subplot(gsB[0, i]) for i in range(4)]

    if ds_present_freq:
        f_min, f_max, f_min_plot, f_max_plot = _shared_freq_range(
            data, ds_present_freq)
        bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)
        _draw_freq_panels(axes, data, DATASETS, bins, f_min_plot, f_max_plot)

    # Shared x-label for freq panels only
    mid_x = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(mid_x, 0.08, 'Preferred frequency (Hz)', ha='center', fontsize=10)

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'repr_analysis_combined.{ext}'))
    plt.close()
    print("  Saved: repr_analysis_combined.pdf/png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--out_dir',     default=None)
    parser.add_argument('--style',       default='paper',
                        choices=['paper', 'slides'])
    parser.add_argument('--cka_feature', default='mean',
                        choices=['mean', 'rms', 'final'])
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(
        script_dir, 'analysis_results_extended')
    out_dir     = args.out_dir or os.path.join(
        script_dir, 'figures_extended')
    os.makedirs(out_dir, exist_ok=True)

    set_style(args.style)

    print(f"\nLoading results from: {results_dir}")
    data = load_results(results_dir, cka_feature=args.cka_feature)
    if not data:
        print("No data loaded — check --results_dir path.")
        return

    print(f"\nGenerating figures in: {out_dir}")
    plot_bandpass_summary(data, out_dir, args.style)
    plot_combined(data, out_dir, args.style)
    print("\nDone.")


if __name__ == '__main__':
    main()


'''
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# Style
# =============================================================================

def set_style(style='paper'):
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':           8  if style == 'paper' else 11,
        'axes.titlesize':      9  if style == 'paper' else 12,
        'axes.labelsize':      8  if style == 'paper' else 11,
        'xtick.labelsize':     7  if style == 'paper' else 10,
        'ytick.labelsize':     7  if style == 'paper' else 10,
        'legend.fontsize':     7  if style == 'paper' else 10,
        'axes.linewidth':      0.8,
        'xtick.major.width':   0.8,
        'ytick.major.width':   0.8,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'savefig.dpi':         300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches':  0.02,
    })


# =============================================================================
# Config
# =============================================================================

HRF_COLOR = '#2166AC'
LIF_COLOR = '#D6604D'

DATASETS = ['sMNIST', 'fordA', 'shd', 'dvs_gesture']

DATASET_LABELS = {
    'sMNIST':      'sMNIST',
    'fordA':       'FordA',
    'shd':         'SHD',
    'dvs_gesture': 'DVS\nGesture',
}

DATASET_LABELS_SHORT = {
    'sMNIST':      'sMNIST',
    'fordA':       'FordA',
    'shd':         'SHD',
    'dvs_gesture': 'DVS Gesture',
}


# =============================================================================
# Load results
# =============================================================================

def load_results(results_dir, cka_feature='mean'):
    data = {}
    for ds in DATASETS:
        path_new = os.path.join(results_dir, ds,
                                f'analysis_summary_{ds}_{cka_feature}.json')
        path_old = os.path.join(results_dir, ds,
                                f'analysis_summary_{ds}.json')
        if os.path.exists(path_new):
            with open(path_new) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_new}")
        elif os.path.exists(path_old):
            with open(path_old) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_old}  (old filename)")
        else:
            print(f"  WARNING [{ds}]: not found, skipping")
    return data


# =============================================================================
# Helper: shared frequency axis
# =============================================================================

def _shared_freq_range(data, ds_present):
    f_mins, f_maxs = [], []
    for ds in ds_present:
        r = data[ds]['freq_selectivity']
        if 'f_min' in r and 'f_max' in r:
            f_mins.append(r['f_min'])
            f_maxs.append(r['f_max'])
        else:
            vals = np.array(r['pref_freq_hrf_array'] + r['pref_freq_lif_array'])
            f_mins.append(np.percentile(vals, 1) * 0.5)
            f_maxs.append(np.percentile(vals, 99) * 2.0)
    return min(f_mins), max(f_maxs), min(f_mins) / 3.0, max(f_maxs)


# =============================================================================
# MAIN PAPER: Figure 1 — CKA bar chart
# =============================================================================

def plot_cka(data, out_dir=None, style='paper', ax=None):
    """CKA scores across all four datasets."""
    ds_present = [ds for ds in DATASETS if ds in data and 'cka' in data[ds]]
    if not ds_present:
        print("  WARNING: no CKA data, skipping")
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(3.2, 2.4) if style == 'paper' else (5, 3.5))

    x = np.arange(len(ds_present))
    width = 0.32
    cka_hrf = [data[ds]['cka']['cka_hrf'] for ds in ds_present]
    cka_lif = [data[ds]['cka']['cka_lif'] for ds in ds_present]

    ax.bar(x - width/2, cka_hrf, width,
           color=HRF_COLOR, label='s-RON', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, cka_lif, width,
           color=LIF_COLOR, label='LIF-RC', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    ax.set_ylabel('CKA score')
    ax.set_ylim(0, max(max(cka_hrf), max(cka_lif)) * 1.35)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25)
    ax.text(-0.18, 1.05, 'A', transform=ax.transAxes,
            fontsize=11, fontweight='bold')

    if standalone:
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(out_dir, f'cka_all_datasets.{ext}'))
        plt.close()
        print("  Saved: cka_all_datasets.pdf/png")


# =============================================================================
# MAIN PAPER: Figure 2 — % band-pass + Q-factor summary (headline numbers)
# =============================================================================

def plot_bandpass_summary(data, out_dir, style='paper'):
    """
    Two-panel figure: (A) % band-pass neurons per dataset, (B) mean Q-factor.
    These are the headline results of Section 3.4 — consistent HRF >> LIF
    across all four datasets.
    """
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'freq_selectivity' in data[ds]]
    if not ds_present:
        print("  WARNING: no freq_selectivity data, skipping bandpass_summary")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 2.4) if style == 'paper' else (9, 3.5))

    x     = np.arange(len(ds_present))
    width = 0.32

    frac_h = [data[ds]['freq_selectivity']['frac_resonant_hrf'] * 100
              for ds in ds_present]
    frac_l = [data[ds]['freq_selectivity']['frac_resonant_lif'] * 100
              for ds in ds_present]
    q_h    = [data[ds]['freq_selectivity']['q_hrf_mean_resonant']
              for ds in ds_present]
    q_l    = [data[ds]['freq_selectivity']['q_lif_mean_resonant']
              for ds in ds_present]

    # Panel A: % band-pass
    axes[0].bar(x - width/2, frac_h, width,
                color=HRF_COLOR, label='s-RON', edgecolor='white', linewidth=0.5)
    axes[0].bar(x + width/2, frac_l, width,
                color=LIF_COLOR, label='LIF-RC', edgecolor='white', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[0].set_ylabel('Band-pass neurons (%)')
    axes[0].set_title('Fraction of resonant neurons')
    axes[0].set_ylim(0, 115)
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(frameon=False)
    for xi, (vh, vl) in enumerate(zip(frac_h, frac_l)):
        axes[0].text(xi - width/2, vh + 1.5, f'{vh:.0f}%',
                     ha='center', va='bottom', fontsize=6.5, color=HRF_COLOR)
        axes[0].text(xi + width/2, vl + 1.5, f'{vl:.0f}%',
                     ha='center', va='bottom', fontsize=6.5, color=LIF_COLOR)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold')

    # Panel B: mean Q-factor (resonant neurons only)
    axes[1].bar(x - width/2, q_h, width,
                color=HRF_COLOR, edgecolor='white', linewidth=0.5)
    axes[1].bar(x + width/2, q_l, width,
                color=LIF_COLOR, edgecolor='white', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[1].set_ylabel('Mean Q-factor')
    axes[1].set_title('Band-pass sharpness (resonant neurons)')
    axes[1].set_ylim(0, max(max(q_h), max(q_l)) * 1.35)
    axes[1].grid(axis='y', alpha=0.25)
    for xi, (vh, vl) in enumerate(zip(q_h, q_l)):
        axes[1].text(xi - width/2, vh + 0.05, f'{vh:.1f}',
                     ha='center', va='bottom', fontsize=6.5, color=HRF_COLOR)
        axes[1].text(xi + width/2, vl + 0.05, f'{vl:.1f}',
                     ha='center', va='bottom', fontsize=6.5, color=LIF_COLOR)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'bandpass_summary.{ext}'))
    plt.close()
    print("  Saved: bandpass_summary.pdf/png")


# =============================================================================
# MAIN PAPER: Figure 3 — Q-factor vs preferred frequency scatter
# =============================================================================

def plot_freq_q_scatter(data, out_dir, style='paper'):
    """
    Per-neuron scatter: x = preferred input frequency (log), y = Q-factor.

    HRF neurons form a cloud at moderate/high frequencies with high Q
    (sharp band-pass filters). LIF neurons collapse to a dense cluster at
    low frequencies with Q ≈ 0 (pure integrators, no resonance).

    This is the mechanistic explanation figure. It is more informative than
    the frequency histograms because Q is dimensionless and directly
    comparable across models regardless of their independently tuned dt.
    """
    ds_present = [ds for ds in DATASETS
                  if ds in data
                  and 'freq_selectivity' in data[ds]
                  and 'q_hrf_array' in data[ds]['freq_selectivity']]
    if not ds_present:
        print("  WARNING: no Q-factor arrays in JSON, skipping freq_q_scatter. "
              "Re-run the analysis script to populate them.")
        return

    n   = len(ds_present)
    fig, axes = plt.subplots(
        1, n,
        figsize=(1.9 * n, 2.2) if style == 'paper' else (3 * n, 3.5))
    if n == 1:
        axes = [axes]

    # Shared Q-axis: 99th percentile of all resonant neurons across datasets
    all_q = []
    for ds in ds_present:
        r = data[ds]['freq_selectivity']
        all_q.extend([q for q in r['q_hrf_array'] if q > 0])
        all_q.extend([q for q in r['q_lif_array'] if q > 0])
    q_max = np.percentile(all_q, 99) * 1.1 if all_q else 5.0
    q_max = max(q_max, 1.0)

    for i, (ax, ds) in enumerate(zip(axes, ds_present)):
        r     = data[ds]['freq_selectivity']
        f_hrf = np.array(r['pref_freq_hrf_array'])
        f_lif = np.array(r['pref_freq_lif_array'])
        q_hrf = np.array(r['q_hrf_array'])
        q_lif = np.array(r['q_lif_array'])

        # LIF first so HRF dots sit on top
        ax.scatter(f_lif, q_lif, c=LIF_COLOR, s=4, alpha=0.35,
                   edgecolors='none', label='LIF-RC', rasterized=True)
        ax.scatter(f_hrf, q_hrf, c=HRF_COLOR, s=4, alpha=0.45,
                   edgecolors='none', label='s-RON', rasterized=True)

        ax.set_xscale('log')
        ax.set_xlim(r.get('f_min', 0.005) / 3.0, r.get('f_max', 2.0))
        ax.set_ylim(-0.1, q_max)
        ax.set_xlabel('Preferred frequency (Hz)')
        if i == 0:
            ax.set_ylabel('Q-factor')
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.grid(True, alpha=0.25)

        # Annotation box: fraction of resonant neurons per model
        frac_h = r.get('frac_resonant_hrf', 0)
        frac_l = r.get('frac_resonant_lif', 0)
        ax.text(0.03, 0.97,
                f'band-pass:\nHRF {frac_h*100:.0f}%\nLIF {frac_l*100:.0f}%',
                transform=ax.transAxes, fontsize=6.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.5, alpha=0.9))

    # Shared legend above the figure
    handles = [
        plt.Line2D([], [], marker='o', ls='', color=HRF_COLOR,
                   markersize=5, alpha=0.85, label='s-RON (HRF)'),
        plt.Line2D([], [], marker='o', ls='', color=LIF_COLOR,
                   markersize=5, alpha=0.85, label='LIF-RC'),
    ]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'freq_q_scatter.{ext}'))
    plt.close()
    print("  Saved: freq_q_scatter.pdf/png")


# =============================================================================
# MAIN PAPER: Combined figure — CKA (A) + freq histograms (B)
# =============================================================================

def _draw_freq_panels(axes, data, ds_list, bins, f_min_plot, f_max_plot,
                      add_panel_letter=False):
    """Draw preferred-frequency histograms into a row of axes."""
    for i, (ax, ds) in enumerate(zip(axes, ds_list)):
        if ds not in data or 'freq_selectivity' not in data[ds]:
            ax.set_visible(False)
            continue
        r   = data[ds]['freq_selectivity']
        hrf = np.array(r['pref_freq_hrf_array'])
        lif = np.array(r['pref_freq_lif_array'])
        w_h = np.ones_like(hrf) / len(hrf)
        w_l = np.ones_like(lif) / len(lif)

        ax.hist(hrf, bins=bins, weights=w_h, color=HRF_COLOR, alpha=0.40)
        ax.hist(lif, bins=bins, weights=w_l, color=LIF_COLOR, alpha=0.40)
        ax.hist(hrf, bins=bins, weights=w_h, histtype='step', color=HRF_COLOR)
        ax.hist(lif, bins=bins, weights=w_l, histtype='step', color=LIF_COLOR)
        ax.set_xscale('log')
        ax.set_xlim(f_min_plot, f_max_plot)
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Fraction of neurons')
            if add_panel_letter:
                ax.text(-0.35, 1.05, 'B', transform=ax.transAxes,
                        fontsize=11, fontweight='bold')

    legend_els = [
        Patch(facecolor=HRF_COLOR, alpha=0.7, label='s-RON'),
        Patch(facecolor=LIF_COLOR, alpha=0.7, label='LIF-RC'),
    ]
    axes[-1].legend(handles=legend_els, loc='upper right', frameon=False)


def plot_freq_selectivity(data, out_dir, style='paper'):
    """Standalone preferred-frequency histograms (appendix)."""
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'freq_selectivity' in data[ds]]
    if not ds_present:
        print("  WARNING: no freq_selectivity data")
        return

    f_min, f_max, f_min_plot, f_max_plot = _shared_freq_range(data, ds_present)
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)

    fig, axes = plt.subplots(1, len(ds_present), figsize=(6.5, 1.9))
    if len(ds_present) == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.22, wspace=0.10)
    _draw_freq_panels(axes, data, ds_present, bins, f_min_plot, f_max_plot,
                      add_panel_letter=True)
    fig.text(0.55, 0.02, 'Preferred frequency (Hz)', ha='center')

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'freq_selectivity_all.{ext}'))
    plt.close()
    print("  Saved: freq_selectivity_all.pdf/png")


def plot_combined(data, out_dir, style='paper'):
    """Combined figure: CKA bars (A) + freq histograms (B)."""
    ds_present_freq = [ds for ds in DATASETS
                       if ds in data and 'freq_selectivity' in data[ds]]
    fig = plt.figure(figsize=(9.5, 2.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.8], wspace=0.20)
    axA = fig.add_subplot(gs[0])
    plot_cka(data, style=style, ax=axA)

    gsB  = gs[1].subgridspec(1, 4, wspace=0.10)
    axes = [fig.add_subplot(gsB[0, i]) for i in range(4)]
    if ds_present_freq:
        f_min, f_max, f_min_plot, f_max_plot = _shared_freq_range(data, ds_present_freq)
        bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)
        _draw_freq_panels(axes, data, DATASETS, bins,
                          f_min_plot, f_max_plot, add_panel_letter=True)
    fig.text(0.63, -0.04, 'Preferred frequency (Hz)', ha='center')

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'repr_analysis_combined.{ext}'))
    plt.close()
    print("  Saved: repr_analysis_combined.pdf/png")


# =============================================================================
# APPENDIX: Probe accuracy (with CV overlay)
# NOTE: Train/test probe results are unreliable (sampling artifact confirmed
# by CV sanity check). Use only for appendix with explicit caveat.
# =============================================================================

def plot_probe_vs_cka(data, out_dir, style='paper'):
    """
    APPENDIX ONLY. Train/test gap was shown to be a sampling artifact on
    three of four datasets (CV shows HRF ≈ LIF on SHD, DVS, FordA).
    Only sMNIST shows a real HRF advantage on CV (0.817 vs 0.560).
    See paper appendix for discussion.
    """
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'cka' in data[ds] and 'probe' in data[ds]]
    if not ds_present:
        print("  WARNING: no CKA+probe data, skipping probe_vs_cka")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 2.4) if style == 'paper' else (9, 3.5))
    x     = np.arange(len(ds_present))
    width = 0.32

    cka_h  = [data[ds]['cka']['cka_hrf']          for ds in ds_present]
    cka_l  = [data[ds]['cka']['cka_lif']          for ds in ds_present]
    prob_h = [data[ds]['probe']['probe_acc_hrf']  for ds in ds_present]
    prob_l = [data[ds]['probe']['probe_acc_lif']  for ds in ds_present]
    prob_h_err = [data[ds]['probe'].get('probe_std_hrf', 0) for ds in ds_present]
    prob_l_err = [data[ds]['probe'].get('probe_std_lif', 0) for ds in ds_present]

    # Left: CKA
    axes[0].bar(x - width/2, cka_h, width, color=HRF_COLOR, label='s-RON',
                edgecolor='white', linewidth=0.5)
    axes[0].bar(x + width/2, cka_l, width, color=LIF_COLOR, label='LIF-RC',
                edgecolor='white', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[0].set_ylabel('CKA score')
    axes[0].set_title('Class-label alignment')
    axes[0].set_ylim(0, max(max(cka_h), max(cka_l)) * 1.35)
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold')

    # Right: probe (train/test bars + CV overlay)
    axes[1].bar(x - width/2, prob_h, width, yerr=prob_h_err,
                color=HRF_COLOR, edgecolor='white', linewidth=0.5,
                error_kw=dict(elinewidth=0.8, capsize=2))
    axes[1].bar(x + width/2, prob_l, width, yerr=prob_l_err,
                color=LIF_COLOR, edgecolor='white', linewidth=0.5,
                error_kw=dict(elinewidth=0.8, capsize=2))

    # CV sanity-check overlay (open circles)
    cv_h = [data[ds]['probe'].get('probe_cv_acc_hrf') for ds in ds_present]
    cv_l = [data[ds]['probe'].get('probe_cv_acc_lif') for ds in ds_present]
    if any(v is not None for v in cv_h):
        cv_h_err = [data[ds]['probe'].get('probe_cv_std_hrf', 0)
                    if cv_h[i] is not None else 0
                    for i, ds in enumerate(ds_present)]
        cv_l_err = [data[ds]['probe'].get('probe_cv_std_lif', 0)
                    if cv_l[i] is not None else 0
                    for i, ds in enumerate(ds_present)]
        axes[1].errorbar(x - width/2, cv_h, yerr=cv_h_err,
                         fmt='o', mfc='white', mec='black', ms=4,
                         lw=0.8, capsize=2, zorder=5, label='5-fold CV')
        axes[1].errorbar(x + width/2, cv_l, yerr=cv_l_err,
                         fmt='o', mfc='white', mec='black', ms=4,
                         lw=0.8, capsize=2, zorder=5)
        axes[1].legend(frameon=False, loc='lower right', fontsize=6.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[1].set_ylabel('Linear-probe accuracy')
    axes[1].set_title('Downstream classifier\n(bars: train/test; circles: 5-fold CV)')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'probe_vs_cka.{ext}'))
    plt.close()
    print("  Saved: probe_vs_cka.pdf/png  [APPENDIX]")


# =============================================================================
# APPENDIX: Richness bars (PR + LUD + ASE — inconsistent across datasets)
# =============================================================================

def plot_richness_bars(data, out_dir, style='paper'):
    """
    APPENDIX ONLY. Results are inconsistent across datasets: HRF wins on
    FordA, LIF wins on sMNIST/SHD/DVS. Cannot be reported as a general
    finding. Included for completeness and honest reporting.
    """
    ds_present = [ds for ds in DATASETS if ds in data and 'eff_dim' in data[ds]]
    if not ds_present:
        print("  WARNING: no eff_dim data, skipping richness_bars")
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(7.5, 2.2) if style == 'paper' else (11, 3.5))
    x     = np.arange(len(ds_present))
    width = 0.32

    for ax, (kh, kl, title) in zip(axes, [
            ('pr_hrf',  'pr_lif',  'Participation Ratio'),
            ('lud_hrf', 'lud_lif', 'LUD (90% var)'),
            ('ase_hrf', 'ase_lif', 'Avg State Entropy'),
    ]):
        vals_h = [data[ds]['eff_dim'][kh] for ds in ds_present]
        vals_l = [data[ds]['eff_dim'][kl] for ds in ds_present]
        ax.bar(x - width/2, vals_h, width, color=HRF_COLOR, label='s-RON',
               edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, vals_l, width, color=LIF_COLOR, label='LIF-RC',
               edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in ds_present])
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)

    axes[0].legend(frameon=False)
    mode = data[ds_present[0]]['eff_dim'].get('mode', 'pooled')
    fig.text(0.5, -0.03,
             f'Computed on {mode} reservoir states — direction inconsistent '
             f'across datasets, see appendix.',
             ha='center', fontsize=6.5, style='italic', color='#666666')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'richness_bars.{ext}'))
    plt.close()
    print("  Saved: richness_bars.pdf/png  [APPENDIX]")


# =============================================================================
# APPENDIX: Memory capacity (only FordA gives reliable results)
# =============================================================================

def plot_memory_capacity_curves(data, out_dir, style='paper'):
    """
    APPENDIX ONLY. MC is dataset-specific: only FordA gives reliable results.
    On sMNIST/SHD/DVS, i.i.d. noise drives HRF out of its operating regime.
    """
    ds_present = [ds for ds in DATASETS if ds in data and 'mc' in data[ds]]
    if not ds_present:
        print("  WARNING: no MC data, skipping memory_capacity_curves")
        return

    n = len(ds_present)
    fig, axes = plt.subplots(
        1, n,
        figsize=(1.9 * n, 2.2) if style == 'paper' else (3 * n, 3.5),
        sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ax, ds) in enumerate(zip(axes, ds_present)):
        r = data[ds]['mc']
        k = np.arange(1, len(r['mc_lin_curve_hrf']) + 1)
        ax.plot(k, r['mc_lin_curve_hrf'],  color=HRF_COLOR, lw=1.2, label='s-RON linear')
        ax.plot(k, r['mc_lin_curve_lif'],  color=LIF_COLOR, lw=1.2, label='LIF-RC linear')
        ax.plot(k, r['mc_nlin_curve_hrf'], color=HRF_COLOR, lw=1.2,
                linestyle='--', label='s-RON nonlin.')
        ax.plot(k, r['mc_nlin_curve_lif'], color=LIF_COLOR, lw=1.2,
                linestyle='--', label='LIF-RC nonlin.')
        ax.set_xlabel('Delay k')
        if i == 0:
            ax.set_ylabel(r'$r^2$')
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'memory_capacity_curves.{ext}'))
    plt.close()
    print("  Saved: memory_capacity_curves.pdf/png  [APPENDIX]")


def plot_mc_summary(data, out_dir, style='paper'):
    """APPENDIX ONLY. Aggregate MC bars."""
    ds_present = [ds for ds in DATASETS if ds in data and 'mc' in data[ds]]
    if not ds_present:
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 2.4) if style == 'paper' else (9, 3.5))
    x     = np.arange(len(ds_present))
    width = 0.32

    for ax, kh, kl, title in [
            (axes[0], 'MC_linear_hrf',    'MC_linear_lif',    'Linear MC'),
            (axes[1], 'MC_nonlinear_hrf', 'MC_nonlinear_lif', 'Nonlinear MC'),
    ]:
        ax.bar(x - width/2, [data[ds]['mc'][kh] for ds in ds_present],
               width, color=HRF_COLOR, label='s-RON')
        ax.bar(x + width/2, [data[ds]['mc'][kl] for ds in ds_present],
               width, color=LIF_COLOR, label='LIF-RC')
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in ds_present])
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)

    axes[0].legend(frameon=False)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'mc_summary.{ext}'))
    plt.close()
    print("  Saved: mc_summary.pdf/png  [APPENDIX]")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate representational analysis figures.'
    )
    parser.add_argument('--results_dir', default=None,
                        help='Where JSON results live. '
                             'Default: <script_dir>/analysis_results_extended')
    parser.add_argument('--out_dir', default=None,
                        help='Where figures are written. '
                             'Default: <script_dir>/figures_extended')
    parser.add_argument('--style', default='paper',
                        choices=['paper', 'slides'])
    parser.add_argument('--cka_feature', default='mean',
                        choices=['mean', 'rms', 'final'])
    parser.add_argument('--appendix', action='store_true',
                        help='Also generate appendix figures '
                             '(probe, freq histograms, richness, MC).')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Figures to skip by name: cka, bandpass, '
                             'scatter, combined, probe, freq, richness, '
                             'mc_curves, mc_summary')
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(
        script_dir, 'analysis_results_extended')
    out_dir     = args.out_dir or os.path.join(
        script_dir, 'figures_extended')
    os.makedirs(out_dir, exist_ok=True)

    set_style(args.style)

    print(f"\nLoading results from: {results_dir}")
    data = load_results(results_dir, cka_feature=args.cka_feature)
    if not data:
        print("No data loaded — check --results_dir path.")
        return

    print(f"\nGenerating figures in: {out_dir}")
    print(f"Mode: {'main paper + appendix' if args.appendix else 'main paper only'}")

    # ---- Main paper figures (always produced) ----
    if 'cka'      not in args.skip:
        plot_cka(data, out_dir, args.style)
    if 'bandpass' not in args.skip:
        plot_bandpass_summary(data, out_dir, args.style)
    if 'scatter'  not in args.skip:
        plot_freq_q_scatter(data, out_dir, args.style)
    if 'combined' not in args.skip:
        plot_combined(data, out_dir, args.style)

    # ---- Appendix figures (opt-in with --appendix) ----
    if args.appendix:
        if 'probe'     not in args.skip:
            plot_probe_vs_cka(data, out_dir, args.style)
        if 'freq'      not in args.skip:
            plot_freq_selectivity(data, out_dir, args.style)
        if 'richness'  not in args.skip:
            plot_richness_bars(data, out_dir, args.style)
        if 'mc_curves' not in args.skip:
            plot_memory_capacity_curves(data, out_dir, args.style)
        if 'mc_summary' not in args.skip:
            plot_mc_summary(data, out_dir, args.style)

    print("\nDone.")


if __name__ == '__main__':
    main()
'''
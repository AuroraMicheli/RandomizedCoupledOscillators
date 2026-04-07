"""
Generate publication-quality figures for the representational analysis.

Produces two figures:
  A) cka_all_datasets.pdf  — grouped bar chart of CKA scores across datasets
  B) freq_selectivity_all.pdf — overlaid frequency histograms, 1x4 grid

Run after analyze_representations.py has been executed for all 4 datasets.
Reads from analysis_results/<dataset>/analysis_summary_<dataset>.json

Usage:
    python plot_repr_analysis.py
    python plot_repr_analysis.py --results_dir /path/to/analysis_results
    python plot_repr_analysis.py --style paper   # publication style
    python plot_repr_analysis.py --style slides  # larger fonts
"""

"""
Generate publication-quality figures for the representational analysis.

Produces:

A) cka_all_datasets.pdf/png
B) freq_selectivity_all.pdf/png
Combined:
repr_analysis_combined.pdf/png
"""

"""
Generate publication-quality figures for the representational analysis.

Produces:
  figures/cka_all_datasets.pdf/png
  figures/freq_selectivity_all.pdf/png
  figures/repr_analysis_combined.pdf/png

Usage:
    python plot_repr_analysis.py
    python plot_repr_analysis.py --cka_feature rms
    python plot_repr_analysis.py --results_dir /path/to/analysis_results --out_dir /path/to/figures
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
    """
    Load per-dataset JSON summaries.
    Tries the filename with the cka_feature suffix first
    (analysis_summary_{ds}_{cka_feature}.json), then falls back to the
    old filename without the suffix (analysis_summary_{ds}.json).
    Prints a clear message so you always know which file was loaded.
    """
    data = {}
    for ds in DATASETS:
        # Primary: new filename with feature suffix
        path_new = os.path.join(
            results_dir, ds,
            f'analysis_summary_{ds}_{cka_feature}.json'
        )
        # Fallback: old filename without suffix
        path_old = os.path.join(
            results_dir, ds,
            f'analysis_summary_{ds}.json'
        )

        if os.path.exists(path_new):
            with open(path_new) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_new}")
        elif os.path.exists(path_old):
            with open(path_old) as f:
                data[ds] = json.load(f)
            print(f"  Loaded [{ds}]: {path_old}  (old filename, no feature suffix)")
        else:
            print(f"  WARNING [{ds}]: neither {path_new} nor {path_old} found — skipping")

    return data


# =============================================================================
# Shared frequency axis
# =============================================================================

def _shared_freq_range(data, ds_present):
    """
    Use f_min / f_max stored in each result JSON if available (set by the
    analysis script).  Falls back to distribution percentiles if missing.
    Using the stored sweep range is more reliable than percentiles because
    the distributions can be heavily skewed.
    """
    f_mins, f_maxs = [], []
    for ds in ds_present:
        r = data[ds]['freq_selectivity']
        if 'f_min' in r and 'f_max' in r:
            f_mins.append(r['f_min'])
            f_maxs.append(r['f_max'])
        else:
            # fallback: use distribution values
            vals = np.array(
                r['pref_freq_hrf_array'] + r['pref_freq_lif_array']
            )
            f_mins.append(np.percentile(vals, 1) * 0.5)
            f_maxs.append(np.percentile(vals, 99) * 2.0)
    return min(f_mins), max(f_maxs)


# =============================================================================
# Panel A: CKA bar chart
# =============================================================================

def plot_cka(data, out_dir=None, style='paper', ax=None):
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'cka' in data[ds]]
    if not ds_present:
        print("  WARNING: no CKA data found, skipping panel A")
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(3.2, 2.4) if style == 'paper' else (5, 3.5))

    x     = np.arange(len(ds_present))
    width = 0.32

    cka_hrf = [data[ds]['cka']['cka_hrf'] for ds in ds_present]
    cka_lif = [data[ds]['cka']['cka_lif'] for ds in ds_present]

    ax.bar(x - width/2, cka_hrf, width,
           color=HRF_COLOR, label='s-RON (HRF)',
           edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, cka_lif, width,
           color=LIF_COLOR,  label='LIF-RC',
           edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    ax.set_ylabel('CKA score')
    ax.set_ylim(0, max(max(cka_hrf), max(cka_lif)) * 1.35)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25)

    # Panel letter — visible in both standalone and combined modes
    ax.text(-0.18, 1.05, 'A',
            transform=ax.transAxes, fontsize=11, fontweight='bold')

    if standalone:
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(out_dir, f'cka_all_datasets.{ext}'))
        plt.close()
        print("  Saved: cka_all_datasets.pdf/png")


# =============================================================================
# Panel B: Frequency selectivity histograms
# =============================================================================

def _draw_freq_panels(axes, data, ds_list, bins, xmin, xmax,
                      add_panel_letter=False):
    """
    Draw frequency histograms into a list of axes, one per dataset.
    Shared logic used by both the standalone and combined figure functions.
    """
    for i, (ax, ds) in enumerate(zip(axes, ds_list)):
        if ds not in data or 'freq_selectivity' not in data[ds]:
            ax.set_visible(False)
            continue

        r   = data[ds]['freq_selectivity']
        hrf = np.array(r['pref_freq_hrf_array'])
        lif = np.array(r['pref_freq_lif_array'])

        w_hrf = np.ones_like(hrf) / len(hrf)
        w_lif = np.ones_like(lif) / len(lif)

        # Filled bars
        ax.hist(hrf, bins=bins, weights=w_hrf, color=HRF_COLOR, alpha=0.40)
        ax.hist(lif, bins=bins, weights=w_lif, color=LIF_COLOR, alpha=0.40)
        # Outlines
        ax.hist(hrf, bins=bins, weights=w_hrf, histtype='step', color=HRF_COLOR)
        ax.hist(lif, bins=bins, weights=w_lif, histtype='step', color=LIF_COLOR)
        # Mean lines
        ax.axvline(r['pref_freq_hrf_mean'], color=HRF_COLOR, linestyle='--', lw=1.0)
        ax.axvline(r['pref_freq_lif_mean'], color=LIF_COLOR, linestyle='--', lw=1.0)

        ax.set_xscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.set_yticks([])
        ax.set_xlabel('Pref. freq. (Hz)')

        if i == 0:
            ax.set_ylabel('Fraction of neurons')
            if add_panel_letter:
                ax.text(-0.35, 1.05, 'B',
                        transform=ax.transAxes, fontsize=11, fontweight='bold')

    # Legend on last visible axis
    legend_els = [
        Patch(facecolor=HRF_COLOR, alpha=0.7, label='s-RON (HRF)'),
        Patch(facecolor=LIF_COLOR, alpha=0.7, label='LIF-RC'),
    ]
    axes[-1].legend(handles=legend_els, loc='upper right', frameon=False)


def plot_freq_selectivity(data, out_dir, style='paper'):
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'freq_selectivity' in data[ds]]
    if not ds_present:
        print("  WARNING: no freq_selectivity data found, skipping panel B")
        return

    f_min, f_max = _shared_freq_range(data, ds_present)
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)

    fig, axes = plt.subplots(
        1, len(ds_present),
        figsize=(6.5, 1.7),
        constrained_layout=True
    )
    if len(ds_present) == 1:
        axes = [axes]

    _draw_freq_panels(axes, data, ds_present, bins, f_min, f_max,
                      add_panel_letter=True)

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'freq_selectivity_all.{ext}'))
    plt.close()
    print("  Saved: freq_selectivity_all.pdf/png")


# =============================================================================
# Combined figure (A + B side by side)
# =============================================================================

def plot_combined(data, out_dir, style='paper'):
    ds_present_freq = [ds for ds in DATASETS
                       if ds in data and 'freq_selectivity' in data[ds]]

    fig = plt.figure(figsize=(9.5, 2.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.8], wspace=0.20)

    # --- Panel A ---
    axA = fig.add_subplot(gs[0])
    plot_cka(data, style=style, ax=axA)

    # --- Panel B ---
    gsB  = gs[1].subgridspec(1, 4, wspace=0.10)
    axes = [fig.add_subplot(gsB[0, i]) for i in range(4)]

    if ds_present_freq:
        f_min, f_max = _shared_freq_range(data, ds_present_freq)
        bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)
        _draw_freq_panels(axes, data, DATASETS, bins, f_min, f_max,
                          add_panel_letter=True)

    # Shared x-axis label for panel B
    fig.text(0.63, -0.02, 'Preferred frequency (Hz)', ha='center')

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'repr_analysis_combined.{ext}'))
    plt.close()
    print("  Saved: repr_analysis_combined.pdf/png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None,
                        help='Path to analysis_results/ directory. '
                             'Defaults to analysis_results/ next to this script.')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory. Defaults to figures/ next to this script.')
    parser.add_argument('--style', default='paper', choices=['paper', 'slides'])
    parser.add_argument('--cka_feature', default='mean',
                        choices=['mean', 'rms', 'final'],
                        help='Which CKA feature variant to load '
                             '(must match what was used when running the analysis).')
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(script_dir, 'analysis_results')
    out_dir     = args.out_dir     or os.path.join(script_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    set_style(args.style)

    print(f"\nLoading results from: {results_dir}")
    print(f"CKA feature: {args.cka_feature}")
    data = load_results(results_dir, cka_feature=args.cka_feature)

    if not data:
        print("No data loaded — check your results_dir path.")
        return

    print(f"\nGenerating figures in: {out_dir}")
    plot_cka(data, out_dir, args.style)
    plot_freq_selectivity(data, out_dir, args.style)
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
import matplotlib.ticker as ticker

# =============================================================================
# Style
# =============================================================================

def set_style(style='paper'):
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        # Font
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':        8  if style == 'paper' else 11,
        'axes.titlesize':   9  if style == 'paper' else 12,
        'axes.labelsize':   8  if style == 'paper' else 11,
        'xtick.labelsize':  7  if style == 'paper' else 10,
        'ytick.labelsize':  7  if style == 'paper' else 10,
        'legend.fontsize':  7  if style == 'paper' else 10,
        # Lines
        'axes.linewidth':   0.8,
        'xtick.major.width':0.8,
        'ytick.major.width':0.8,
        'lines.linewidth':  1.0,
        # Layout
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         False,
        # PDF output
        'pdf.fonttype':     42,   # TrueType fonts in PDF (Overleaf compatible)
        'ps.fonttype':      42,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
        'savefig.pad_inches': 0.02,
    })


# =============================================================================
# Colors and dataset labels
# =============================================================================

HRF_COLOR = '#2166AC'   # blue
LIF_COLOR = '#D6604D'   # red-orange
# Both chosen from ColorBrewer RdBu diverging palette — colorblind safe

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

def load_results(results_dir):
    """Load JSON summaries for all datasets. Returns dict keyed by dataset."""
    data = {}
    for ds in DATASETS:
        path = os.path.join(results_dir, ds,
                            f'analysis_summary_{ds}.json')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {ds}")
            continue
        with open(path) as f:
            data[ds] = json.load(f)
    return data


# =============================================================================
# Figure A: CKA grouped bar chart
# =============================================================================

def plot_cka(data, out_dir, style='paper'):
    datasets_present = [ds for ds in DATASETS if ds in data
                        and 'cka' in data[ds]]
    if not datasets_present:
        print("No CKA data found, skipping Figure A")
        return

    n = len(datasets_present)
    x = np.arange(n)
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.2, 2.4) if style == 'paper' else (5, 3.5))

    cka_hrf = [data[ds]['cka']['cka_hrf'] for ds in datasets_present]
    cka_lif = [data[ds]['cka']['cka_lif'] for ds in datasets_present]

    bars_hrf = ax.bar(x - width/2, cka_hrf, width,
                      color=HRF_COLOR, label='s-RON',
                      edgecolor='white', linewidth=0.5, zorder=3)
    bars_lif = ax.bar(x + width/2, cka_lif, width,
                      color=LIF_COLOR, label='LIF-RC',
                      edgecolor='white', linewidth=0.5, zorder=3)

    # Value labels on top of bars
    for bar in bars_hrf:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.004,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=5.5 if style == 'paper' else 8,
                color=HRF_COLOR, fontweight='bold')
    for bar in bars_lif:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.004,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=5.5 if style == 'paper' else 8,
                color=LIF_COLOR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets_present])
    ax.set_ylabel('CKA (reservoir states vs. labels)')
    ax.set_ylim(0, max(max(cka_hrf), max(cka_lif)) * 1.35)
    ax.set_title('Linear CKA — class alignment\nwithout readout training')
    ax.legend(loc='upper right', frameon=False,
              handlelength=1.2, handleheight=0.8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, zorder=0)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'cka_all_datasets.{ext}')
        plt.savefig(path)
        print(f"  Saved: {path}")
    plt.close()


# =============================================================================
# Figure B: Frequency selectivity histograms (1x4 grid, overlaid)
# =============================================================================



def plot_freq_selectivity(data, out_dir, style='paper'):
    """
    Plot overlaid histograms of per-neuron preferred frequencies.
    Uses pref_freq_hrf_array / pref_freq_lif_array saved in JSON.
    All panels share the same x-axis range for direct comparison.
    """
    datasets_present = [ds for ds in DATASETS if ds in data
                        and 'freq_selectivity' in data[ds]]
    if not datasets_present:
        print("No freq_selectivity data found, skipping Figure B")
        return

    n = len(datasets_present)
    fig_w = 6.5 if style == 'paper' else 10.0
    fig_h = 1.7 if style == 'paper' else 2.6
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Compute shared x-axis range from all per-neuron values
    all_vals = []
    for ds in datasets_present:
        r = data[ds]['freq_selectivity']
        all_vals += r.get('pref_freq_hrf_array', [])
        all_vals += r.get('pref_freq_lif_array', [])
        all_vals += r.get('freqs_hrf', [])
        all_vals += r.get('freqs_lif', [])
    pos_vals = [v for v in all_vals if v > 0]
    x_min = min(pos_vals) * 0.7 if pos_vals else 0.005
    x_max = max(pos_vals) * 1.4 if pos_vals else 5.0
    shared_bins = np.logspace(np.log10(x_min), np.log10(x_max), 30)

    for i, (ax, ds) in enumerate(zip(axes, datasets_present)):
        r        = data[ds]['freq_selectivity']
        pref_hrf = np.array(r.get('pref_freq_hrf_array', []))
        pref_lif = np.array(r.get('pref_freq_lif_array', []))
        hrf_mean = r['pref_freq_hrf_mean']
        lif_mean = r['pref_freq_lif_mean']

        if len(pref_hrf) == 0 or len(pref_lif) == 0:
            ax.text(0.5, 0.5, 'No array data\nRe-run analysis',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=6, color='red')
            ax.set_title(DATASET_LABELS_SHORT[ds])
            continue

        # Filled histograms + step outlines
        ax.hist(pref_hrf, bins=shared_bins, density=True,
                color=HRF_COLOR, alpha=0.40, edgecolor='none')
        ax.hist(pref_lif, bins=shared_bins, density=True,
                color=LIF_COLOR, alpha=0.40, edgecolor='none')
        ax.hist(pref_hrf, bins=shared_bins, density=True,
                histtype='step', color=HRF_COLOR, linewidth=1.1)
        ax.hist(pref_lif, bins=shared_bins, density=True,
                histtype='step', color=LIF_COLOR, linewidth=1.1)

        # Dashed vertical lines at means
        ax.axvline(hrf_mean, color=HRF_COLOR, linewidth=0.9,
                   linestyle='--', alpha=0.9)
        ax.axvline(lif_mean, color=LIF_COLOR, linewidth=0.9,
                   linestyle='--', alpha=0.9)

        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Pref. freq. (Hz)',
                      fontsize=7 if style == 'paper' else 10)
        ax.set_title(DATASET_LABELS_SHORT[ds],
                     fontsize=8 if style == 'paper' else 11)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

        if i == 0:
            ax.set_ylabel('Norm. density',
                          fontsize=7 if style == 'paper' else 10)

    # Legend on last panel
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=HRF_COLOR, alpha=0.6, label='s-RON'),
                  Patch(facecolor=LIF_COLOR, alpha=0.6, label='LIF-RC')]
    axes[-1].legend(handles=legend_els, loc='upper right', frameon=False,
                    handlelength=1.0,
                    fontsize=6 if style == 'paper' else 9)

    fig.suptitle('Frequency tuning of reservoir neurons',
                 fontsize=9 if style == 'paper' else 12,
                 fontweight='bold')

    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'freq_selectivity_all.{ext}')
        plt.savefig(path)
        print(f"  Saved: {path}")
    plt.close()

def plot_combined(data, out_dir, style='paper'):
    datasets_present = [ds for ds in DATASETS if ds in data]
    if not datasets_present:
        return

    from matplotlib.patches import Patch

    # Font sizes — same for both panels
    fs_title  = 9  if style == 'paper' else 12   # subplot titles
    fs_label  = 8  if style == 'paper' else 11   # axis labels
    fs_tick   = 8  if style == 'paper' else 10   # tick labels
    fs_legend = 8  if style == 'paper' else 10   # legend
    fs_panel  = 11 if style == 'paper' else 13   # A / B panel labels

    fig_w = 9.5  if style == 'paper' else 13.0
    fig_h = 2.8  if style == 'paper' else 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Tighter wspace so both panels are closer and B has more horizontal room
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1, 2.8],
        left=0.07, right=0.98,
        bottom=0.24, top=0.88,
        wspace=0.20,
    )

    # ── Panel A: CKA bar chart ────────────────────────────────────────────────
    ax_cka = fig.add_subplot(gs[0])

    ds_cka  = [ds for ds in datasets_present if 'cka' in data[ds]]
    x       = np.arange(len(ds_cka))
    width   = 0.32
    cka_hrf = [data[ds]['cka']['cka_hrf'] for ds in ds_cka]
    cka_lif = [data[ds]['cka']['cka_lif'] for ds in ds_cka]

    ax_cka.bar(x - width/2, cka_hrf, width, color=HRF_COLOR,
               label='s-RON', edgecolor='white', linewidth=0.4, zorder=3)
    ax_cka.bar(x + width/2, cka_lif, width, color=LIF_COLOR,
               label='LIF-RC', edgecolor='white', linewidth=0.4, zorder=3)

    ax_cka.set_xticks(x)
    ax_cka.set_xticklabels([DATASET_LABELS[ds] for ds in ds_cka],
                            fontsize=fs_tick)
    ax_cka.set_ylabel('CKA score', fontsize=fs_label)
    ax_cka.set_ylim(0, max(max(cka_hrf), max(cka_lif)) * 1.45)
    ax_cka.tick_params(axis='y', labelsize=fs_tick)
    # No title — explained in caption
    ax_cka.legend(loc='upper left', frameon=False,
                  handlelength=1.0, handleheight=0.7,
                  fontsize=fs_legend)
    ax_cka.grid(axis='y', alpha=0.2, linewidth=0.5, zorder=0)
    ax_cka.text(-0.22, 1.06, 'A', transform=ax_cka.transAxes,
                fontsize=fs_panel, fontweight='bold')

    # ── Panel B: frequency histograms ─────────────────────────────────────────
    ds_freq = [ds for ds in datasets_present if 'freq_selectivity' in data[ds]]
    nf      = len(ds_freq)
    gs_inner = gs[1].subgridspec(1, nf, wspace=0.10)
    axes_freq = [fig.add_subplot(gs_inner[0, i]) for i in range(nf)]

    # Shared x range
    all_vals = []
    for ds in ds_freq:
        r = data[ds]['freq_selectivity']
        all_vals += r.get('pref_freq_hrf_array', [])
        all_vals += r.get('pref_freq_lif_array', [])
        all_vals += r.get('freqs_hrf', [])
        all_vals += r.get('freqs_lif', [])
    pos_vals = [v for v in all_vals if v > 0]
    x_min = min(pos_vals) * 0.7 if pos_vals else 0.005
    x_max = max(pos_vals) * 1.4 if pos_vals else 5.0
    sh_bins = np.logspace(np.log10(x_min), np.log10(x_max), 30)

    for i, (ax, ds) in enumerate(zip(axes_freq, ds_freq)):
        r        = data[ds]['freq_selectivity']
        pref_hrf = np.array(r.get('pref_freq_hrf_array', []))
        pref_lif = np.array(r.get('pref_freq_lif_array', []))
        hrf_mean = r['pref_freq_hrf_mean']
        lif_mean = r['pref_freq_lif_mean']

        if len(pref_hrf) > 0 and len(pref_lif) > 0:
            ax.hist(pref_hrf, bins=sh_bins, density=True,
                    color=HRF_COLOR, alpha=0.40, edgecolor='none')
            ax.hist(pref_lif, bins=sh_bins, density=True,
                    color=LIF_COLOR, alpha=0.40, edgecolor='none')
            ax.hist(pref_hrf, bins=sh_bins, density=True,
                    histtype='step', color=HRF_COLOR, linewidth=1.0)
            ax.hist(pref_lif, bins=sh_bins, density=True,
                    histtype='step', color=LIF_COLOR, linewidth=1.0)
            ax.axvline(hrf_mean, color=HRF_COLOR, linewidth=0.8,
                       linestyle='--', alpha=0.9)
            ax.axvline(lif_mean, color=LIF_COLOR, linewidth=0.8,
                       linestyle='--', alpha=0.9)
        else:
            ax.text(0.5, 0.5, 'Re-run\nanalysis',
                    transform=ax.transAxes, ha='center',
                    fontsize=fs_tick, color='red')

        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_title(DATASET_LABELS_SHORT[ds], pad=3, fontsize=fs_title)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=fs_tick)

        if i == 0:
            ax.set_ylabel('Norm. density', fontsize=fs_label)
            ax.text(-0.38, 1.06, 'B', transform=ax.transAxes,
                    fontsize=fs_panel, fontweight='bold')

    # Single shared x-axis label centred under all four histogram panels
    b_left  = axes_freq[0].get_position().x0
    b_right = axes_freq[-1].get_position().x1
    b_cx    = (b_left + b_right) / 2.0
    b_bot   = axes_freq[0].get_position().y0
    fig.text(b_cx, b_bot - 0.10, 'Preferred frequency (Hz)',
             ha='center', va='top', fontsize=fs_label)

    # Legend inside the last panel (DVS Gesture), top-right corner.
    # Data on all datasets piles up on the left of the x-axis so the
    # right side is always empty — no overlap with histogram bars.
    legend_els = [
        Patch(facecolor=HRF_COLOR, alpha=0.7, label='s-RON'),
        Patch(facecolor=LIF_COLOR, alpha=0.7, label='LIF-RC'),
    ]
    axes_freq[-1].legend(handles=legend_els,
                         loc='upper right', frameon=False,
                         handlelength=1.0, handleheight=0.8,
                         fontsize=fs_legend)

    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'repr_analysis_combined.{ext}')
        plt.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to analysis_results/ directory. '
                             'Default: analysis_results/ next to this script.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to save figures. '
                             'Default: figures/ next to this script.')
    parser.add_argument('--style', type=str, default='paper',
                        choices=['paper', 'slides'],
                        help='paper: compact for NeurIPS column width. '
                             'slides: larger fonts.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = args.results_dir or os.path.join(script_dir, 'analysis_results')
    out_dir     = args.out_dir     or os.path.join(script_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    set_style(args.style)

    print(f"Loading results from: {results_dir}")
    data = load_results(results_dir)
    print(f"Loaded data for: {list(data.keys())}")

    print("\nGenerating Figure A: CKA bar chart...")
    plot_cka(data, out_dir, style=args.style)

    print("\nGenerating Figure B: Frequency selectivity...")
    plot_freq_selectivity(data, out_dir, style=args.style)

    print("\nGenerating combined figure (A+B)...")
    plot_combined(data, out_dir, style=args.style)

    print(f"\nAll figures saved to: {out_dir}")
    print("Include in Overleaf with:")
    print("  \\includegraphics[width=\\textwidth]{figures/repr_analysis_combined.pdf}")


if __name__ == '__main__':
    main()

'''












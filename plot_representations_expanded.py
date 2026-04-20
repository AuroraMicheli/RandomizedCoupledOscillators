"""
Generate publication-quality figures for the representational analysis.

Produces (from analysis_results/<dataset>/analysis_summary_<dataset>_<feat>.json):

  figures/cka_all_datasets.{pdf,png}            — existing, unchanged
  figures/freq_selectivity_all.{pdf,png}        — existing, unchanged
  figures/repr_analysis_combined.{pdf,png}      — existing, unchanged

  figures/probe_vs_cka.{pdf,png}                — NEW: class discriminability (CKA + linear probe)
  figures/memory_capacity_curves.{pdf,png}      — NEW: linear + nonlinear MC forgetting curves
  figures/mc_summary.{pdf,png}                  — NEW: aggregate MC bars (optional)
  figures/freq_q_scatter.{pdf,png}              — NEW: (f_pref, Q) scatter per neuron
  figures/richness_bars.{pdf,png}               — NEW: PR + LUD + ASE across datasets

Usage:
    python plot_repr_analysis.py
    python plot_repr_analysis.py --cka_feature rms
    python plot_repr_analysis.py --results_dir /path --out_dir /path
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
# Style (UNCHANGED from your original)
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
# Shared frequency axis helper (UNCHANGED)
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
    f_min = min(f_mins)
    f_max = max(f_maxs)
    f_min_plot = f_min / 3.0
    f_max_plot = f_max
    return f_min, f_max, f_min_plot, f_max_plot


# =============================================================================
# Panel A: CKA bar chart (UNCHANGED)
# =============================================================================

def plot_cka(data, out_dir=None, style='paper', ax=None):
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
           color=LIF_COLOR,  label='LIF-RC', edgecolor='white', linewidth=0.5)
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
# Panel B: Frequency histograms (UNCHANGED)
# =============================================================================

def _draw_freq_panels(axes, data, ds_list, bins, f_min_plot, f_max_plot,
                      add_panel_letter=False):
    for i, (ax, ds) in enumerate(zip(axes, ds_list)):
        if ds not in data or 'freq_selectivity' not in data[ds]:
            ax.set_visible(False)
            continue
        r   = data[ds]['freq_selectivity']
        hrf = np.array(r['pref_freq_hrf_array'])
        lif = np.array(r['pref_freq_lif_array'])
        w_hrf = np.ones_like(hrf) / len(hrf)
        w_lif = np.ones_like(lif) / len(lif)
        ax.hist(hrf, bins=bins, weights=w_hrf, color=HRF_COLOR, alpha=0.40)
        ax.hist(lif, bins=bins, weights=w_lif, color=LIF_COLOR, alpha=0.40)
        ax.hist(hrf, bins=bins, weights=w_hrf, histtype='step', color=HRF_COLOR)
        ax.hist(lif, bins=bins, weights=w_lif, histtype='step', color=LIF_COLOR)
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


# =============================================================================
# Combined A+B figure (UNCHANGED)
# =============================================================================

def plot_combined(data, out_dir, style='paper'):
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
        _draw_freq_panels(axes, data, DATASETS, bins, f_min_plot, f_max_plot,
                          add_panel_letter=True)
    fig.text(0.63, -0.04, 'Preferred frequency (Hz)', ha='center')

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'repr_analysis_combined.{ext}'))
    plt.close()
    print("  Saved: repr_analysis_combined.pdf/png")


# =============================================================================
# NEW Panel: CKA + Linear Probe side-by-side
# =============================================================================

def plot_probe_vs_cka(data, out_dir, style='paper'):
    """
    Two bar groups per dataset: CKA (left) and probe accuracy (right).
    Confirms that the CKA advantage translates to actual downstream
    classifier accuracy, not just a similarity-matrix artifact.
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

    cka_h  = [data[ds]['cka']['cka_hrf']       for ds in ds_present]
    cka_l  = [data[ds]['cka']['cka_lif']       for ds in ds_present]
    prob_h = [data[ds]['probe']['probe_acc_hrf'] for ds in ds_present]
    prob_l = [data[ds]['probe']['probe_acc_lif'] for ds in ds_present]
    prob_h_err = [data[ds]['probe'].get('probe_std_hrf', 0) for ds in ds_present]
    prob_l_err = [data[ds]['probe'].get('probe_std_lif', 0) for ds in ds_present]

    # --- Left: CKA ---
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

    # --- Right: Linear probe accuracy ---
    axes[1].bar(x - width/2, prob_h, width, yerr=prob_h_err,
                color=HRF_COLOR, edgecolor='white', linewidth=0.5,
                error_kw=dict(elinewidth=0.8, capsize=2))
    axes[1].bar(x + width/2, prob_l, width, yerr=prob_l_err,
                color=LIF_COLOR, edgecolor='white', linewidth=0.5,
                error_kw=dict(elinewidth=0.8, capsize=2))

    # CV sanity-check overlay: if the JSON contains probe_cv_acc_*, add
    # open-circle markers (with error bars) to show the CV estimate
    # alongside each train/test bar. Close agreement = robust result.
    cv_h = [data[ds]['probe'].get('probe_cv_acc_hrf') for ds in ds_present]
    cv_l = [data[ds]['probe'].get('probe_cv_acc_lif') for ds in ds_present]
    if any(v is not None for v in cv_h):
        cv_h_err = [data[ds]['probe'].get('probe_cv_std_hrf', 0)
                    if data[ds]['probe'].get('probe_cv_acc_hrf') is not None else 0
                    for ds in ds_present]
        cv_l_err = [data[ds]['probe'].get('probe_cv_std_lif', 0)
                    if data[ds]['probe'].get('probe_cv_acc_lif') is not None else 0
                    for ds in ds_present]
        # Plot as open circles just above each bar's location
        axes[1].errorbar(x - width/2, cv_h, yerr=cv_h_err,
                         fmt='o', mfc='white', mec='black', ms=4, lw=0.8,
                         capsize=2, zorder=5, label='CV sanity check')
        axes[1].errorbar(x + width/2, cv_l, yerr=cv_l_err,
                         fmt='o', mfc='white', mec='black', ms=4, lw=0.8,
                         capsize=2, zorder=5)
        axes[1].legend(frameon=False, loc='lower right')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[1].set_ylabel('Linear-probe accuracy')
    axes[1].set_title('Downstream classifier')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'probe_vs_cka.{ext}'))
    plt.close()
    print("  Saved: probe_vs_cka.pdf/png")


# =============================================================================
# NEW Panel: Memory Capacity curves
# =============================================================================

def plot_memory_capacity_curves(data, out_dir, style='paper'):
    """
    One row of subpanels (one per dataset), with linear and nonlinear MC
    forgetting curves overlaid for both models.

    Key asset: dataset-independent characterization of reservoir dynamics.
    """
    ds_present = [ds for ds in DATASETS if ds in data and 'mc' in data[ds]]
    if not ds_present:
        print("  WARNING: no MC data, skipping memory_capacity_curves")
        return

    n = len(ds_present)
    fig, axes = plt.subplots(
        1, n, figsize=(1.9 * n, 2.2) if style == 'paper' else (3 * n, 3.5),
        sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ax, ds) in enumerate(zip(axes, ds_present)):
        r = data[ds]['mc']
        k = np.arange(1, len(r['mc_lin_curve_hrf']) + 1)
        # Linear MC: solid lines
        ax.plot(k, r['mc_lin_curve_hrf'],  color=HRF_COLOR, lw=1.2,
                label='s-RON linear')
        ax.plot(k, r['mc_lin_curve_lif'],  color=LIF_COLOR, lw=1.2,
                label='LIF-RC linear')
        # Nonlinear MC: dashed
        ax.plot(k, r['mc_nlin_curve_hrf'], color=HRF_COLOR, lw=1.2,
                linestyle='--', label='s-RON nonlin.')
        ax.plot(k, r['mc_nlin_curve_lif'], color=LIF_COLOR, lw=1.2,
                linestyle='--', label='LIF-RC nonlin.')
        ax.set_xlabel('Delay k')
        if i == 0:
            ax.set_ylabel(r'$r^2$ (forgetting curve)')
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=4, frameon=False)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'memory_capacity_curves.{ext}'))
    plt.close()
    print("  Saved: memory_capacity_curves.pdf/png")


def plot_mc_summary(data, out_dir, style='paper'):
    """
    Aggregate bar chart: total linear MC + total nonlinear MC per dataset.
    Summary-at-a-glance alternative to the full curves.
    """
    ds_present = [ds for ds in DATASETS if ds in data and 'mc' in data[ds]]
    if not ds_present:
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 2.4) if style == 'paper' else (9, 3.5),
        sharey=False)
    x     = np.arange(len(ds_present))
    width = 0.32

    lin_h  = [data[ds]['mc']['MC_linear_hrf']    for ds in ds_present]
    lin_l  = [data[ds]['mc']['MC_linear_lif']    for ds in ds_present]
    nlin_h = [data[ds]['mc']['MC_nonlinear_hrf'] for ds in ds_present]
    nlin_l = [data[ds]['mc']['MC_nonlinear_lif'] for ds in ds_present]

    axes[0].bar(x - width/2, lin_h, width, color=HRF_COLOR, label='s-RON')
    axes[0].bar(x + width/2, lin_l, width, color=LIF_COLOR, label='LIF-RC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[0].set_ylabel('Linear MC')
    axes[0].set_title('Linear memory capacity')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width/2, nlin_h, width, color=HRF_COLOR)
    axes[1].bar(x + width/2, nlin_l, width, color=LIF_COLOR)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in ds_present])
    axes[1].set_ylabel('Nonlinear MC')
    axes[1].set_title('Nonlinear memory capacity')
    axes[1].grid(axis='y', alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'mc_summary.{ext}'))
    plt.close()
    print("  Saved: mc_summary.pdf/png")


# =============================================================================
# NEW Panel: Q-factor vs preferred frequency scatter
# =============================================================================

def plot_freq_q_scatter(data, out_dir, style='paper'):
    """
    Per-neuron scatter: x = preferred input frequency (log), y = Q-factor.
    Colored by model.

    HRF neurons form a cloud at moderate frequencies with high Q (sharp
    band-pass filters). LIF neurons collapse to a line at low frequencies
    with Q ≈ 0 (pure integrators, no resonance). This visualization kills
    the hyperparameter confound of the histogram version because Q is
    dimensionless and directly comparable across models.
    """
    ds_present = [ds for ds in DATASETS
                  if ds in data
                  and 'freq_selectivity' in data[ds]
                  and 'q_hrf_array' in data[ds]['freq_selectivity']]
    if not ds_present:
        print("  WARNING: no Q-factor arrays, skipping freq_q_scatter "
              "(re-run analysis script to populate)")
        return

    n = len(ds_present)
    fig, axes = plt.subplots(
        1, n, figsize=(1.9 * n, 2.2) if style == 'paper' else (3 * n, 3.5))
    if n == 1:
        axes = [axes]

    # Shared y-limit across datasets for fair comparison
    all_q = []
    for ds in ds_present:
        r = data[ds]['freq_selectivity']
        all_q.extend(r['q_hrf_array'])
        all_q.extend(r['q_lif_array'])
    q_max = np.percentile([q for q in all_q if q > 0], 99) if any(q > 0 for q in all_q) else 5.0
    q_max = max(q_max, 1.0) * 1.1

    for i, (ax, ds) in enumerate(zip(axes, ds_present)):
        r = data[ds]['freq_selectivity']
        f_hrf = np.array(r['pref_freq_hrf_array'])
        f_lif = np.array(r['pref_freq_lif_array'])
        q_hrf = np.array(r['q_hrf_array'])
        q_lif = np.array(r['q_lif_array'])

        # Plot LIF first (usually sits at Q≈0, HRF on top)
        ax.scatter(f_lif, q_lif, c=LIF_COLOR, s=4, alpha=0.35,
                   edgecolors='none', label='LIF-RC')
        ax.scatter(f_hrf, q_hrf, c=HRF_COLOR, s=4, alpha=0.45,
                   edgecolors='none', label='s-RON')

        ax.set_xscale('log')
        if 'f_min' in r and 'f_max' in r:
            ax.set_xlim(r['f_min'] / 3.0, r['f_max'])
        ax.set_ylim(-0.1, q_max)

        ax.set_xlabel('Preferred frequency (Hz)')
        if i == 0:
            ax.set_ylabel('Q-factor')
        ax.set_title(DATASET_LABELS_SHORT[ds])
        ax.grid(True, alpha=0.25)

        # Annotate fraction resonant
        frac_h = r.get('frac_resonant_hrf', None)
        frac_l = r.get('frac_resonant_lif', None)
        if frac_h is not None and frac_l is not None:
            ax.text(
                0.03, 0.97,
                f'band-pass:\nHRF {frac_h*100:.0f}% · LIF {frac_l*100:.0f}%',
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', linewidth=0.5, alpha=0.9))

    # Shared legend
    handles = [plt.Line2D([], [], marker='o', ls='', color=HRF_COLOR,
                          markersize=5, alpha=0.8, label='s-RON'),
               plt.Line2D([], [], marker='o', ls='', color=LIF_COLOR,
                          markersize=5, alpha=0.8, label='LIF-RC')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, frameon=False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'freq_q_scatter.{ext}'))
    plt.close()
    print("  Saved: freq_q_scatter.pdf/png")


# =============================================================================
# NEW Panel: Richness bars (PR + LUD + ASE)
# =============================================================================

def plot_richness_bars(data, out_dir, style='paper'):
    """
    Three side-by-side subpanels for PR, LUD, ASE.
    If your eff_dim analysis shows both reservoirs are similarly low-rank,
    you may choose to keep only ASE (entropy still discriminates) or drop
    this figure entirely.
    """
    ds_present = [ds for ds in DATASETS
                  if ds in data and 'eff_dim' in data[ds]]
    if not ds_present:
        print("  WARNING: no eff_dim data, skipping richness_bars")
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(7.5, 2.2) if style == 'paper' else (11, 3.5))

    x     = np.arange(len(ds_present))
    width = 0.32

    metrics = [
        ('pr_hrf', 'pr_lif',  'Participation Ratio'),
        ('lud_hrf', 'lud_lif', 'LUD (90% var)'),
        ('ase_hrf', 'ase_lif', 'Avg State Entropy'),
    ]

    for ax, (kh, kl, title) in zip(axes, metrics):
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
    # Note the mode (temporal or pooled) once at the bottom of the figure
    mode = data[ds_present[0]]['eff_dim'].get('mode', 'pooled')
    fig.text(0.5, -0.03,
             f'Richness metrics computed on {mode} reservoir states',
             ha='center', fontsize=7, style='italic')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(out_dir, f'richness_bars.{ext}'))
    plt.close()
    print("  Saved: richness_bars.pdf/png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None,
                        help="Where JSON results live. Default: "
                             "<script_dir>/analysis_results_extended")
    parser.add_argument('--out_dir', default=None,
                        help="Where figures are written. Default: "
                             "<script_dir>/figures_extended")
    parser.add_argument('--style',       default='paper', choices=['paper', 'slides'])
    parser.add_argument('--cka_feature', default='mean',
                        choices=['mean', 'rms', 'final'])
    parser.add_argument('--skip', nargs='*', default=[],
                        help="Plots to skip. Choices: cka, freq, combined, "
                             "probe, scatter, mc_curves, mc_summary, richness")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(script_dir, 'analysis_results_extended')
    out_dir     = args.out_dir     or os.path.join(script_dir, 'figures_extended')
    os.makedirs(out_dir, exist_ok=True)

    set_style(args.style)

    print(f"\nLoading results from: {results_dir}")
    print(f"CKA feature: {args.cka_feature}")
    data = load_results(results_dir, cka_feature=args.cka_feature)
    if not data:
        print("No data loaded — check results_dir path.")
        return

    print(f"\nGenerating figures in: {out_dir}")

    # Produce every figure by default. After seeing all four datasets,
    # some of these (MC, richness) may be moved to appendix or dropped.
    if 'cka'        not in args.skip: plot_cka(data, out_dir, args.style)
    if 'freq'       not in args.skip: plot_freq_selectivity(data, out_dir, args.style)
    if 'combined'   not in args.skip: plot_combined(data, out_dir, args.style)
    if 'probe'      not in args.skip: plot_probe_vs_cka(data, out_dir, args.style)
    if 'scatter'    not in args.skip: plot_freq_q_scatter(data, out_dir, args.style)
    if 'mc_curves'  not in args.skip: plot_memory_capacity_curves(data, out_dir, args.style)
    if 'mc_summary' not in args.skip: plot_mc_summary(data, out_dir, args.style)
    if 'richness'   not in args.skip: plot_richness_bars(data, out_dir, args.style)

    print("\nDone.")


if __name__ == '__main__':
    main()
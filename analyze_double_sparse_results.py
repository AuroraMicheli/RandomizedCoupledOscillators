"""
Analysis script for double-sparse connectivity experiments
Creates heatmaps showing test accuracy and energy as a function of both:
  - LIF→HRF connectivity
  - HRF→HRF recurrent connectivity
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_results(results_dir):
    """Load all JSON result files"""
    result_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    if len(result_files) == 0:
        print(f"❌ No result files found in {results_dir}")
        return []
    
    results = []
    for filepath in result_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    print(f"✅ Loaded {len(results)} result files")
    return results


def aggregate_double_sparse_results(results):
    """Aggregate results by n_hid, LIF→HRF connectivity, and HRF→HRF connectivity"""
    
    grouped = {}
    
    for result in results:
        n_hid = result['n_hid']
        conn_lif = result['connectivity_lif2hrf']
        conn_hrf = result['connectivity_hrf2hrf']
        key = (n_hid, conn_lif, conn_hrf)
        
        if key not in grouped:
            grouped[key] = {
                'test_accs': [],
                'train_accs': [],
                'energies': [],
                'sops': [],
                'hrf_sops': [],
                'lif_sops': [],
                'n_hid': n_hid,
                'conn_lif': conn_lif,
                'conn_hrf': conn_hrf,
            }
        
        grouped[key]['test_accs'].append(result['test_acc'])
        grouped[key]['train_accs'].append(result['train_acc'])
        grouped[key]['energies'].append(result['energy_J'])
        grouped[key]['sops'].append(result['SOPs'])
        grouped[key]['hrf_sops'].append(result['HRF_SOPs'])
        grouped[key]['lif_sops'].append(result['LIF_SOPs'])
    
    # Compute statistics
    aggregated = []
    for key, data in grouped.items():
        n_hid, conn_lif, conn_hrf = key
        
        entry = {
            'n_hid': n_hid,
            'conn_lif': conn_lif,
            'conn_hrf': conn_hrf,
            'test_acc_mean': np.mean(data['test_accs']),
            'test_acc_std': np.std(data['test_accs']),
            'train_acc_mean': np.mean(data['train_accs']),
            'train_acc_std': np.std(data['train_accs']),
            'energy_J_mean': np.mean(data['energies']),
            'energy_J_std': np.std(data['energies']),
            'SOPs_mean': np.mean(data['sops']),
            'SOPs_std': np.std(data['sops']),
            'HRF_SOPs_mean': np.mean(data['hrf_sops']),
            'LIF_SOPs_mean': np.mean(data['lif_sops']),
            'n_runs': len(data['test_accs'])
        }
        
        aggregated.append(entry)
    
    return aggregated


def create_heatmap(aggregated, n_hid, metric='test_acc_mean', save_path=None):
    """Create heatmap showing metric as function of both connectivities"""
    
    # Filter for specific n_hid
    filtered = [e for e in aggregated if e['n_hid'] == n_hid]
    
    if len(filtered) == 0:
        print(f"❌ No results for n_hid={n_hid}")
        return
    
    # Get unique connectivity values (sorted in reverse for better visualization)
    conn_lif_vals = sorted(list(set([e['conn_lif'] for e in filtered])), reverse=True)
    conn_hrf_vals = sorted(list(set([e['conn_hrf'] for e in filtered])), reverse=True)
    
    # Create matrix
    data_matrix = np.zeros((len(conn_hrf_vals), len(conn_lif_vals)))
    
    for entry in filtered:
        i = conn_hrf_vals.index(entry['conn_hrf'])
        j = conn_lif_vals.index(entry['conn_lif'])
        data_matrix[i, j] = entry[metric]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Labels
    x_labels = [f"{c*100:.0f}%" if c < 1.0 else "Dense" for c in conn_lif_vals]
    y_labels = [f"{c*100:.0f}%" if c < 1.0 else "Dense" for c in conn_hrf_vals]
    
    # Plot
    if 'acc' in metric:
        # Accuracy heatmap
        vmin = max(data_matrix.min() - 5, 0)
        vmax = min(data_matrix.max() + 5, 100)
        sns.heatmap(data_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=x_labels, yticklabels=y_labels,
                   cbar_kws={'label': 'Test Accuracy (%)'}, ax=ax,
                   vmin=vmin, vmax=vmax, linewidths=0.5)
        title = f'Test Accuracy Heatmap (n_hid={n_hid})'
    elif 'energy' in metric or 'SOP' in metric:
        # Energy/SOPs heatmap (lower is better, so reverse colormap)
        sns.heatmap(data_matrix, annot=True, fmt='.2e', cmap='RdYlGn_r',
                   xticklabels=x_labels, yticklabels=y_labels,
                   cbar_kws={'label': 'Energy (J)' if 'energy' in metric else 'SOPs'}, 
                   ax=ax, linewidths=0.5)
        title = f'Energy Consumption Heatmap (n_hid={n_hid})' if 'energy' in metric else f'Synaptic Operations Heatmap (n_hid={n_hid})'
    else:
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=x_labels, yticklabels=y_labels,
                   ax=ax, linewidths=0.5)
        title = f'{metric} Heatmap (n_hid={n_hid})'
    
    ax.set_xlabel('LIF→HRF Connectivity', fontsize=16, fontweight='bold')
    ax.set_ylabel('HRF→HRF Recurrent Connectivity', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap saved: {save_path}")
    
    plt.close()


def create_summary_table(aggregated, n_hid, save_path=None):
    """Create comprehensive summary table for specific n_hid"""
    
    filtered = [e for e in aggregated if e['n_hid'] == n_hid]
    filtered = sorted(filtered, key=lambda x: (-x['conn_lif'], -x['conn_hrf']))
    
    data = []
    for entry in filtered:
        conn_lif_label = "Dense" if entry['conn_lif'] == 1.0 else f"{entry['conn_lif']*100:.0f}%"
        conn_hrf_label = "Dense" if entry['conn_hrf'] == 1.0 else f"{entry['conn_hrf']*100:.0f}%"
        
        # Calculate total sparsity
        total_sparsity = entry['conn_lif'] * entry['conn_hrf'] * 100
        
        data.append({
            'LIF→HRF': conn_lif_label,
            'HRF→HRF': conn_hrf_label,
            'Total Sparsity (%)': f"{total_sparsity:.1f}%",
            'Train Acc (%)': f"{entry['train_acc_mean']:.2f} ± {entry['train_acc_std']:.2f}",
            'Test Acc (%)': f"{entry['test_acc_mean']:.2f} ± {entry['test_acc_std']:.2f}",
            'Energy (J)': f"{entry['energy_J_mean']:.3e} ± {entry['energy_J_std']:.2e}",
            'SOPs': f"{entry['SOPs_mean']:.3e} ± {entry['SOPs_std']:.2e}",
            'Runs': entry['n_runs']
        })
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*120}")
    print(f"SUMMARY TABLE: n_hid = {n_hid}")
    print(f"{'='*120}")
    print(df.to_string(index=False))
    print(f"{'='*120}\n")
    
    if save_path:
        csv_path = save_path.replace('.txt', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Table saved: {csv_path}")
        
        # Also save as formatted text
        with open(save_path, 'w') as f:
            f.write(f"SUMMARY TABLE: n_hid = {n_hid}\n")
            f.write("=" * 120 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n" + "=" * 120 + "\n")
        print(f"✅ Table saved: {save_path}")
    
    return df


def create_pareto_plot(aggregated, n_hid, save_path=None):
    """Create Pareto front plot: accuracy vs energy with connectivity labels"""
    
    filtered = [e for e in aggregated if e['n_hid'] == n_hid]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate total connectivity (product of both)
    total_conn = [e['conn_lif'] * e['conn_hrf'] for e in filtered]
    
    # Create scatter plot
    scatter = ax.scatter(
        [e['energy_J_mean'] for e in filtered],
        [e['test_acc_mean'] for e in filtered],
        c=total_conn,
        s=200,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=2,
        vmin=0,
        vmax=1
    )
    
    # Add labels for each point
    for entry in filtered:
        lif_pct = int(entry['conn_lif'] * 100)
        hrf_pct = int(entry['conn_hrf'] * 100)
        
        # Different label format based on connectivity
        if entry['conn_lif'] == 1.0 and entry['conn_hrf'] == 1.0:
            label = "Both Dense"
        elif entry['conn_lif'] == 1.0:
            label = f"LIF:D, HRF:{hrf_pct}%"
        elif entry['conn_hrf'] == 1.0:
            label = f"LIF:{lif_pct}%, HRF:D"
        else:
            label = f"L{lif_pct}/H{hrf_pct}"
        
        ax.annotate(label, 
                   (entry['energy_J_mean'], entry['test_acc_mean']),
                   textcoords="offset points",
                   xytext=(10, 10),
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Connectivity (LIF×HRF)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Energy Consumption (J)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'Accuracy vs Energy: Double-Sparse Pareto Front (n_hid={n_hid})', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Pareto plot saved: {save_path}")
    
    plt.close()


def create_energy_breakdown_plot(aggregated, n_hid, save_path=None):
    """Create stacked bar plot showing HRF vs LIF energy contributions"""
    
    filtered = [e for e in aggregated if e['n_hid'] == n_hid]
    # Sort by total energy
    filtered = sorted(filtered, key=lambda x: x['energy_J_mean'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    labels = [f"L{int(e['conn_lif']*100)}/H{int(e['conn_hrf']*100)}" for e in filtered]
    hrf_sops = [e['HRF_SOPs_mean'] for e in filtered]
    lif_sops = [e['LIF_SOPs_mean'] for e in filtered]
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Create stacked bars
    p1 = ax.bar(x, hrf_sops, width, label='HRF→HRF SOPs', color='steelblue')
    p2 = ax.bar(x, lif_sops, width, bottom=hrf_sops, label='LIF→HRF SOPs', color='coral')
    
    ax.set_xlabel('Connectivity Configuration (LIF%/HRF%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Synaptic Operations (SOPs)', fontsize=14, fontweight='bold')
    ax.set_title(f'Energy Breakdown: HRF vs LIF Contributions (n_hid={n_hid})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Energy breakdown plot saved: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze double-sparse connectivity results')
    parser.add_argument('--results_dir', type=str, default='results_double_sparse',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='analysis_double_sparse',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    print("=" * 120)
    print("DOUBLE SPARSE CONNECTIVITY ANALYSIS")
    print("=" * 120)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 120)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("\n📂 Loading results...")
    results = load_results(args.results_dir)
    
    if len(results) == 0:
        print("❌ No results to analyze")
        return
    
    # Aggregate results
    print("\n📊 Aggregating results...")
    aggregated = aggregate_double_sparse_results(results)
    
    # Get unique n_hid values
    n_hids = sorted(list(set([e['n_hid'] for e in aggregated])))
    print(f"✅ Found results for n_hid values: {n_hids}")
    
    # Process each n_hid
    for n_hid in n_hids:
        print(f"\n{'='*120}")
        print(f"PROCESSING n_hid = {n_hid}")
        print(f"{'='*120}")
        
        # Summary table
        table_path = os.path.join(args.output_dir, f'summary_table_nhid{n_hid}.txt')
        create_summary_table(aggregated, n_hid, save_path=table_path)
        
        # Accuracy heatmap
        acc_heatmap_path = os.path.join(args.output_dir, f'heatmap_accuracy_nhid{n_hid}.png')
        create_heatmap(aggregated, n_hid, metric='test_acc_mean', save_path=acc_heatmap_path)
        
        # Energy heatmap
        energy_heatmap_path = os.path.join(args.output_dir, f'heatmap_energy_nhid{n_hid}.png')
        create_heatmap(aggregated, n_hid, metric='energy_J_mean', save_path=energy_heatmap_path)
        
        # SOPs heatmap
        sops_heatmap_path = os.path.join(args.output_dir, f'heatmap_sops_nhid{n_hid}.png')
        create_heatmap(aggregated, n_hid, metric='SOPs_mean', save_path=sops_heatmap_path)
        
        # Pareto plot
        pareto_path = os.path.join(args.output_dir, f'pareto_front_nhid{n_hid}.png')
        create_pareto_plot(aggregated, n_hid, save_path=pareto_path)
        
        # Energy breakdown plot
        breakdown_path = os.path.join(args.output_dir, f'energy_breakdown_nhid{n_hid}.png')
        create_energy_breakdown_plot(aggregated, n_hid, save_path=breakdown_path)
    
    # Save aggregated data
    agg_path = os.path.join(args.output_dir, 'aggregated_double_sparse.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n✅ Aggregated data saved: {agg_path}")
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*120}")
    print(f"All outputs saved in: {args.output_dir}")
    print(f"\nGenerated files per n_hid:")
    print(f"  - summary_table_nhidXXX.csv/txt")
    print(f"  - heatmap_accuracy_nhidXXX.png")
    print(f"  - heatmap_energy_nhidXXX.png")
    print(f"  - heatmap_sops_nhidXXX.png")
    print(f"  - pareto_front_nhidXXX.png")
    print(f"  - energy_breakdown_nhidXXX.png")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
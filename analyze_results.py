import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def load_results(results_dir):
    """Load all JSON result files from directory"""
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


def aggregate_results(results):
    """Aggregate results by n_hid and connectivity"""
    
    # Group by n_hid and connectivity
    grouped = {}
    
    for result in results:
        n_hid = result['n_hid']
        connectivity = result['connectivity']
        key = (n_hid, connectivity)
        
        if key not in grouped:
            grouped[key] = {
                'train_accs': [],
                'test_accs': [],
                'energies': [],
                'n_hid': n_hid,
                'connectivity': connectivity,
                'n_connections': result['n_lif2hrf_connections'],
                'SOPs': []
            }
        
        grouped[key]['train_accs'].append(result['train_acc'])
        grouped[key]['test_accs'].append(result['test_acc'])
        grouped[key]['energies'].append(result['energy_J'])
        grouped[key]['SOPs'].append(result['SOPs'])
    
    # Compute statistics
    aggregated = []
    for key, data in grouped.items():
        n_hid, connectivity = key
        
        # Compute means and stds
        entry = {
            'n_hid': n_hid,
            'connectivity': connectivity,
            'connectivity_pct': connectivity * 100,
            'n_connections': data['n_connections'],
            'train_acc_mean': np.mean(data['train_accs']),
            'train_acc_std': np.std(data['train_accs']),
            'test_acc_mean': np.mean(data['test_accs']),
            'test_acc_std': np.std(data['test_accs']),
            'energy_J_mean': np.mean(data['energies']),
            'energy_J_std': np.std(data['energies']),
            'SOPs_mean': np.mean(data['SOPs']),
            'SOPs_std': np.std(data['SOPs']),
            'n_runs': len(data['train_accs'])
        }
        
        aggregated.append(entry)
    
    return aggregated


def create_summary_table(aggregated, n_hid, save_path=None):
    """Create summary table for specific n_hid"""
    
    # Filter for specific n_hid
    filtered = [entry for entry in aggregated if entry['n_hid'] == n_hid]
    
    if len(filtered) == 0:
        print(f"❌ No results found for n_hid={n_hid}")
        return None
    
    # Sort by connectivity descending (dense first)
    filtered = sorted(filtered, key=lambda x: x['connectivity'], reverse=True)
    
    # Create DataFrame
    data = []
    for entry in filtered:
        conn_label = "Dense" if entry['connectivity'] == 1.0 else f"{entry['connectivity_pct']:.0f}%"
        
        data.append({
            'Connectivity': conn_label,
            'Connections': f"{entry['n_connections']:.0f}",
            'Train Acc (%)': f"{entry['train_acc_mean']:.2f} ± {entry['train_acc_std']:.2f}",
            'Test Acc (%)': f"{entry['test_acc_mean']:.2f} ± {entry['test_acc_std']:.2f}",
            'Energy (J)': f"{entry['energy_J_mean']:.3e} ± {entry['energy_J_std']:.2e}",
            'SOPs': f"{entry['SOPs_mean']:.3e} ± {entry['SOPs_std']:.2e}",
            'Runs': entry['n_runs']
        })
    
    df = pd.DataFrame(data)
    
    # Print table
    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE: n_hid = {n_hid}")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")
    
    # Save to CSV
    if save_path:
        csv_path = save_path.replace('.txt', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Table saved to: {csv_path}")
        
        # Also save as formatted text
        with open(save_path, 'w') as f:
            f.write(f"SUMMARY TABLE: n_hid = {n_hid}\n")
            f.write("=" * 100 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n" + "=" * 100 + "\n")
        print(f"✅ Table saved to: {save_path}")
    
    return df


def create_accuracy_vs_energy_plot(aggregated, n_hid, save_path=None):
    """Create scatter plot of test accuracy vs energy for specific n_hid"""
    
    # Filter for specific n_hid
    filtered = [entry for entry in aggregated if entry['n_hid'] == n_hid]
    
    if len(filtered) == 0:
        print(f"❌ No results found for n_hid={n_hid}")
        return
    
    # Extract data
    connectivities = [entry['connectivity'] for entry in filtered]
    test_accs_mean = [entry['test_acc_mean'] for entry in filtered]
    test_accs_std = [entry['test_acc_std'] for entry in filtered]
    energies_mean = [entry['energy_J_mean'] for entry in filtered]
    energies_std = [entry['energy_J_std'] for entry in filtered]
    
    # Create labels
    labels = []
    for entry in filtered:
        if entry['connectivity'] == 1.0:
            labels.append("Dense")
        else:
            labels.append(f"{entry['connectivity_pct']:.0f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color map based on connectivity
    colors = plt.cm.viridis(np.linspace(0, 1, len(connectivities)))
    
    # Plot points with error bars
    for i, (conn, acc, acc_std, energy, energy_std, label, color) in enumerate(
        zip(connectivities, test_accs_mean, test_accs_std, energies_mean, energies_std, labels, colors)
    ):
        ax.errorbar(energy, acc, 
                   xerr=energy_std, yerr=acc_std,
                   marker='o', markersize=10, 
                   color=color, capsize=5, capthick=2,
                   label=f"{label} ({conn*100:.0f}%)",
                   linewidth=2, alpha=0.8)
        
        # Add text label
        ax.annotate(label, (energy, acc), 
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Energy Consumption (J)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Test Accuracy vs Energy Consumption (n_hid = {n_hid})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Connectivity', fontsize=10, title_fontsize=11, loc='best')
    
    # Use scientific notation for x-axis
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.close()


def create_combined_plot(aggregated, save_path=None):
    """Create combined plot showing both n_hid configurations"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    n_hids = sorted(list(set([entry['n_hid'] for entry in aggregated])))
    
    for idx, n_hid in enumerate(n_hids):
        ax = axes[idx]
        
        # Filter for this n_hid
        filtered = [entry for entry in aggregated if entry['n_hid'] == n_hid]
        filtered = sorted(filtered, key=lambda x: x['connectivity'], reverse=True)
        
        # Extract data
        connectivities = [entry['connectivity'] for entry in filtered]
        test_accs_mean = [entry['test_acc_mean'] for entry in filtered]
        test_accs_std = [entry['test_acc_std'] for entry in filtered]
        energies_mean = [entry['energy_J_mean'] for entry in filtered]
        energies_std = [entry['energy_J_std'] for entry in filtered]
        
        # Create labels
        labels = []
        for entry in filtered:
            if entry['connectivity'] == 1.0:
                labels.append("Dense")
            else:
                labels.append(f"{entry['connectivity_pct']:.0f}%")
        
        # Color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(connectivities)))
        
        # Plot
        for i, (conn, acc, acc_std, energy, energy_std, label, color) in enumerate(
            zip(connectivities, test_accs_mean, test_accs_std, energies_mean, energies_std, labels, colors)
        ):
            ax.errorbar(energy, acc, 
                       xerr=energy_std, yerr=acc_std,
                       marker='o', markersize=10, 
                       color=color, capsize=5, capthick=2,
                       label=f"{label}",
                       linewidth=2, alpha=0.8)
            
            ax.annotate(label, (energy, acc), 
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Energy Consumption (J)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'n_hid = {n_hid}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(title='Connectivity', fontsize=9, title_fontsize=10)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    fig.suptitle('Test Accuracy vs Energy Consumption: Sparse vs Dense Connectivity', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined plot saved to: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze sparse connectivity experiment results')
    parser.add_argument('--results_dir', type=str, default='results_sparse_connectivity',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("SPARSE CONNECTIVITY EXPERIMENT ANALYSIS")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 100)
    
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
    aggregated = aggregate_results(results)
    
    # Get unique n_hid values
    n_hids = sorted(list(set([entry['n_hid'] for entry in aggregated])))
    print(f"✅ Found results for n_hid values: {n_hids}")
    
    # Create tables and plots for each n_hid
    for n_hid in n_hids:
        print(f"\n{'='*100}")
        print(f"PROCESSING n_hid = {n_hid}")
        print(f"{'='*100}")
        
        # Create summary table
        table_path = os.path.join(args.output_dir, f'summary_table_nhid{n_hid}.txt')
        create_summary_table(aggregated, n_hid, save_path=table_path)
        
        # Create plot
        plot_path = os.path.join(args.output_dir, f'accuracy_vs_energy_nhid{n_hid}.png')
        create_accuracy_vs_energy_plot(aggregated, n_hid, save_path=plot_path)
    
    # Create combined plot
    print(f"\n{'='*100}")
    print("CREATING COMBINED PLOT")
    print(f"{'='*100}")
    combined_plot_path = os.path.join(args.output_dir, 'accuracy_vs_energy_combined.png')
    create_combined_plot(aggregated, save_path=combined_plot_path)
    
    # Save aggregated data
    aggregated_path = os.path.join(args.output_dir, 'aggregated_results.json')
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n✅ Aggregated data saved to: {aggregated_path}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")
    print(f"All outputs saved in: {args.output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# =============================
# Load results
# =============================

with open("summary.json") as f:
    results = json.load(f)

print(f"✅ Loaded {len(results)} configurations")

# =============================
# 1. Best Configuration
# =============================

best = results[0]
print("\n" + "=" * 70)
print("🏆 BEST CONFIGURATION")
print("=" * 70)
print(f"Mean Test Accuracy: {best['mean_test_acc']:.2f}% ± {best['std_test_acc']:.2f}%")
print(f"Config ID: {best['config_id']}")
print("\nHyperparameters:")
for key in ['gamma', 'epsilon', 'gamma_range', 'epsilon_range', 'rho', 'inp_scaling']:
    if key in best['params']:
        print(f"  {key:15s}: {best['params'][key]:.4f}")

# =============================
# 2. Top 10 Configurations
# =============================

print("\n" + "=" * 70)
print("📊 TOP 10 CONFIGURATIONS")
print("=" * 70)
print(f"{'Rank':<6} {'Acc (%)':>10} {'Std':>7} {'gamma':>8} {'epsilon':>8} {'rho':>8}")
print("-" * 70)

for rank, cfg in enumerate(results[:10], 1):
    p = cfg['params']
    print(f"{rank:<6} {cfg['mean_test_acc']:>9.2f}% {cfg['std_test_acc']:>6.2f}% "
          f"{p['gamma']:>7.2f} {p['epsilon']:>7.3f} {p['rho']:>7.3f}")

# =============================
# 3. Parameter Importance Analysis
# =============================

# Extract data for correlation analysis
data = []
for cfg in results:
    row = {
        'test_acc': cfg['mean_test_acc'],
        'std_acc': cfg['std_test_acc'],
    }
    for key in ['gamma', 'epsilon', 'gamma_range', 'epsilon_range', 'rho', 'inp_scaling']:
        if key in cfg['params']:
            row[key] = cfg['params'][key]
    data.append(row)

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("📈 PARAMETER CORRELATIONS WITH TEST ACCURACY")
print("=" * 70)

correlations = df.corr()['test_acc'].sort_values(ascending=False)
print(correlations[1:])  # Exclude self-correlation

# =============================
# 4. Visualizations
# =============================

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('FordA Hyperparameter Search Analysis', fontsize=16, fontweight='bold')

params_to_plot = ['gamma', 'epsilon', 'gamma_range', 'epsilon_range', 'rho', 'inp_scaling']

for idx, param in enumerate(params_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    # Scatter plot
    scatter = ax.scatter(df[param], df['test_acc'], 
                        c=df['test_acc'], cmap='RdYlGn', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Best configuration marker
    best_val = best['params'][param]
    best_acc = best['mean_test_acc']
    ax.scatter([best_val], [best_acc], 
              marker='*', s=500, c='gold', edgecolors='black', 
              linewidth=2, label='Best Config', zorder=10)
    
    # Formatting
    ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    # Add correlation text
    corr = correlations[param]
    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', 
           transform=ax.transAxes, 
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           verticalalignment='top')

plt.tight_layout()
plt.savefig('hparam_search_fordA/parameter_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: hparam_search_fordA/parameter_analysis.png")

# =============================
# 5. Stability Analysis (Low std = robust)
# =============================

fig, ax = plt.subplots(figsize=(10, 6))

# Color by stability (inverse of std)
stability = 1 / (df['std_acc'] + 0.1)  # Avoid division by zero
scatter = ax.scatter(df['test_acc'], df['std_acc'], 
                    c=df['test_acc'], cmap='RdYlGn',
                    s=stability * 100, alpha=0.6, 
                    edgecolors='black', linewidth=0.5)

# Best configuration
ax.scatter([best['mean_test_acc']], [best['std_test_acc']], 
          marker='*', s=800, c='gold', edgecolors='black', 
          linewidth=2, label='Best Config', zorder=10)

ax.set_xlabel('Mean Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Std Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Stability\n(Larger markers = more stable)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Test Accuracy (%)', fontsize=10)

plt.tight_layout()
plt.savefig('hparam_search_fordA/stability_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: hparam_search_fordA/stability_analysis.png")

# =============================
# 6. Performance Distribution
# =============================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['test_acc'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(best['mean_test_acc'], color='red', linestyle='--', linewidth=2, label='Best Config')
axes[0].set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Test Accuracies', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot([df['test_acc']], labels=['All Configs'], widths=0.5)
axes[1].scatter([1], [best['mean_test_acc']], marker='*', s=500, c='gold', 
               edgecolors='red', linewidth=2, label='Best Config', zorder=10)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Overall Performance Range', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hparam_search_fordA/performance_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: hparam_search_fordA/performance_distribution.png")

# =============================
# 7. Statistical Summary
# =============================

print("\n" + "=" * 70)
print("📊 STATISTICAL SUMMARY")
print("=" * 70)
print(f"Mean accuracy (all configs): {df['test_acc'].mean():.2f}% ± {df['test_acc'].std():.2f}%")
print(f"Median accuracy:             {df['test_acc'].median():.2f}%")
print(f"Best accuracy:               {df['test_acc'].max():.2f}%")
print(f"Worst accuracy:              {df['test_acc'].min():.2f}%")
print(f"Range:                       {df['test_acc'].max() - df['test_acc'].min():.2f}%")

print("\n" + "=" * 70)
print("🎯 PARAMETER RANGES FOR TOP 10% CONFIGS")
print("=" * 70)

top_10_percent = int(len(results) * 0.1)
top_configs = results[:max(1, top_10_percent)]

for param in params_to_plot:
    values = [cfg['params'][param] for cfg in top_configs if param in cfg['params']]
    if values:
        print(f"{param:15s}: [{min(values):.3f}, {max(values):.3f}] "
              f"(mean: {np.mean(values):.3f})")

# =============================
# 8. Save LaTeX table for paper
# =============================

latex_table = "\\begin{table}[h]\n"
latex_table += "\\centering\n"
latex_table += "\\caption{Top 5 Hyperparameter Configurations for FordA Dataset}\n"
latex_table += "\\label{tab:forda_hparams}\n"
latex_table += "\\begin{tabular}{ccccccc}\n"
latex_table += "\\hline\n"
latex_table += "Rank & Acc (\\%) & $\\gamma$ & $\\epsilon$ & $\\rho$ & Input Scale \\\\\n"
latex_table += "\\hline\n"

for rank, cfg in enumerate(results[:5], 1):
    p = cfg['params']
    latex_table += f"{rank} & {cfg['mean_test_acc']:.2f} $\\pm$ {cfg['std_test_acc']:.2f} & "
    latex_table += f"{p['gamma']:.2f} & {p['epsilon']:.3f} & {p['rho']:.2f} & {p['inp_scaling']:.2f} \\\\\n"

latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\end{table}\n"

with open('hparam_search_fordA/table_for_paper.tex', 'w') as f:
    f.write(latex_table)

print("\n✅ Saved: hparam_search_fordA/table_for_paper.tex")

# =============================
# 9. Recommendations
# =============================

print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS FOR PAPER")
print("=" * 70)

print("\n1. **Optimal Hyperparameters** (use these for final experiments):")
print(f"   gamma:          {best['params']['gamma']:.3f}")
print(f"   epsilon:        {best['params']['epsilon']:.4f}")
print(f"   gamma_range:    {best['params']['gamma_range']:.3f}")
print(f"   epsilon_range:  {best['params']['epsilon_range']:.4f}")
print(f"   rho:            {best['params']['rho']:.3f}")
print(f"   input_scaling:  {best['params']['inp_scaling']:.3f}")

print("\n2. **Key Findings:**")
most_important = correlations[1:].abs().idxmax()
print(f"   - Most influential parameter: {most_important} (ρ = {correlations[most_important]:.3f})")
print(f"   - Best test accuracy: {best['mean_test_acc']:.2f}% ± {best['std_test_acc']:.2f}%")
print(f"   - Improvement over baseline: {best['mean_test_acc'] - df['test_acc'].mean():.2f}%")

print("\n3. **For Paper Figures:**")
print("   - Use: parameter_analysis.png (6-panel parameter sweep)")
print("   - Use: stability_analysis.png (accuracy vs robustness)")
print("   - Use: performance_distribution.png (overall performance)")

print("\n4. **For Paper Text:**")
print("   - Report best config in main results")
print("   - Include top 5 table (table_for_paper.tex)")
print("   - Discuss parameter importance in ablation section")
print("   - Mention robustness (low std across seeds)")

print("\n" + "=" * 70)
plt.show()
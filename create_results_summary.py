"""
Generate summary figures and tables from analysis results

Run this after main.py completes to create visualizations for the report.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create results directories if they don't exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

print("📊 Creating result summaries...")

# Example data (replace with actual results from main.py)
clustering_results = {
    'Portfolio': ['Conservative', 'Balanced', 'Aggressive'],
    'Total Return (%)': [60.78, 69.79, 84.03],
    'CAGR (%)': [12.61, 14.15, 16.47],
    'Sharpe Ratio': [0.85, 0.86, 0.72],
    'Max Drawdown (%)': [-17.94, -22.52, -30.42]
}

ml_results = {
    'Portfolio': ['Conservative', 'Balanced', 'Aggressive'],
    'Total Return (%)': [50.15, 60.63, 80.86],
    'CAGR (%)': [10.70, 12.58, 15.97],
    'Sharpe Ratio': [0.73, 0.81, 0.85],
    'Max Drawdown (%)': [-18.08, -20.92, -26.74]
}

# Create DataFrames
df_clustering = pd.DataFrame(clustering_results)
df_ml = pd.DataFrame(ml_results)

# Save tables
df_clustering.to_csv('results/tables/clustering_performance.csv', index=False)
df_ml.to_csv('results/tables/ml_performance.csv', index=False)
print("✓ Saved performance tables")

# Create comparison plot - Total Return
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(df_clustering))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], df_clustering['Total Return (%)'], 
               width, label='Clustering', alpha=0.8, color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], df_ml['Total Return (%)'], 
               width, label='ML-Driven', alpha=0.8, color='coral')

# Add S&P 500 benchmark line
ax.axhline(y=59.62, color='red', linestyle='--', linewidth=2, 
           label='S&P 500 (59.62%)', alpha=0.7)

ax.set_xlabel('Portfolio Type', fontsize=12)
ax.set_ylabel('Total Return (%)', fontsize=12)
ax.set_title('Portfolio Performance Comparison (2021-2024)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_clustering['Portfolio'])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved performance comparison chart")

# Create Sharpe Ratio comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], df_clustering['Sharpe Ratio'], 
               width, label='Clustering', alpha=0.8, color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], df_ml['Sharpe Ratio'], 
               width, label='ML-Driven', alpha=0.8, color='coral')

# Add S&P 500 Sharpe
ax.axhline(y=0.63, color='red', linestyle='--', linewidth=2, 
           label='S&P 500 (0.63)', alpha=0.7)

ax.set_xlabel('Portfolio Type', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Risk-Adjusted Performance (Sharpe Ratio)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_clustering['Portfolio'])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/sharpe_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Sharpe ratio comparison chart")

print("\n✅ Results summary created!")
print("📁 Files saved:")
print("   - results/tables/clustering_performance.csv")
print("   - results/tables/ml_performance.csv")
print("   - results/figures/performance_comparison.png")
print("   - results/figures/sharpe_comparison.png")

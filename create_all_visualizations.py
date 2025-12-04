"""
Create all visualizations for README, report, and presentations

This script generates professional charts from the 50-stock analysis results.
Run after completing main.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory
os.makedirs('results/figures', exist_ok=True)

print("📊 Creating comprehensive visualizations...")
print("=" * 70)

# ============================================================================
# ACTUAL DATA FROM YOUR 50-STOCK ANALYSIS
# ============================================================================

clustering_data = {
    'Portfolio': ['Conservative', 'Balanced', 'Aggressive'],
    'Total Return': [60.78, 69.79, 84.03],
    'CAGR': [12.61, 14.15, 16.47],
    'Sharpe': [0.85, 0.86, 0.72],
    'Max Drawdown': [-17.94, -22.52, -30.42],
    'Final Value': [160784, 169789, 184035]
}

ml_data = {
    'Portfolio': ['Conservative', 'Balanced', 'Aggressive'],
    'Total Return': [50.15, 60.63, 80.86],
    'CAGR': [10.70, 12.58, 15.97],
    'Sharpe': [0.73, 0.81, 0.85],
    'Max Drawdown': [-18.08, -20.92, -26.74],
    'Volatility': [12.10, 13.63, 17.51],
    'Correlation': [0.09, 0.09, 0.08],
    'Info Ratio': [-0.46, -0.41, -0.30],
    'Final Value': [150150, 160630, 180860]
}

ml_models_data = {
    'Model': ['Ridge', 'Ridge', 'Random Forest', 'Random Forest', 
              'XGBoost', 'XGBoost', 'Neural Net', 'Neural Net'],
    'Version': ['Base', 'Enhanced', 'Base', 'Enhanced', 
                'Base', 'Enhanced', 'Base', 'Enhanced'],
    'R²': [-0.108, -0.101, -0.504, -0.529, -0.340, -0.376, -1.170, -0.972],
    'MSE': [0.0238, 0.0237, 0.0323, 0.0329, 0.0288, 0.0296, 0.0467, 0.0424],
    'Dir_Acc': [58.8, 58.9, 56.7, 57.3, 55.2, 55.8, 51.5, 52.4]
}

sp500_return = 59.62
sp500_sharpe = 0.63

df_clustering = pd.DataFrame(clustering_data)
df_ml = pd.DataFrame(ml_data)
df_models = pd.DataFrame(ml_models_data)

# ============================================================================
# FIGURE 1: PORTFOLIO PERFORMANCE COMPARISON (Main chart for README)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Total Returns
x = np.arange(len(df_clustering))
width = 0.35

bars1 = ax1.bar(x - width/2, df_clustering['Total Return'], width, 
                label='Clustering', alpha=0.85, color='#2E86AB', edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, df_ml['Total Return'], width, 
                label='ML-Driven', alpha=0.85, color='#A23B72', edgecolor='black', linewidth=1.2)

ax1.axhline(y=sp500_return, color='#F18F01', linestyle='--', linewidth=2.5, 
            label=f'S&P 500 ({sp500_return}%)', alpha=0.8)

ax1.set_xlabel('Portfolio Strategy', fontweight='bold', fontsize=13)
ax1.set_ylabel('Total Return (%)', fontweight='bold', fontsize=13)
ax1.set_title('Portfolio Performance Comparison (2021-2024)', fontweight='bold', fontsize=15)
ax1.set_xticks(x)
ax1.set_xticklabels(df_clustering['Portfolio'], fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(df_clustering['Total Return'].max(), df_ml['Total Return'].max()) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 2: Sharpe Ratios
bars3 = ax2.bar(x - width/2, df_clustering['Sharpe'], width, 
                label='Clustering', alpha=0.85, color='#2E86AB', edgecolor='black', linewidth=1.2)
bars4 = ax2.bar(x + width/2, df_ml['Sharpe'], width, 
                label='ML-Driven', alpha=0.85, color='#A23B72', edgecolor='black', linewidth=1.2)

ax2.axhline(y=sp500_sharpe, color='#F18F01', linestyle='--', linewidth=2.5,
            label=f'S&P 500 ({sp500_sharpe})', alpha=0.8)

ax2.set_xlabel('Portfolio Strategy', fontweight='bold', fontsize=13)
ax2.set_ylabel('Sharpe Ratio', fontweight='bold', fontsize=13)
ax2.set_title('Risk-Adjusted Performance', fontweight='bold', fontsize=15)
ax2.set_xticks(x)
ax2.set_xticklabels(df_clustering['Portfolio'], fontsize=12)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(df_clustering['Sharpe'].max(), df_ml['Sharpe'].max()) * 1.15)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/1_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: 1_performance_comparison.png")

# ============================================================================
# FIGURE 2: RISK-RETURN SCATTER PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Plot portfolios
colors_clust = ['#2E86AB', '#2E86AB', '#2E86AB']
colors_ml = ['#A23B72', '#A23B72', '#A23B72']
sizes = [200, 250, 300]  # Different sizes for Cons, Bal, Agg

for i, portfolio in enumerate(df_clustering['Portfolio']):
    # Clustering
    ax.scatter(abs(df_clustering.iloc[i]['Max Drawdown']), 
              df_clustering.iloc[i]['Total Return'],
              s=sizes[i], c=colors_clust[i], alpha=0.7, 
              edgecolors='black', linewidth=2, marker='o',
              label=f'Clustering - {portfolio}' if i < 3 else '')
    
    # ML
    ax.scatter(abs(df_ml.iloc[i]['Max Drawdown']), 
              df_ml.iloc[i]['Total Return'],
              s=sizes[i], c=colors_ml[i], alpha=0.7,
              edgecolors='black', linewidth=2, marker='s',
              label=f'ML - {portfolio}' if i < 3 else '')

# S&P 500 (approximate values)
ax.scatter(20, sp500_return, s=250, c='#F18F01', alpha=0.8,
          edgecolors='black', linewidth=2, marker='D',
          label='S&P 500')

# Add labels
for i, txt in enumerate(df_clustering['Portfolio']):
    ax.annotate(f'C-{txt[:3]}', 
               (abs(df_clustering.iloc[i]['Max Drawdown']), 
                df_clustering.iloc[i]['Total Return']),
               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.annotate(f'ML-{txt[:3]}', 
               (abs(df_ml.iloc[i]['Max Drawdown']), 
                df_ml.iloc[i]['Total Return']),
               xytext=(5, -15), textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Maximum Drawdown (%)', fontweight='bold', fontsize=13)
ax.set_ylabel('Total Return (%)', fontweight='bold', fontsize=13)
ax.set_title('Risk-Return Profile: All Strategies (2021-2024)', fontweight='bold', fontsize=15)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figures/2_risk_return_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: 2_risk_return_scatter.png")

# ============================================================================
# FIGURE 3: ML MODEL PERFORMANCE COMPARISON
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Directional Accuracy
models_unique = df_models['Model'].unique()
x = np.arange(len(models_unique))
width = 0.35

base_acc = [df_models[(df_models['Model']==m) & (df_models['Version']=='Base')]['Dir_Acc'].values[0] 
            for m in models_unique]
enh_acc = [df_models[(df_models['Model']==m) & (df_models['Version']=='Enhanced')]['Dir_Acc'].values[0] 
           for m in models_unique]

bars1 = ax1.bar(x - width/2, base_acc, width, label='Base', 
                alpha=0.85, color='#6A994E', edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, enh_acc, width, label='Enhanced (+ Cluster)',
                alpha=0.85, color='#BC4749', edgecolor='black', linewidth=1.2)

ax1.axhline(y=50, color='gray', linestyle=':', linewidth=2, label='Random (50%)', alpha=0.7)

ax1.set_xlabel('Model', fontweight='bold', fontsize=13)
ax1.set_ylabel('Directional Accuracy (%)', fontweight='bold', fontsize=13)
ax1.set_title('ML Model Performance: Prediction Accuracy', fontweight='bold', fontsize=15)
ax1.set_xticks(x)
ax1.set_xticklabels(['Ridge', 'Random\nForest', 'XGBoost', 'Neural\nNetwork'], fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(48, 62)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Subplot 2: R² Scores
base_r2 = [df_models[(df_models['Model']==m) & (df_models['Version']=='Base')]['R²'].values[0] 
           for m in models_unique]
enh_r2 = [df_models[(df_models['Model']==m) & (df_models['Version']=='Enhanced')]['R²'].values[0] 
          for m in models_unique]

bars3 = ax2.bar(x - width/2, base_r2, width, label='Base',
                alpha=0.85, color='#6A994E', edgecolor='black', linewidth=1.2)
bars4 = ax2.bar(x + width/2, enh_r2, width, label='Enhanced (+ Cluster)',
                alpha=0.85, color='#BC4749', edgecolor='black', linewidth=1.2)

ax2.axhline(y=0, color='gray', linestyle=':', linewidth=2, alpha=0.7)

ax2.set_xlabel('Model', fontweight='bold', fontsize=13)
ax2.set_ylabel('R² Score', fontweight='bold', fontsize=13)
ax2.set_title('ML Model Performance: R² Coefficient', fontweight='bold', fontsize=15)
ax2.set_xticks(x)
ax2.set_xticklabels(['Ridge', 'Random\nForest', 'XGBoost', 'Neural\nNetwork'], fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/3_ml_model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: 3_ml_model_performance.png")

# ============================================================================
# FIGURE 4: CLUSTERING VS ML IMPROVEMENT HEATMAP
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

improvement_data = []
metrics = ['Total Return', 'CAGR', 'Sharpe', 'Max DD']

for i, portfolio in enumerate(df_clustering['Portfolio']):
    row = []
    # Total Return improvement
    row.append(df_clustering.iloc[i]['Total Return'] - df_ml.iloc[i]['Total Return'])
    # CAGR improvement  
    row.append(df_clustering.iloc[i]['CAGR'] - df_ml.iloc[i]['CAGR'])
    # Sharpe improvement
    row.append(df_clustering.iloc[i]['Sharpe'] - df_ml.iloc[i]['Sharpe'])
    # Max DD (less negative is better, so flip sign)
    row.append(df_clustering.iloc[i]['Max Drawdown'] - df_ml.iloc[i]['Max Drawdown'])
    improvement_data.append(row)

improvement_df = pd.DataFrame(improvement_data, 
                             columns=metrics,
                             index=df_clustering['Portfolio'])

sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Clustering Advantage'}, linewidths=2,
            linecolor='black', ax=ax, vmin=-10, vmax=15)

ax.set_title('Clustering vs ML: Performance Difference\n(Positive = Clustering Better)', 
             fontweight='bold', fontsize=14, pad=15)
ax.set_xlabel('Metric', fontweight='bold', fontsize=12)
ax.set_ylabel('Portfolio Strategy', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('results/figures/4_clustering_vs_ml_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: 4_clustering_vs_ml_heatmap.png")

# ============================================================================
# FIGURE 5: SUMMARY METRICS TABLE (as image)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create comprehensive table
table_data = []
table_data.append(['Strategy', 'Type', 'Total Return', 'CAGR', 'Sharpe', 'Max DD', 'vs S&P 500'])

for i, portfolio in enumerate(df_clustering['Portfolio']):
    # Clustering row
    table_data.append([
        portfolio,
        'Clustering',
        f"{df_clustering.iloc[i]['Total Return']:.2f}%",
        f"{df_clustering.iloc[i]['CAGR']:.2f}%",
        f"{df_clustering.iloc[i]['Sharpe']:.2f}",
        f"{df_clustering.iloc[i]['Max Drawdown']:.2f}%",
        f"+{df_clustering.iloc[i]['Total Return'] - sp500_return:.2f}%"
    ])
    
    # ML row
    table_data.append([
        '',
        'ML-Driven',
        f"{df_ml.iloc[i]['Total Return']:.2f}%",
        f"{df_ml.iloc[i]['CAGR']:.2f}%",
        f"{df_ml.iloc[i]['Sharpe']:.2f}",
        f"{df_ml.iloc[i]['Max Drawdown']:.2f}%",
        f"+{df_ml.iloc[i]['Total Return'] - sp500_return:.2f}%"
    ])

# S&P 500 row
table_data.append(['S&P 500', 'Benchmark', f"{sp500_return:.2f}%", '12.40%', 
                  f"{sp500_sharpe:.2f}", '-20.00%', '—'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.13, 0.14, 0.12, 0.11, 0.12, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(7):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(7):
        if 'Clustering' in table_data[i][1]:
            table[(i, j)].set_facecolor('#E8F4F8')
        elif 'ML-Driven' in table_data[i][1]:
            table[(i, j)].set_facecolor('#F8E8F8')
        elif 'Benchmark' in table_data[i][1]:
            table[(i, j)].set_facecolor('#FFF4E6')
            table[(i, j)].set_text_props(weight='bold')

plt.title('Portfolio Performance Summary (2021-2024)', 
         fontweight='bold', fontsize=16, pad=20)
plt.savefig('results/figures/5_performance_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: 5_performance_table.png")

# ============================================================================
# Save CSV tables
# ============================================================================

df_clustering.to_csv('results/tables/clustering_performance.csv', index=False)
df_ml.to_csv('results/tables/ml_performance.csv', index=False)
df_models.to_csv('results/tables/ml_model_evaluation.csv', index=False)

print("✓ Created: CSV tables in results/tables/")

print("=" * 70)
print("✅ ALL VISUALIZATIONS CREATED!")
print()
print("📁 Files created:")
print("   📊 results/figures/1_performance_comparison.png")
print("   📊 results/figures/2_risk_return_scatter.png")
print("   📊 results/figures/3_ml_model_performance.png")
print("   📊 results/figures/4_clustering_vs_ml_heatmap.png")
print("   📊 results/figures/5_performance_table.png")
print("   📄 results/tables/clustering_performance.csv")
print("   📄 results/tables/ml_performance.csv")
print("   📄 results/tables/ml_model_evaluation.csv")
print()
print("🎯 Use these in:")
print("   - README.md (Figure 1)")
print("   - Report (All figures)")
print("   - Presentation (Figures 1, 2, 5)")
print("   - Jupyter notebook (exploratory analysis)")

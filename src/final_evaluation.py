"""
Final evaluation module - comprehensive comparison with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '/files/portfolio-clustering-project/src')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_visualizations(clustering_results, ml_results, sp500_benchmark, clustering_histories, ml_histories, sp500_history):
    """Create comprehensive visualizations."""
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # Extract scalar values
    sp500_return = float(sp500_benchmark['return'])
    sp500_sharpe = float(sp500_benchmark['sharpe'])
    sp500_volatility = float(sp500_benchmark['volatility'])
    sp500_cagr = float(sp500_benchmark['cagr'])
    
    # 1. Portfolio Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a. Total Returns Bar Chart
    ax = axes[0, 0]
    portfolios = ['Conservative', 'Balanced', 'Aggressive']
    clustering_returns = [clustering_results[p]['total_return'] * 100 for p in portfolios]
    ml_returns = [ml_results[p]['total_return'] * 100 for p in portfolios]
    
    x = np.arange(len(portfolios))
    width = 0.25
    
    ax.bar(x - width, clustering_returns, width, label='Clustering-Based', color='steelblue')
    ax.bar(x, ml_returns, width, label='ML-Driven', color='coral')
    ax.axhline(y=sp500_return * 100, color='green', linestyle='--', label='S&P 500', linewidth=2)
    
    ax.set_xlabel('Portfolio Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Total Returns Comparison (2021-2024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1b. Sharpe Ratio Comparison
    ax = axes[0, 1]
    clustering_sharpe = [clustering_results[p]['sharpe_ratio'] for p in portfolios]
    ml_sharpe = [ml_results[p]['sharpe_ratio'] for p in portfolios]
    
    ax.bar(x - width, clustering_sharpe, width, label='Clustering-Based', color='steelblue')
    ax.bar(x, ml_sharpe, width, label='ML-Driven', color='coral')
    ax.axhline(y=sp500_sharpe, color='green', linestyle='--', label='S&P 500', linewidth=2)
    
    ax.set_xlabel('Portfolio Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1c. Max Drawdown Comparison
    ax = axes[1, 0]
    clustering_dd = [abs(clustering_results[p]['max_drawdown']) * 100 for p in portfolios]
    ml_dd = [abs(ml_results[p]['max_drawdown']) * 100 for p in portfolios]
    
    ax.bar(x - width/2, clustering_dd, width, label='Clustering-Based', color='steelblue')
    ax.bar(x + width/2, ml_dd, width, label='ML-Driven', color='coral')
    
    ax.set_xlabel('Portfolio Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk Measure: Maximum Drawdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1d. Improvement Table
    ax = axes[1, 1]
    ax.axis('off')
    
    improvement_data = []
    for p in portfolios:
        improvement = (ml_results[p]['total_return'] - clustering_results[p]['total_return']) * 100
        improvement_data.append([
            p,
            f"{clustering_results[p]['total_return']*100:.1f}%",
            f"{ml_results[p]['total_return']*100:.1f}%",
            f"+{improvement:.1f}%"
        ])
    
    table = ax.table(cellText=improvement_data,
                     colLabels=['Portfolio', 'Clustering', 'ML-Driven', 'Improvement'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('results/figures/1_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 1_performance_comparison.png")
    
    # 2. Cumulative Returns Over Time
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for name, history in clustering_histories.items():
        if len(history) > 0:
            history = history.sort_values('date').reset_index(drop=True)
            history['date'] = pd.to_datetime(history['date'])
            cumulative = (history['value'] / 100000 - 1) * 100
            ax.plot(history['date'], cumulative, label=f'{name} (Clustering)', linestyle='--', linewidth=2)
    
    for name, history in ml_histories.items():
        if len(history) > 0:
            history = history.sort_values('date').reset_index(drop=True)
            history['date'] = pd.to_datetime(history['date'])
            cumulative = (history['value'] / 100000 - 1) * 100
            ax.plot(history['date'], cumulative, label=f'{name} (ML)', linewidth=2.5)
    
    # Add S&P 500
    sp500_cumulative = (sp500_history / sp500_history.iloc[0] - 1) * 100
    ax.plot(sp500_history.index, sp500_cumulative, label='S&P 500', color='green', linestyle=':', linewidth=3)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Performance Over Time (2021-2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/2_cumulative_returns.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 2_cumulative_returns.png")
    
    # 3. Risk-Return Scatter Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Conservative': 'blue', 'Balanced': 'purple', 'Aggressive': 'red'}
    
    for p in portfolios:
        # Clustering
        ax.scatter(clustering_results[p]['volatility'] * 100, 
                  clustering_results[p]['cagr'] * 100,
                  s=200, alpha=0.6, marker='o', color=colors[p], 
                  edgecolors='black', linewidths=2, label=f'{p} (Clustering)')
        # ML
        ax.scatter(ml_results[p]['volatility'] * 100,
                  ml_results[p]['cagr'] * 100,
                  s=200, alpha=0.6, marker='s', color=colors[p],
                  edgecolors='black', linewidths=2, label=f'{p} (ML)')
    
    # S&P 500
    ax.scatter(sp500_volatility * 100,
              sp500_cagr * 100,
              s=300, alpha=0.8, marker='*', color='green', 
              edgecolors='black', linewidths=2, label='S&P 500')
    
    ax.set_xlabel('Volatility (Annual %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CAGR (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/3_risk_return_scatter.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 3_risk_return_scatter.png")
    
    # 4. ML Improvement Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Total Return', 'CAGR', 'Sharpe Ratio', 'Drawdown Reduction']
    improvement_matrix = []
    
    for p in portfolios:
        row = [
            (ml_results[p]['total_return'] - clustering_results[p]['total_return']) * 100,
            (ml_results[p]['cagr'] - clustering_results[p]['cagr']) * 100,
            (ml_results[p]['sharpe_ratio'] - clustering_results[p]['sharpe_ratio']),
            (abs(clustering_results[p]['max_drawdown']) - abs(ml_results[p]['max_drawdown'])) * 100
        ]
        improvement_matrix.append(row)
    
    improvement_df = pd.DataFrame(improvement_matrix, index=portfolios, columns=metrics)
    
    sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement'}, ax=ax, linewidths=1)
    
    ax.set_title('ML Enhancement: Improvement Over Clustering-Based Portfolios', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig('results/figures/4_ml_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 4_ml_improvement_heatmap.png")
    
    plt.close('all')


def run_complete_evaluation():
    """Run complete project evaluation with visualizations."""
    
    print("\n" + "="*80)
    print("PORTFOLIO CLUSTERING PROJECT - FINAL EVALUATION")
    print("="*80)
    
    from data_loader import TEST_TICKERS, load_stock_data
    from feature_engineering import calculate_all_features
    from clustering import prepare_feature_matrix, standardize_features, apply_pca, perform_kmeans, label_clusters_by_volatility
    from portfolio import Portfolio
    from backtesting import quarterly_rebalancing_backtest, calculate_metrics
    from ml_portfolios import train_ml_model, ml_backtest
    import yfinance as yf
    
    print("\n[1/6] Loading data...")
    stock_data = load_stock_data(TEST_TICKERS)
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
    
    print("\n[2/6] Calculating features...")
    all_features = {}
    for ticker in TEST_TICKERS:
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        if len(features) > 0:
            all_features[ticker] = features
    
    print("\n[3/6] Clustering stocks...")
    feature_matrix = prepare_feature_matrix(all_features, use_latest=True)
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
    X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
    X_pca, pca = apply_pca(X_scaled, explained_variance_threshold=0.95)
    kmeans, labels, silhouette = perform_kmeans(X_pca, n_clusters=3)
    cluster_names = label_clusters_by_volatility(feature_matrix, labels)
    
    cluster_assignments = {}
    for idx, ticker in enumerate(feature_matrix['ticker']):
        cluster_assignments[ticker] = cluster_names[labels[idx]]
    
    print(f"  ✓ Clustered {len(cluster_assignments)} stocks (Silhouette: {silhouette:.3f})")
    
    print("\n[4/6] Backtesting clustering-based portfolios...")
    
    clustering_results = {}
    clustering_histories = {}
    
    portfolio_configs = {
        'Conservative': {'low-volatility': 0.6, 'moderate': 0.3, 'high-volatility': 0.1},
        'Balanced': {'low-volatility': 0.4, 'moderate': 0.4, 'high-volatility': 0.2},
        'Aggressive': {'low-volatility': 0.2, 'moderate': 0.3, 'high-volatility': 0.5}
    }
    
    for name, allocation in portfolio_configs.items():
        portfolio = Portfolio(name, allocation, 100000)
        history = quarterly_rebalancing_backtest(portfolio, stock_data, sp500, '2021-01-01', '2024-12-31')
        if len(history) > 0:
            metrics = calculate_metrics(history)
            clustering_results[name] = metrics
            clustering_histories[name] = history
    
    print("\n[5/6] Backtesting ML-driven portfolios...")
    
    model = train_ml_model(stock_data, all_features, train_end_date='2020-12-31')
    
    ml_results = {}
    ml_histories = {}
    
    ml_configs = {
        'Conservative': {'high-predicted': 0.1, 'medium-predicted': 0.3, 'low-predicted': 0.6},
        'Balanced': {'high-predicted': 0.33, 'medium-predicted': 0.34, 'low-predicted': 0.33},
        'Aggressive': {'high-predicted': 0.6, 'medium-predicted': 0.3, 'low-predicted': 0.1}
    }
    
    for name, allocation in ml_configs.items():
        portfolio = Portfolio(name, allocation, 100000)
        history = ml_backtest(portfolio, stock_data, all_features, model, '2021-01-01', '2024-12-31')
        if len(history) > 0:
            metrics = calculate_metrics(history)
            ml_results[name] = metrics
            ml_histories[name] = history
    
    print("\n[6/6] Calculating benchmark and creating visualizations...")
    sp500_2021 = sp500.loc['2021-01-04':'2024-12-30']['Close']
    sp500_return = float((sp500_2021.iloc[-1] / sp500_2021.iloc[0]) - 1)
    sp500_returns = sp500_2021.pct_change()
    sp500_volatility = float(sp500_returns.std() * np.sqrt(252))
    sp500_cagr = float((1 + sp500_return) ** (1 / 4) - 1)
    sp500_sharpe = float((sp500_cagr - 0.02) / sp500_volatility)
    
    sp500_benchmark = {
        'return': sp500_return,
        'cagr': sp500_cagr,
        'volatility': sp500_volatility,
        'sharpe': sp500_sharpe
    }
    
    print("  Creating visualizations...")
    create_visualizations(clustering_results, ml_results, sp500_benchmark,
                         clustering_histories, ml_histories, sp500_2021)
    
    # Text summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    comparison_data = []
    for name in ['Conservative', 'Balanced', 'Aggressive']:
        clustering_return = clustering_results[name]['total_return']
        ml_return = ml_results[name]['total_return']
        improvement = ml_return - clustering_return
        
        comparison_data.append({
            'Portfolio': name,
            'Clustering': f"{clustering_return:.2%}",
            'ML-Driven': f"{ml_return:.2%}",
            'Improvement': f"+{improvement:.2%}",
            'vs S&P 500': f"+{ml_return - sp500_return:.2%}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE! Visualizations saved in results/figures/")
    print("="*80)
    
    return {
        'clustering': clustering_results,
        'ml': ml_results,
        'sp500': sp500_benchmark
    }


if __name__ == "__main__":
    results = run_complete_evaluation()

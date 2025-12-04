"""
Dynamic Portfolio Clustering Project - Main Entry Point
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

from data_loader import load_stock_data, FINAL_TICKERS
from feature_engineering import calculate_all_features
from clustering import (prepare_feature_matrix, standardize_features, apply_pca, 
                       perform_kmeans, perform_gmm, label_clusters_by_volatility)
from portfolio import create_portfolios
from backtesting import quarterly_rebalancing_backtest, backtest_ml_portfolio
from ml_models import train_all_models, evaluate_models

print("="*80)
print("DYNAMIC PORTFOLIO CLUSTERING PROJECT")
print("="*80)
print("\nThis will take approximately 10-15 minutes to complete.")
print("Starting analysis...")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1/7: LOADING DATA")
print("="*80)

print(f"\nLoading {len(FINAL_TICKERS)} stocks from 2015-2024...")
stock_data = load_stock_data(FINAL_TICKERS)

print("\nDownloading S&P 500 benchmark...")
sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
print(f"✓ Loaded {len(sp500)} days of S&P 500 data")

# ============================================================================
# STEP 2: CALCULATE FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 2/7: CALCULATING FEATURES")
print("="*80)

print("\nCalculating 10 features for each stock (252-day rolling window)...")
print("Features: return, volatility, sharpe, max_drawdown, beta, correlation, 4 momentum")

stock_features_dict = {}
for i, ticker in enumerate(stock_data.keys(), 1):
    print(f"  [{i}/{len(stock_data)}] Processing {ticker}...", end='')
    try:
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        stock_features_dict[ticker] = features
        print(f" ✓ ({len(features)} data points)")
    except Exception as e:
        print(f" ERROR: {e}")

print(f"\n✓ Calculated features for {len(stock_features_dict)} stocks")

# ============================================================================
# STEP 3: CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3/7: CLUSTERING ANALYSIS")
print("="*80)

# Prepare feature matrix
print("\nPreparing feature matrix (using most recent values)...")
feature_matrix = pd.DataFrame({
    ticker: features.iloc[-1] 
    for ticker, features in stock_features_dict.items()
}).T
print(f"✓ Feature matrix: {len(feature_matrix)} stocks × {len(feature_matrix.columns)} features")

# Standardize features
print("\nStandardizing features...")
feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
print(f"✓ Standardized {len(feature_cols)} features")

# Apply PCA
print("\nApplying PCA for dimensionality reduction...")
X_pca, pca = apply_pca(X_scaled, explained_variance_threshold=0.95)

# Perform K-means
print("\nPerforming K-means clustering (k=3)...")
kmeans_model, kmeans_labels, kmeans_score = perform_kmeans(X_pca, n_clusters=3)
print(f"✓ K-means silhouette score: {kmeans_score:.3f}")

# Perform GMM
print("\nPerforming GMM clustering (k=3)...")
gmm_model, gmm_labels, gmm_score = perform_gmm(X_pca, n_components=3)
print(f"✓ GMM silhouette score: {gmm_score:.3f}")

# Label clusters by volatility
print("\nLabeling clusters by volatility...")
cluster_volatilities = feature_matrix.groupby(kmeans_labels)['volatility'].mean().sort_values()
cluster_map = {
    cluster_volatilities.index[0]: 'low-volatility',
    cluster_volatilities.index[1]: 'moderate',
    cluster_volatilities.index[2]: 'high-volatility'
}
cluster_assignments = {ticker: cluster_map[label] for ticker, label in zip(feature_matrix.index, kmeans_labels)}

print(f"✓ Stocks clustered into: {list(cluster_map.values())}")
for cluster_name in ['high-volatility', 'moderate', 'low-volatility']:
    count = sum(1 for c in cluster_assignments.values() if c == cluster_name)
    print(f"  {cluster_name}: {count} stocks")

# ============================================================================
# STEP 4: BACKTEST CLUSTERING-BASED PORTFOLIOS
# ============================================================================
print("\n" + "="*80)
print("STEP 4/7: BACKTESTING CLUSTERING-BASED PORTFOLIOS")
print("="*80)

print("\nBacktesting 3 portfolios with quarterly rebalancing (2021-2024)...")
print("Transaction costs: 0.15% per trade\n")

portfolios = create_portfolios()
clustering_results = {}

for name, portfolio in portfolios.items():
    print(f"\n--- {name} Portfolio (Clustering-Based) ---")
    history = quarterly_rebalancing_backtest(portfolio, stock_data, stock_features_dict, sp500, '2021-01-01', '2024-12-31')
    clustering_results[name] = history

# ============================================================================
# STEP 5: TRAIN ALL ML MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 5/7: TRAINING & EVALUATING ML MODELS")
print("="*80)

# Add cluster assignments to features
for ticker, features_df in stock_features_dict.items():
    if ticker in cluster_assignments:
        features_df['cluster'] = cluster_assignments[ticker]

# Train all models
ml_models = train_all_models(stock_features_dict, stock_data, train_end_date='2020-12-31')

# Evaluate
evaluation_df = evaluate_models(ml_models, stock_features_dict, stock_data, test_start_date='2021-01-01')

print("\n" + "="*80)
print("ML MODEL EVALUATION RESULTS")
print("="*80)
print("\n" + evaluation_df.to_string(index=False))

# Find best model
enhanced_models = evaluation_df[evaluation_df['Version'] == 'Enhanced']
best_model_name = enhanced_models.loc[enhanced_models['R²'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name} (Enhanced)")

# ============================================================================
# STEP 6: BACKTEST ML-DRIVEN PORTFOLIOS
# ============================================================================
print("\n" + "="*80)
print(f"STEP 6/7: BACKTESTING ML-DRIVEN PORTFOLIOS")
print("="*80)

best_model_data = ml_models[best_model_name]['enhanced']

ml_portfolios = create_portfolios()
ml_results = {}

for name, portfolio in ml_portfolios.items():
    print(f"\n--- {name} Portfolio (ML-Driven) ---")
    history = backtest_ml_portfolio(
        portfolio, stock_data, stock_features_dict,
        best_model_data['model'], best_model_data['scaler'], 
        best_model_data['feature_cols'],
        '2021-01-01', '2024-12-31'
    )
    ml_results[name] = history

# ============================================================================
# STEP 7: FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 7/7: FINAL COMPARISON")
print("="*80)

sp500_2021 = sp500.loc['2021-01-04':'2024-12-30']['Close']
if isinstance(sp500_2021, pd.DataFrame):
    sp500_2021 = sp500_2021.iloc[:, 0]

sp500_return = float((sp500_2021.iloc[-1] / sp500_2021.iloc[0]) - 1)
sp500_returns = sp500_2021.pct_change()
sp500_volatility = float(sp500_returns.std() * np.sqrt(252))
sp500_cagr = float((1 + sp500_return) ** (1 / 4) - 1)
sp500_sharpe = float((sp500_cagr - 0.02) / sp500_volatility)

print("\nS&P 500 Benchmark:")
print(f"  Total Return:  {sp500_return:>8.2%}")
print(f"  CAGR:          {sp500_cagr:>8.2%}")
print(f"  Sharpe Ratio:  {sp500_sharpe:>8.2f}")

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print("\nPortfolio       Clustering      ML-Driven       Improvement     vs S&P 500")
print("-"*80)

for name in ['Conservative', 'Balanced', 'Aggressive']:
    clust = clustering_results[name]['total_return']
    ml = ml_results[name]['total_return']
    improvement = ml - clust
    vs_sp500 = ml - sp500_return
    print(f"{name:15s} {clust:>12.2%} {ml:>12.2%} {improvement:>12.2%} {vs_sp500:>12.2%}")

print("-"*80)
print(f"{'S&P 500':15s} {sp500_return:>12.2%}")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)

"""
⚡ COMPLETE 5-STOCK TEST - ALL 7 STEPS
"""
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

from data_loader import load_stock_data
from feature_engineering import calculate_all_features
from clustering import (prepare_feature_matrix, standardize_features, apply_pca, 
                       perform_kmeans, perform_gmm)
from portfolio import create_portfolios
from backtesting import quarterly_rebalancing_backtest, backtest_ml_portfolio
from ml_models import train_all_models, evaluate_models

# Use 5 stocks for speed
TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']

print("="*80)
print("⚡ COMPLETE 5-STOCK TEST - ALL 7 STEPS")
print("="*80)
print(f"\nStocks: {', '.join(TEST_TICKERS)}")
print("Expected runtime: ~3-4 minutes\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1/7: LOADING DATA")
print("="*80)

print(f"\nLoading {len(TEST_TICKERS)} stocks...")
stock_data = load_stock_data(TEST_TICKERS)
print(f"✓ Loaded {len(stock_data)} stocks\n")

print("Loading S&P 500 benchmark...")
sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
print(f"✓ Loaded S&P 500")

# ============================================================================
# STEP 2: CALCULATE FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 2/7: CALCULATING FEATURES")
print("="*80)
print()

stock_features_dict = {}
for i, ticker in enumerate(stock_data.keys(), 1):
    print(f"  [{i}/{len(stock_data)}] Processing {ticker}...", end=' ')
    try:
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        stock_features_dict[ticker] = features
        print(f"✓ {len(features)} data points")
    except Exception as e:
        print(f"✗ ERROR: {e}")

print(f"\n✓ Features calculated for {len(stock_features_dict)} stocks")

# ============================================================================
# STEP 3: CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3/7: CLUSTERING ANALYSIS")
print("="*80)

feature_matrix = pd.DataFrame({
    ticker: features.iloc[-1] 
    for ticker, features in stock_features_dict.items()
}).T
print(f"\n✓ Feature matrix: {len(feature_matrix)} stocks × {len(feature_matrix.columns)} features")

feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
print(f"✓ Standardized features")

X_pca, pca = apply_pca(X_scaled, explained_variance_threshold=0.95)
print(f"✓ PCA applied")

kmeans_model, kmeans_labels, kmeans_score = perform_kmeans(X_pca, n_clusters=3)
print(f"✓ K-means clustering (score: {kmeans_score:.3f})")

gmm_model, gmm_labels, gmm_score = perform_gmm(X_pca, n_components=3)
print(f"✓ GMM clustering (score: {gmm_score:.3f})")

cluster_volatilities = feature_matrix.groupby(kmeans_labels)['volatility'].mean().sort_values()
cluster_map = {
    cluster_volatilities.index[0]: 'low-volatility',
    cluster_volatilities.index[1]: 'moderate',
    cluster_volatilities.index[2]: 'high-volatility'
}
cluster_assignments = {ticker: cluster_map[label] for ticker, label in zip(feature_matrix.index, kmeans_labels)}

print(f"\n✓ Cluster distribution:")
for cluster_name in ['high-volatility', 'moderate', 'low-volatility']:
    count = sum(1 for c in cluster_assignments.values() if c == cluster_name)
    stocks = [t for t, c in cluster_assignments.items() if c == cluster_name]
    print(f"  {cluster_name}: {count} stocks ({', '.join(stocks)})")

# ============================================================================
# STEP 4: BACKTEST CLUSTERING PORTFOLIOS
# ============================================================================
print("\n" + "="*80)
print("STEP 4/7: BACKTESTING CLUSTERING-BASED PORTFOLIOS")
print("="*80)
print("\n(Quarterly rebalancing 2021-2024, transaction costs: 0.15%)\n")

portfolios = create_portfolios()
clustering_results = {}

for name, portfolio in portfolios.items():
    print(f"--- {name} Portfolio (Clustering-Based) ---")
    history = quarterly_rebalancing_backtest(portfolio, stock_data, stock_features_dict, sp500, '2021-01-01', '2024-12-31')
    clustering_results[name] = history
    print()

# ============================================================================
# STEP 5: TRAIN ML MODELS
# ============================================================================
print("="*80)
print("STEP 5/7: TRAINING ML MODELS")
print("="*80)

# Add cluster assignments
for ticker, features_df in stock_features_dict.items():
    if ticker in cluster_assignments:
        features_df['cluster'] = cluster_assignments[ticker]

print("\nTraining all 4 models (Ridge, RF, XGBoost, Neural Net)...")
ml_models = train_all_models(stock_features_dict, stock_data, train_end_date='2020-12-31')
print("\n✓ All models trained")

# ============================================================================
# STEP 6: EVALUATE ML MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 6/7: EVALUATING ML MODELS")
print("="*80)

evaluation_df = evaluate_models(ml_models, stock_features_dict, stock_data, test_start_date='2021-01-01')

print("\n📊 MODEL PERFORMANCE METRICS:")
print("-"*80)
print(evaluation_df.to_string(index=False))
print("-"*80)

enhanced_models = evaluation_df[evaluation_df['Version'] == 'Enhanced']
best_model_name = enhanced_models.loc[enhanced_models['R²'].idxmax(), 'Model']
best_r2 = enhanced_models['R²'].max()

print(f"\n🏆 BEST MODEL: {best_model_name} (Enhanced) with R² = {best_r2:.4f}")

# ============================================================================
# STEP 7: BACKTEST ML PORTFOLIOS
# ============================================================================
print("\n" + "="*80)
print("STEP 7/7: BACKTESTING ML-DRIVEN PORTFOLIOS")
print("="*80)
print(f"\nUsing {best_model_name} (Enhanced) for predictions\n")

best_model_data = ml_models[best_model_name]['enhanced']
ml_portfolios = create_portfolios()
ml_results = {}

for name, portfolio in ml_portfolios.items():
    print(f"--- {name} Portfolio (ML-Driven) ---")
    history = backtest_ml_portfolio(
        portfolio, stock_data, stock_features_dict,
        best_model_data['model'], best_model_data['scaler'], 
        best_model_data['feature_cols'],
        '2021-01-01', '2024-12-31'
    )
    ml_results[name] = history
    print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("="*80)
print("FINAL COMPARISON")
print("="*80)

# Calculate S&P 500 benchmark
sp500_2021 = sp500.loc['2021-01-04':'2024-12-30']['Close']
if isinstance(sp500_2021, pd.DataFrame):
    sp500_2021 = sp500_2021.iloc[:, 0]

sp500_return = float((sp500_2021.iloc[-1] / sp500_2021.iloc[0]) - 1)

print(f"\nS&P 500 Benchmark (2021-2024): {sp500_return:.2%}")

print("\n" + "="*80)
print("PORTFOLIO COMPARISON TABLE")
print("="*80)
print("\nPortfolio       Clustering    ML-Driven     Improvement   vs S&P 500")
print("-"*75)

for name in ['Conservative', 'Balanced', 'Aggressive']:
    clust = clustering_results[name]['total_return']
    ml = ml_results[name]['total_return']
    improvement = ml - clust
    vs_sp500 = ml - sp500_return
    
    print(f"{name:15s} {clust:>10.2%}   {ml:>10.2%}   {improvement:>10.2%}    {vs_sp500:>10.2%}")

print("-"*75)
print(f"{'S&P 500':15s} {sp500_return:>10.2%}")

print("\n" + "="*80)
print("✅ COMPLETE 5-STOCK TEST FINISHED!")
print("="*80)
print("\n🎯 Ready for full 50-stock analysis:")
print("   python main.py")
print()


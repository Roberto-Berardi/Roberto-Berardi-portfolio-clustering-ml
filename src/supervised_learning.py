"""
Supervised learning module - predict future returns using cluster features + MOMENTUM
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def prepare_supervised_data(stock_data, all_features, cluster_assignments, forward_months=3):
    """
    Prepare data for supervised learning.
    Target: forward returns over next N months
    """
    data_rows = []
    
    for ticker in stock_data.keys():
        if ticker not in all_features or ticker not in cluster_assignments:
            continue
        
        stock_prices = stock_data[ticker]['Close']
        features_df = all_features[ticker]
        
        # For each feature row, calculate forward return
        for idx in range(len(features_df)):
            date = features_df.index[idx]
            
            # Get current features
            current_features = features_df.iloc[idx].to_dict()
            
            # Calculate forward return (3 months = ~63 trading days)
            forward_days = forward_months * 21
            date_idx = stock_prices.index.get_loc(date)
            
            if date_idx + forward_days < len(stock_prices):
                current_price = stock_prices.iloc[date_idx]
                future_price = stock_prices.iloc[date_idx + forward_days]
                forward_return = (future_price / current_price) - 1
                
                # Add to dataset
                row = current_features.copy()
                row['ticker'] = ticker
                row['date'] = date
                row['cluster'] = cluster_assignments[ticker]
                row['forward_return'] = forward_return
                
                data_rows.append(row)
    
    return pd.DataFrame(data_rows)


def split_train_test(data, train_end_date='2020-12-31'):
    """Split data into training (2015-2020) and testing (2021-2024)."""
    data['date'] = pd.to_datetime(data['date'])
    train_end = pd.to_datetime(train_end_date)
    
    train_data = data[data['date'] <= train_end].copy()
    test_data = data[data['date'] > train_end].copy()
    
    return train_data, test_data


def train_base_models(X_train, y_train, X_test, y_test):
    """Train base models (features only, no clusters)."""
    results = {}
    
    print("  Training base models (features + momentum, no clusters)...")
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results['Ridge_Base'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['RandomForest_Base'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results['XGBoost_Base'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # Neural Network
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    results['NeuralNet_Base'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    return results


def train_enhanced_models(X_train, y_train, X_test, y_test):
    """Train enhanced models (features + momentum + cluster assignments)."""
    results = {}
    
    print("  Training enhanced models (features + momentum + clusters)...")
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results['Ridge_Enhanced'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['RandomForest_Enhanced'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results['XGBoost_Enhanced'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    # Neural Network
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    results['NeuralNet_Enhanced'] = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'directional_accuracy': np.mean((y_pred > 0) == (y_test > 0))
    }
    
    return results


def run_supervised_learning_experiment(stock_data, all_features, cluster_assignments):
    """Run complete supervised learning experiment."""
    
    print("\n" + "="*70)
    print("SUPERVISED LEARNING EXPERIMENT (WITH MOMENTUM)")
    print("="*70)
    
    # Step 1: Prepare data
    print("\nPreparing data...")
    data = prepare_supervised_data(stock_data, all_features, cluster_assignments, forward_months=3)
    print(f"  Total samples: {len(data)}")
    
    # Step 2: Split train/test
    train_data, test_data = split_train_test(data, train_end_date='2020-12-31')
    print(f"  Training samples (2015-2020): {len(train_data)}")
    print(f"  Testing samples (2021-2024): {len(test_data)}")
    
    # Step 3: Prepare features (NOW INCLUDING MOMENTUM)
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation',
                    'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    
    print(f"\n  Using {len(feature_cols)} features: {feature_cols}")
    
    # Base models (features + momentum)
    X_train_base = train_data[feature_cols].fillna(0)
    X_test_base = test_data[feature_cols].fillna(0)
    
    # Enhanced models (features + momentum + clusters)
    train_with_clusters = train_data.copy()
    test_with_clusters = test_data.copy()
    
    # One-hot encode clusters
    cluster_dummies_train = pd.get_dummies(train_with_clusters['cluster'], prefix='cluster')
    cluster_dummies_test = pd.get_dummies(test_with_clusters['cluster'], prefix='cluster')
    
    X_train_enhanced = pd.concat([train_data[feature_cols].fillna(0).reset_index(drop=True), 
                                   cluster_dummies_train.reset_index(drop=True)], axis=1)
    X_test_enhanced = pd.concat([test_data[feature_cols].fillna(0).reset_index(drop=True), 
                                  cluster_dummies_test.reset_index(drop=True)], axis=1)
    
    # Ensure same columns in train/test
    for col in X_train_enhanced.columns:
        if col not in X_test_enhanced.columns:
            X_test_enhanced[col] = 0
    X_test_enhanced = X_test_enhanced[X_train_enhanced.columns]
    
    # Target variable
    y_train = train_data['forward_return'].values
    y_test = test_data['forward_return'].values
    
    # Step 4: Train models
    print("\n" + "-"*70)
    base_results = train_base_models(X_train_base, y_train, X_test_base, y_test)
    
    print("\n" + "-"*70)
    enhanced_results = train_enhanced_models(X_train_enhanced, y_train, X_test_enhanced, y_test)
    
    # Step 5: Display results
    all_results = {**base_results, **enhanced_results}
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    
    print("\n" + results_df.to_string())
    
    # Compare base vs enhanced
    print("\n" + "="*70)
    print("BASE vs ENHANCED COMPARISON")
    print("="*70)
    
    for model_type in ['Ridge', 'RandomForest', 'XGBoost', 'NeuralNet']:
        base_key = f'{model_type}_Base'
        enhanced_key = f'{model_type}_Enhanced'
        
        if base_key in all_results and enhanced_key in all_results:
            base_r2 = all_results[base_key]['r2']
            enhanced_r2 = all_results[enhanced_key]['r2']
            improvement = enhanced_r2 - base_r2
            
            print(f"\n{model_type}:")
            print(f"  Base R²:     {base_r2:>8.4f}")
            print(f"  Enhanced R²: {enhanced_r2:>8.4f}")
            print(f"  Improvement: {improvement:>8.4f} {'✓' if improvement > 0 else '✗'}")
    
    print("\n" + "="*70)
    
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/files/portfolio-clustering-project/src')
    
    print("\n🧪 TESTING SUPERVISED LEARNING WITH MOMENTUM\n")
    
    from data_loader import TEST_TICKERS, load_stock_data
    from feature_engineering import calculate_all_features
    from clustering import prepare_feature_matrix, standardize_features, apply_pca, perform_kmeans, label_clusters_by_volatility
    import yfinance as yf
    
    print("Loading data...")
    stock_data = load_stock_data(TEST_TICKERS)
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
    
    print("Calculating features (with momentum)...")
    all_features = {}
    for ticker in TEST_TICKERS:
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        if len(features) > 0:
            all_features[ticker] = features
    
    print("Clustering stocks...")
    feature_matrix = prepare_feature_matrix(all_features, use_latest=True)
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
    X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
    X_pca, pca = apply_pca(X_scaled, explained_variance_threshold=0.95)
    kmeans, labels, silhouette = perform_kmeans(X_pca, n_clusters=3)
    cluster_names = label_clusters_by_volatility(feature_matrix, labels)
    
    cluster_assignments = {}
    for idx, ticker in enumerate(feature_matrix['ticker']):
        cluster_assignments[ticker] = cluster_names[labels[idx]]
    
    # Run experiment
    results = run_supervised_learning_experiment(stock_data, all_features, cluster_assignments)
    
    print("\n✓ Supervised learning experiment complete!")

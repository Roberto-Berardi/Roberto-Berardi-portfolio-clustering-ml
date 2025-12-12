"""
Machine Learning Models for Stock Return Prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy

print("Importing ML libraries...")
print("✓ scikit-learn, xgboost, pandas, numpy loaded")


def prepare_ml_data(stock_features_dict, stock_data_dict, end_date=None, forward_periods=63, include_clusters=False):
    """
    Prepare training data for ML models.
    
    Args:
        stock_features_dict: Dictionary of stock features DataFrames
        stock_data_dict: Dictionary of stock price DataFrames (for calculating forward returns)
        end_date: End date for training data (None = use all)
        forward_periods: Number of trading days to look ahead (63 ≈ 3 months/quarter)
        include_clusters: If True, include cluster assignments as features
    
    Returns:
        X: Feature matrix
        y: Target returns (forward-looking)
    """
    X_list = []
    y_list = []
    
    # Base features (always included)
    base_features = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 
                     'correlation', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    
    # Add cluster if requested
    feature_cols = base_features.copy()
    if include_clusters:
        feature_cols.append('cluster')
    
    for ticker, df in stock_features_dict.items():
        # Get corresponding price data
        if ticker not in stock_data_dict:
            continue
            
        price_data = stock_data_dict[ticker]
        
        if end_date:
            df = df[df.index <= end_date].copy()
            price_data = price_data[price_data.index <= end_date].copy()
        
        # Check if we have all required features
        available_features = [f for f in feature_cols if f in df.columns]
        if len(available_features) < len(base_features):
            continue
        
        # Drop rows with missing features
        df_clean = df[available_features].dropna()
        
        if len(df_clean) == 0:
            continue
        
        # Calculate forward returns using actual price data
        # Align price data with feature dates
        aligned_prices = price_data.loc[price_data.index.isin(df_clean.index), 'Close']
        
        if len(aligned_prices) == 0:
            continue
        
        # Shift prices backward to get future prices
        future_prices = aligned_prices.shift(-forward_periods)
        
        # Calculate forward returns
        forward_returns = (future_prices / aligned_prices) - 1
        
        # Only keep rows where we have both features and forward returns
        valid_idx = forward_returns.notna()
        
        # Make sure indices align
        common_idx = df_clean.index.intersection(forward_returns[valid_idx].index)
        
        if len(common_idx) == 0:
            continue
        
        X_data = df_clean.loc[common_idx]
        y_data = forward_returns.loc[common_idx]
        
        if len(X_data) > 0:
            # If including clusters, encode them as numeric
            if include_clusters and 'cluster' in X_data.columns:
                cluster_map = {'low-volatility': 0, 'moderate': 1, 'high-volatility': 2}
                X_data = X_data.copy()
                X_data['cluster'] = X_data['cluster'].map(cluster_map).fillna(1)
            
            X_list.append(X_data)
            y_list.append(y_data)
    
    if not X_list:
        return None, None
    
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    
    return X, y


def train_all_models(stock_features_dict, stock_data_dict, train_end_date='2020-12-31'):
    """
    Train all 4 ML models in both base and enhanced versions.
    
    Args:
        stock_features_dict: Dictionary of stock features
        stock_data_dict: Dictionary of stock price data
        train_end_date: End date for training period
    
    Returns:
        Dictionary with all trained models
    """
    print("="*80)
    print("TRAINING ML MODELS (BASE AND ENHANCED VERSIONS)")
    print("="*80)
    
    # Prepare BASE model data (no clusters)
    print("\nPreparing BASE model data (10 features, no clusters)...")
    X_base, y_base = prepare_ml_data(stock_features_dict, stock_data_dict, 
                                      end_date=train_end_date, include_clusters=False)
    
    if X_base is None:
        raise ValueError("No valid training data available")
    
    print(f"✓ Base training data: {len(X_base)} samples, {X_base.shape[1]} features")
    
    # Prepare ENHANCED model data (with clusters)
    print("\nPreparing ENHANCED model data (11 features, includes cluster)...")
    X_enhanced, y_enhanced = prepare_ml_data(stock_features_dict, stock_data_dict,
                                              end_date=train_end_date, include_clusters=True)
    
    if X_enhanced is None:
        print("⚠️  Warning: No enhanced data, using base data only")
        X_enhanced, y_enhanced = X_base, y_base
    
    print(f"✓ Enhanced training data: {len(X_enhanced)} samples, {X_enhanced.shape[1]} features")
    
    # Standardize features
    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)
    
    scaler_enhanced = StandardScaler()
    X_enhanced_scaled = scaler_enhanced.fit_transform(X_enhanced)
    
    # Define models
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
    }
    
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        # Train BASE version
        print(f"    - Base version...", end=' ')
        model_base = deepcopy(model)
        # Remove NaN values and convert to numpy
        valid_idx_base = ~y_base.isna()
        mask = valid_idx_base.to_numpy()
        X_base_clean = X_base_scaled[mask]
        y_base_clean = y_base[mask]
        model_base.fit(X_base_clean, y_base_clean)
        print("✓")
        
        # Train ENHANCED version
        print(f"    - Enhanced version...", end=' ')
        model_enhanced = deepcopy(model)
        # Remove NaN values and convert to numpy
        valid_idx_enh = ~y_enhanced.isna()
        mask_enh = valid_idx_enh.to_numpy()
        X_enhanced_clean = X_enhanced_scaled[mask_enh]
        y_enhanced_clean = y_enhanced[mask_enh]
        model_enhanced.fit(X_enhanced_clean, y_enhanced_clean)
        print("✓")
        
        results[name] = {
            'base': {
                'model': model_base,
                'scaler': scaler_base,
                'feature_cols': list(X_base.columns)
            },
            'enhanced': {
                'model': model_enhanced,
                'scaler': scaler_enhanced,
                'feature_cols': list(X_enhanced.columns)
            }
        }
    
    print("\n✓ All models trained successfully")
    return results


def evaluate_models(ml_models, stock_features_dict, stock_data_dict, test_start_date='2021-01-01'):
    """
    Evaluate all models on test data.
    
    Returns:
        DataFrame with R², MSE, and Directional Accuracy for each model
    """
    print("\n" + "="*80)
    print("EVALUATING MODELS ON TEST DATA")
    print("="*80)
    
    results = []
    
    for model_name, model_dict in ml_models.items():
        for version in ['base', 'enhanced']:
            print(f"\nEvaluating {model_name} ({version})...", end=' ')
            
            model_data = model_dict[version]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            
            # Prepare test data
            include_clusters = (version == 'enhanced' and 'cluster' in feature_cols)
            X_test, y_test = prepare_ml_data(stock_features_dict, stock_data_dict,
                                             end_date=None, include_clusters=include_clusters)
            
            if X_test is None:
                print("No test data!")
                continue
            
            # Filter to test period
            X_test = X_test[X_test.index >= test_start_date]
            y_test = y_test[y_test.index >= test_start_date]
            
            if len(X_test) == 0:
                print("No test samples!")
                continue
            
            # Make predictions
            X_test_scaled = scaler.transform(X_test[feature_cols])
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Directional accuracy
            y_test_direction = (y_test > 0).astype(int)
            y_pred_direction = (y_pred > 0).astype(int)
            directional_accuracy = (y_test_direction == y_pred_direction).mean()
            
            results.append({
                'Model': model_name,
                'Version': version.capitalize(),
                'R²': r2,
                'MSE': mse,
                'Directional Accuracy': directional_accuracy
            })
            
            print(f"R²={r2:.4f}, MSE={mse:.4f}, Dir Acc={directional_accuracy:.2%}")
    
    return pd.DataFrame(results)


def predict_returns(model, scaler, features, feature_cols):
    """
    Predict returns using a trained model.
    """
    X = features[feature_cols]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return pd.Series(predictions, index=features.index)

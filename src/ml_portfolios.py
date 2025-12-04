"""
ML-driven portfolios - use predictions to select stocks
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def train_ml_model(stock_data, all_features, train_end_date='2020-12-31'):
    """Train ML model on historical data."""
    # Prepare training data
    data_rows = []
    
    for ticker in stock_data.keys():
        if ticker not in all_features:
            continue
        
        stock_prices = stock_data[ticker]['Close']
        features_df = all_features[ticker]
        
        for idx in range(len(features_df)):
            date = features_df.index[idx]
            
            # Only use data up to train_end_date
            if pd.to_datetime(date) > pd.to_datetime(train_end_date):
                continue
            
            current_features = features_df.iloc[idx].to_dict()
            
            # Calculate forward return (3 months)
            forward_days = 63
            date_idx = stock_prices.index.get_loc(date)
            
            if date_idx + forward_days < len(stock_prices):
                current_price = stock_prices.iloc[date_idx]
                future_price = stock_prices.iloc[date_idx + forward_days]
                forward_return = (future_price / current_price) - 1
                
                row = current_features.copy()
                row['ticker'] = ticker
                row['date'] = date
                row['forward_return'] = forward_return
                
                data_rows.append(row)
    
    train_data = pd.DataFrame(data_rows)
    
    # Train XGBoost model
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation',
                    'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['forward_return'].values
    
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    print(f"  Trained ML model on {len(train_data)} samples")
    
    return model


def predict_returns(model, all_features, date):
    """Predict returns for all stocks at a given date."""
    predictions = {}
    
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation',
                    'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    
    for ticker, features_df in all_features.items():
        # Find closest date
        if date in features_df.index:
            features = features_df.loc[date][feature_cols].fillna(0).values.reshape(1, -1)
            predicted_return = model.predict(features)[0]
            predictions[ticker] = predicted_return
        elif date < features_df.index[0]:
            continue
        else:
            # Use most recent available
            available_dates = features_df.index[features_df.index <= date]
            if len(available_dates) > 0:
                latest_date = available_dates[-1]
                features = features_df.loc[latest_date][feature_cols].fillna(0).values.reshape(1, -1)
                predicted_return = model.predict(features)[0]
                predictions[ticker] = predicted_return
    
    return predictions


def create_ml_portfolio_allocation(predictions):
    """
    Create portfolio allocation based on ML predictions.
    Rank stocks by predicted return and create groups.
    """
    if len(predictions) == 0:
        return {}
    
    # Sort stocks by predicted return
    sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Divide into thirds
    n_stocks = len(sorted_stocks)
    top_third = n_stocks // 3
    
    allocations = {}
    
    for i, (ticker, pred) in enumerate(sorted_stocks):
        if i < top_third:
            allocations[ticker] = 'high-predicted'
        elif i < 2 * top_third:
            allocations[ticker] = 'medium-predicted'
        else:
            allocations[ticker] = 'low-predicted'
    
    return allocations


def ml_backtest(portfolio, stock_data, all_features, model, start_date='2021-01-01', end_date='2024-12-31'):
    """Backtest using ML predictions with quarterly rebalancing."""
    first_ticker = list(stock_data.keys())[0]
    all_data = stock_data[first_ticker]
    dates = all_data.index
    
    # Find date range
    start_idx = None
    end_idx = None
    
    for i in range(len(dates)):
        date_str = str(dates[i])[:10]
        if start_idx is None and date_str >= start_date:
            start_idx = i
        if date_str <= end_date:
            end_idx = i
    
    if start_idx is None or end_idx is None:
        print("    Error: Could not find date range")
        return pd.DataFrame()
    
    print(f"    Backtesting from {str(dates[start_idx])[:10]} to {str(dates[end_idx])[:10]}")
    
    # Find quarter-end dates
    quarter_ends = []
    last_quarter = None
    
    for i in range(start_idx, end_idx + 1):
        date = dates[i]
        year = date.year
        month = date.month
        day = date.day
        
        current_quarter = (year, (month - 1) // 3 + 1)
        
        if month in [3, 6, 9, 12] and day >= 28:
            if current_quarter != last_quarter:
                quarter_ends.append(i)
                last_quarter = current_quarter
    
    print(f"    Found {len(quarter_ends)} quarter-end rebalancing dates")
    
    # Initial prediction and allocation
    print(f"    Initial ML predictions...")
    predictions = predict_returns(model, all_features, dates[start_idx])
    ml_allocation = create_ml_portfolio_allocation(predictions)
    
    # Get initial prices
    prices = {}
    for ticker in ml_allocation.keys():
        if ticker in stock_data:
            try:
                price = stock_data[ticker].iloc[start_idx]['Close']
                if pd.notna(price):
                    prices[ticker] = float(price)
            except:
                continue
    
    # Initial rebalance
    costs = portfolio.rebalance(prices, ml_allocation, 0.0015)
    print(f"    Initial rebalance cost: ${costs:,.2f}")
    
    rebalance_count = 1
    next_rebalance_idx = 0
    
    # Daily tracking
    for i in range(start_idx, end_idx + 1):
        date = dates[i]
        
        # Check if we need to rebalance
        if next_rebalance_idx < len(quarter_ends) and i >= quarter_ends[next_rebalance_idx]:
            print(f"    Rebalancing on {str(date)[:10]} (Quarter {rebalance_count + 1})")
            
            # Get new ML predictions
            predictions = predict_returns(model, all_features, date)
            ml_allocation = create_ml_portfolio_allocation(predictions)
            
            # Rebalance
            prices = {}
            for ticker in ml_allocation.keys():
                if ticker in stock_data:
                    try:
                        price = stock_data[ticker].iloc[i]['Close']
                        if pd.notna(price):
                            prices[ticker] = float(price)
                    except:
                        continue
            
            costs = portfolio.rebalance(prices, ml_allocation, 0.0015)
            print(f"      Cost: ${costs:,.2f}, Value: ${portfolio.current_value:,.2f}")
            
            rebalance_count += 1
            next_rebalance_idx += 1
        
        # Daily update
        prices = {}
        for ticker in ml_allocation.keys():
            if ticker in stock_data:
                try:
                    price = stock_data[ticker].iloc[i]['Close']
                    if pd.notna(price):
                        prices[ticker] = float(price)
                except:
                    continue
        
        portfolio.update_value(prices, date=date)
    
    print(f"    Total rebalances: {rebalance_count}")
    print(f"    Final value: ${portfolio.current_value:,.2f}")
    
    return portfolio.get_history_df()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/files/portfolio-clustering-project/src')
    
    print("\n🧪 TESTING ML-DRIVEN PORTFOLIOS\n")
    
    from data_loader import TEST_TICKERS, load_stock_data
    from feature_engineering import calculate_all_features
    from portfolio import Portfolio
    from backtesting import calculate_metrics
    import yfinance as yf
    
    print("Loading data...")
    stock_data = load_stock_data(TEST_TICKERS)
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
    
    print("Calculating features...")
    all_features = {}
    for ticker in TEST_TICKERS:
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        if len(features) > 0:
            all_features[ticker] = features
    
    print("\nTraining ML model...")
    model = train_ml_model(stock_data, all_features, train_end_date='2020-12-31')
    
    # Test all three portfolio types
    portfolio_configs = {
        'Conservative': {'high-predicted': 0.1, 'medium-predicted': 0.3, 'low-predicted': 0.6},
        'Balanced': {'high-predicted': 0.33, 'medium-predicted': 0.34, 'low-predicted': 0.33},
        'Aggressive': {'high-predicted': 0.6, 'medium-predicted': 0.3, 'low-predicted': 0.1}
    }
    
    print("\n" + "="*70)
    print("ML-DRIVEN PORTFOLIO BACKTEST (2021-2024)")
    print("="*70)
    
    for name, allocation in portfolio_configs.items():
        print(f"\n{name} Portfolio (ML-driven):")
        portfolio = Portfolio(name, allocation, 100000)
        history = ml_backtest(portfolio, stock_data, all_features, model, '2021-01-01', '2024-12-31')
        
        if len(history) > 0:
            metrics = calculate_metrics(history)
            
            if metrics:
                print(f"  Total Return:  {metrics['total_return']:>8.2%}")
                print(f"  CAGR:          {metrics['cagr']:>8.2%}")
                print(f"  Volatility:    {metrics['volatility']:>8.2%}")
                print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:>8.2f}")
                print(f"  Max Drawdown:  {metrics['max_drawdown']:>8.2%}")
                print(f"  Final Value:   ${metrics['final_value']:>12,.2f}")
    
    print("\n" + "="*70)
    print("✓ ML portfolio backtest complete!")
    print("="*70)

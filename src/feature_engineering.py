"""
Feature engineering module for portfolio clustering project.
Calculates risk-return metrics for each stock.
"""

import pandas as pd
import numpy as np


def calculate_all_features(stock_data, sp500_data, window=252):
    """
    Calculate risk-return features for a stock.
    Now includes momentum features!
    """
    # Handle MultiIndex columns from yfinance
    close_data = stock_data['Close']
    if isinstance(close_data, pd.DataFrame):
        stock_prices = close_data.iloc[:, 0].copy()
    else:
        stock_prices = close_data.copy()
    
    if isinstance(sp500_data, pd.DataFrame):
        if 'Close' in sp500_data.columns:
            sp500_close = sp500_data['Close']
            if isinstance(sp500_close, pd.DataFrame):
                sp500_prices = sp500_close.iloc[:, 0]
            else:
                sp500_prices = sp500_close
        else:
            sp500_prices = sp500_data.iloc[:, 0]
    else:
        sp500_prices = sp500_data
    
    # Flatten if multi-index
    if isinstance(sp500_prices, pd.DataFrame):
        sp500_prices = sp500_prices.iloc[:, 0]
    
    # Align data
    common_dates = stock_prices.index.intersection(sp500_prices.index)
    stock_prices = stock_prices.loc[common_dates]
    sp500_prices = sp500_prices.loc[common_dates]
    
    print(f"  Aligned data: {len(stock_prices)} trading days")
    
    if len(stock_prices) < window:
        return pd.DataFrame()
    
    # Calculate returns
    stock_returns = stock_prices.pct_change()
    sp500_returns = sp500_prices.pct_change()
    
    # Rolling calculations
    features_list = []
    
    for i in range(window, len(stock_prices)):
        window_stock_returns = stock_returns.iloc[i-window:i]
        window_sp500_returns = sp500_returns.iloc[i-window:i]
        
        # Skip if not enough data
        if len(window_stock_returns.dropna()) < window * 0.8:
            continue
        
        # Basic risk-return metrics
        total_return = (1 + window_stock_returns).prod() - 1
        years = window / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        volatility = window_stock_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        cumulative = (1 + window_stock_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Beta and correlation - use numpy to avoid indexing issues
        stock_array = window_stock_returns.dropna().values
        sp500_array = window_sp500_returns.dropna().values
        
        # Align arrays
        min_len = min(len(stock_array), len(sp500_array))
        stock_array = stock_array[:min_len]
        sp500_array = sp500_array[:min_len]
        
        if len(stock_array) > 1 and len(sp500_array) > 1:
            covariance = np.cov(stock_array, sp500_array)[0, 1]
            market_variance = np.var(sp500_array)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            correlation = np.corrcoef(stock_array, sp500_array)[0, 1]
        else:
            beta = 1.0
            correlation = 0.0
        
        # NEW: Momentum features
        # 1-month momentum (21 trading days)
        if i >= 21:
            momentum_1m = (stock_prices.iloc[i] / stock_prices.iloc[i-21]) - 1
        else:
            momentum_1m = 0
        
        # 3-month momentum (63 trading days)
        if i >= 63:
            momentum_3m = (stock_prices.iloc[i] / stock_prices.iloc[i-63]) - 1
        else:
            momentum_3m = 0
        
        # 6-month momentum (126 trading days)
        if i >= 126:
            momentum_6m = (stock_prices.iloc[i] / stock_prices.iloc[i-126]) - 1
        else:
            momentum_6m = 0
        
        # 12-month momentum (252 trading days)
        if i >= 252:
            momentum_12m = (stock_prices.iloc[i] / stock_prices.iloc[i-252]) - 1
        else:
            momentum_12m = 0
        
        features_list.append({
            'date': stock_prices.index[i],
            'return': annualized_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'correlation': correlation,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'momentum_12m': momentum_12m
        })
    
    features_df = pd.DataFrame(features_list)
    features_df.set_index('date', inplace=True)
    
    print(f"  Features after dropna: {len(features_df)} data points")
    
    return features_df


if __name__ == "__main__":
    print("Testing feature engineering...")
    from data_loader import TEST_TICKERS, load_stock_data
    import yfinance as yf
    
    stock_data = load_stock_data(TEST_TICKERS)
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
    
    ticker = 'AAPL'
    features = calculate_all_features(stock_data[ticker], sp500, window=252)
    
    print(f"\n{ticker} features:")
    print(features.tail())
    print(f"\nFeature columns: {features.columns.tolist()}")
    print("✓ Feature engineering test complete!")
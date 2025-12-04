"""
Data loading module for portfolio clustering project.
"""

import pandas as pd
import yfinance as yf
import os

# Top 50 S&P 500 stocks by market cap (as of Dec 31, 2020)
# Note: Using META ticker (Facebook changed ticker from FB to META in 2021)
#       BRK.B excluded due to data issues, replaced with ORCL
FINAL_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'JNJ', 'V',
    'WMT', 'JPM', 'MA', 'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'PYPL', 'BAC',
    'CMCSA', 'VZ', 'ADBE', 'NFLX', 'NKE', 'INTC', 'T', 'CRM', 'TMO', 'PFE',
    'CSCO', 'ABT', 'ABBV', 'CVX', 'XOM', 'MRK', 'KO', 'COST', 'PEP', 'AVGO',
    'ACN', 'MDT', 'NEE', 'TXN', 'UNP', 'LLY', 'DHR', 'QCOM', 'BMY', 'PM',
    'ORCL'
]

# Keep the 10-stock list for quick testing
TEST_TICKERS = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 'WMT', 'PG', 'DIS', 'BA', 'NEE']

def load_stock_data(tickers, start_date='2015-01-01', end_date='2024-12-31', data_dir='data'):
    """Download and cache stock data."""
    os.makedirs(data_dir, exist_ok=True)
    stock_data = {}
    
    print(f"Loading {len(tickers)} stocks...")
    
    for ticker in tickers:
        filepath = f'{data_dir}/{ticker}.csv'
        
        if os.path.exists(filepath):
            # Load from CSV - try with skiprows first, fallback without
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True, skiprows=[1, 2])
                # Verify we got numeric data
                if df['Close'].dtype == 'object':
                    # Try without skiprows
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            except:
                # If skiprows fails, try without
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Ensure Close is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df[df['Close'].notna()]
            
            stock_data[ticker] = df
            print(f"  ✓ Loaded {ticker} from cache")
        else:
            # Download fresh data
            print(f"  ↓ Downloading {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                df.to_csv(filepath)
                stock_data[ticker] = df
                print(f"  ✓ Downloaded {ticker}")
            else:
                print(f"  ✗ Failed to download {ticker}")
    
    print(f"\n✓ Loaded {len(stock_data)}/{len(tickers)} stocks successfully\n")
    
    return stock_data

if __name__ == "__main__":
    print("Testing data loader with 10 stocks...")
    stock_data = load_stock_data(TEST_TICKERS)
    print(f"Loaded {len(stock_data)} stocks")
    
    for ticker in TEST_TICKERS:
        if ticker in stock_data:
            print(f"{ticker}: {len(stock_data[ticker])} trading days")

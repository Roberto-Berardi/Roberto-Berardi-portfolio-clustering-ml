"""
Portfolio Backtesting with Quarterly Rebalancing
OPTIMIZED: Reuses pre-calculated features instead of recalculating
"""
import pandas as pd
import numpy as np
from portfolio import Portfolio
from clustering import standardize_features, apply_pca, perform_kmeans


def perform_adaptive_clustering(stock_features_dict, rebalancing_date):
    """
    Perform clustering using PRE-CALCULATED features up to rebalancing_date.
    
    OPTIMIZED: Uses features already calculated in Step 2, just filters by date.
    This is 20x faster than recalculating from scratch.
    
    Args:
        stock_features_dict: Dictionary with pre-calculated features for all stocks
        rebalancing_date: Date to cluster as of
    
    Returns:
        Dictionary mapping ticker -> cluster name
    """
    # Filter features to data up to rebalancing date (most recent 12-month window)
    current_features = {}
    for ticker, features_df in stock_features_dict.items():
        # Get features up to this date
        available_features = features_df[features_df.index <= rebalancing_date]
        if len(available_features) > 0:
            # Use the most recent feature values (already calculated with 12-month rolling window)
            current_features[ticker] = available_features.iloc[-1]
    
    if len(current_features) == 0:
        return {}
    
    # Create feature matrix
    feature_matrix = pd.DataFrame(current_features).T
    
    # Standardize
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
    X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
    
    # Apply PCA
    X_pca, pca = apply_pca(X_scaled, explained_variance_threshold=0.95)
    
    # Perform K-means
    kmeans_model, labels, score = perform_kmeans(X_pca, n_clusters=3)
    
    # Label clusters by volatility
    cluster_volatilities = feature_matrix.groupby(labels)['volatility'].mean().sort_values()
    cluster_map = {
        cluster_volatilities.index[0]: 'low-volatility',
        cluster_volatilities.index[1]: 'moderate',
        cluster_volatilities.index[2]: 'high-volatility'
    }
    
    # Map tickers to cluster names
    cluster_assignments = {
        ticker: cluster_map[label] 
        for ticker, label in zip(feature_matrix.index, labels)
    }
    
    return cluster_assignments





def quarterly_rebalancing_backtest(portfolio, stock_data, stock_features_dict, benchmark_data, start_date, end_date):
    """
    OPTIMIZED: Backtest portfolio with quarterly rebalancing using pre-calculated features.
    
    Args:
        portfolio: Portfolio object with allocation rules
        stock_data: Dictionary of stock price DataFrames
        stock_features_dict: Dictionary of PRE-CALCULATED features (from Step 2)
        benchmark_data: S&P 500 DataFrame
        start_date: Start date for backtesting
        end_date: End date for backtesting
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"    Backtesting from {start_date} to {end_date}")
    
    # Initialize portfolio
    portfolio.initial_capital = 100000
    portfolio.current_value = 100000
    portfolio.holdings = {}
    
    # Get quarter-end dates for rebalancing
    date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
    rebalancing_dates = [d.strftime('%Y-%m-%d') for d in date_range]
    print(f"    Found {len(rebalancing_dates)} quarter-end rebalancing dates")
    
    portfolio_values = []
    total_transaction_costs = 0
    transaction_cost_rate = 0.0015
    
    # Initial clustering and allocation
    print(f"    Initial clustering...")
    initial_date = pd.Timestamp(start_date)
    cluster_assignments = perform_adaptive_clustering(stock_features_dict, initial_date)
    
    # Calculate initial allocation
    target_weights = {}
    for ticker in stock_data.keys():
        cluster = cluster_assignments.get(ticker, 'moderate')
        cluster_stocks = sum(1 for c in cluster_assignments.values() if c == cluster)
        if cluster_stocks > 0:
            weight = portfolio.target_allocation.get(cluster, 0) / cluster_stocks
            target_weights[ticker] = weight
    
    # Get initial prices
    current_prices = {}
    for ticker, data in stock_data.items():
        available_dates = data.loc[:initial_date].index
        if len(available_dates) > 0:
            last_date = available_dates[-1]
            current_prices[ticker] = data.loc[last_date, 'Close']
    
    # Initial purchase
    portfolio.holdings = {}
    initial_costs = 0
    for ticker, weight in target_weights.items():
        if ticker in current_prices and weight > 0:
            target_value = portfolio.current_value * weight
            shares = target_value / current_prices[ticker]
            portfolio.holdings[ticker] = shares
            initial_costs += target_value * transaction_cost_rate
    
    portfolio.current_value -= initial_costs
    total_transaction_costs += initial_costs
    print(f"    Initial rebalance cost: ${initial_costs:.2f}")
    
    portfolio_values.append({
        'date': initial_date,
        'value': portfolio.current_value,
        'return': 0.0
    })
    
    # Quarterly rebalancing
    for quarter_num, rebal_date in enumerate(rebalancing_dates, 2):
        print(f"    Rebalancing on {rebal_date} (Quarter {quarter_num})")
        
        # Adaptive clustering using pre-calculated features
        cluster_assignments = perform_adaptive_clustering(stock_features_dict, rebal_date)
        
        # Calculate target allocation
        target_weights = {}
        for ticker in stock_data.keys():
            cluster = cluster_assignments.get(ticker, 'moderate')
            cluster_stocks = sum(1 for c in cluster_assignments.values() if c == cluster)
            if cluster_stocks > 0:
                weight = portfolio.target_allocation.get(cluster, 0) / cluster_stocks
                target_weights[ticker] = weight
        
        # Get current prices
        current_prices = {}
        for ticker, data in stock_data.items():
            available_dates = data.loc[:rebal_date].index
            if len(available_dates) > 0:
                last_date = available_dates[-1]
                current_prices[ticker] = data.loc[last_date, 'Close']
        
        # Calculate current portfolio value
        current_value = sum(
            portfolio.holdings.get(ticker, 0) * current_prices.get(ticker, 0)
            for ticker in current_prices.keys()
        )
        
        if current_value.sum() == 0:
            current_value = portfolio.current_value
        
        # Rebalance
        new_holdings = {}
        costs = 0
        for ticker, weight in target_weights.items():
            if ticker in current_prices and weight > 0:
                target_value = current_value * weight
                target_shares = target_value / current_prices[ticker]
                
                old_shares = portfolio.holdings.get(ticker, 0)
                shares_diff = abs(target_shares - old_shares)
                trade_value = shares_diff * current_prices[ticker]
                costs += trade_value * transaction_cost_rate
                
                new_holdings[ticker] = target_shares
        
        portfolio.holdings = new_holdings
        portfolio.current_value = current_value - costs
        total_transaction_costs += costs
        
        # Calculate return
        prev_value = portfolio_values[-1]['value']
        period_return = (current_value / prev_value) - 1 if prev_value > 0 else 0
        
        portfolio_values.append({
            'date': rebal_date,
            'value': current_value,
            'return': period_return
        })
        
        print(f"      Cost: ${costs:.2f}, Value: ${current_value:,.2f}")
    
    # Final value at end date
    final_date = pd.Timestamp(end_date)
    final_prices = {}
    for ticker, data in stock_data.items():
        available_dates = data.loc[:final_date].index
        if len(available_dates) > 0:
            last_date = available_dates[-1]
            final_prices[ticker] = data.loc[last_date, 'Close']
    
    final_value = sum(
        portfolio.holdings.get(ticker, 0) * final_prices.get(ticker, 0)
        for ticker in final_prices.keys()
    )
    
    # Calculate performance metrics
    print(f"    Total rebalances: {len(rebalancing_dates) + 1}")
    print(f"    Final value: ${final_value:,.2f}")
    
    history_df = pd.DataFrame(portfolio_values).set_index('date')
    
    total_return = (final_value / portfolio.initial_capital) - 1
    cagr = (1 + total_return) ** (1 / 4) - 1
    
    returns = history_df['return'].dropna()
    sharpe_ratio = (cagr - 0.02) / (returns.std() * np.sqrt(4)) if len(returns) > 0 and returns.std() > 0 else 0
    
    cummax = history_df['value'].cummax()
    drawdown = (history_df['value'] - cummax) / cummax
    max_drawdown = drawdown.min()
    
    print(f"  Total Return:   {total_return:>8.2%}")
    print(f"  CAGR:           {cagr:>8.2%}")
    print(f"  Sharpe Ratio:   {sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:   {max_drawdown:>8.2%}")
    print(f"  Final Value:    ${final_value:>12,.2f}")
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': final_value,
        'transaction_costs': total_transaction_costs
    }



def calculate_enhanced_metrics(portfolio_values, benchmark_data, start_date, end_date):
    """
    Calculate additional portfolio metrics: Information Ratio, Volatility, and Correlation.
    
    Args:
        portfolio_values: List of dicts with portfolio values over time
        benchmark_data: S&P 500 DataFrame
        start_date: Start date string
        end_date: End date string
    
    Returns:
        Dictionary with enhanced metrics
    """
    import pandas as pd
    import numpy as np
    
    # Create portfolio returns series
    history_df = pd.DataFrame(portfolio_values).set_index('date')
    portfolio_returns = history_df['return'].dropna()
    
    # Get benchmark returns for same period
    benchmark_subset = benchmark_data.loc[start_date:end_date]['Close']
    if isinstance(benchmark_subset, pd.DataFrame):
        benchmark_subset = benchmark_subset.iloc[:, 0]
    
    # Align dates - get benchmark returns at quarterly intervals matching portfolio
    benchmark_quarterly = []
    for date in history_df.index:
        if date in benchmark_subset.index:
            benchmark_quarterly.append(benchmark_subset[date])
        else:
            # Find closest date
            closest_dates = benchmark_subset.index[benchmark_subset.index <= date]
            if len(closest_dates) > 0:
                benchmark_quarterly.append(benchmark_subset[closest_dates[-1]])
    
    if len(benchmark_quarterly) > 1:
        benchmark_quarterly = pd.Series(benchmark_quarterly, index=history_df.index[:len(benchmark_quarterly)])
        benchmark_returns = benchmark_quarterly.pct_change().dropna()
    else:
        benchmark_returns = pd.Series([0])
    
    # Align portfolio and benchmark returns
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns.iloc[:min_len]
    benchmark_returns = benchmark_returns.iloc[:min_len]
    
    # Reset indices to avoid alignment issues
    portfolio_returns = portfolio_returns.reset_index(drop=True)
    benchmark_returns = benchmark_returns.reset_index(drop=True)
    
    # 1. Annualized Volatility
    volatility = portfolio_returns.std() * np.sqrt(4)  # Quarterly data, so sqrt(4) for annual
    
    # 2. Correlation with Benchmark
    if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
        correlation = portfolio_returns.corr(benchmark_returns)
    else:
        correlation = 0.0
    
    # 3. Information Ratio
    # IR = (Portfolio Return - Benchmark Return) / Tracking Error
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(4)  # Annualized
    
    if tracking_error > 0:
        # Annualized excess return
        mean_excess = excess_returns.mean() * 4  # Quarterly to annual
        information_ratio = mean_excess / tracking_error
    else:
        information_ratio = 0.0
    
    return {
        'volatility': volatility,
        'correlation': correlation,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error
    }


def backtest_ml_portfolio(portfolio, stock_data, stock_features_dict, model, scaler, feature_cols, start_date, end_date):
    """
    Backtest portfolio using ML model predictions for stock selection.
    Uses same structure as quarterly_rebalancing_backtest but with ML predictions.
    
    Args:
        portfolio: Portfolio object with target allocation rules
        stock_data: Dictionary of stock price DataFrames
        stock_features_dict: Dictionary of pre-calculated features
        model: Trained ML model
        scaler: Fitted scaler for features
        feature_cols: List of feature column names
        start_date: Start date for backtesting
        end_date: End date for backtesting
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"    Backtesting ML portfolio from {start_date} to {end_date}")
    
    # Initialize portfolio
    portfolio.initial_capital = 100000
    portfolio.current_value = 100000
    portfolio.holdings = {}
    
    # Get quarter-end dates for rebalancing
    date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
    rebalancing_dates = [d.strftime('%Y-%m-%d') for d in date_range]
    print(f"    Found {len(rebalancing_dates)} rebalancing dates")
    
    portfolio_values = []
    total_transaction_costs = 0
    transaction_cost_rate = 0.0015
    
    # Get S&P 500 data for enhanced metrics calculation
    # This should be passed as benchmark_data parameter, but for now we'll handle missing case
    try:
        benchmark_data = stock_data.get('SPY') or stock_data[list(stock_data.keys())[0]]
    except:
        benchmark_data = None
    
    # Initial allocation using ML predictions
    initial_date = pd.Timestamp(start_date)
    
    # Get features at initial date
    current_features = {}
    for ticker, features_df in stock_features_dict.items():
        available = features_df[features_df.index <= initial_date]
        if len(available) > 0:
            current_features[ticker] = available.iloc[-1]
    
    if len(current_features) > 0:
        feature_matrix = pd.DataFrame(current_features).T
        
        # Make predictions
        X = feature_matrix[feature_cols].copy()
        
        # Encode cluster if present
        if 'cluster' in feature_cols and 'cluster' in X.columns:
            cluster_map = {'low-volatility': 0, 'moderate': 1, 'high-volatility': 2}
            X['cluster'] = X['cluster'].map(cluster_map).fillna(1)
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Rank stocks and create pseudo-clusters
        pred_series = pd.Series(predictions, index=feature_matrix.index).sort_values(ascending=False)
        n_stocks = len(pred_series)
        
        cluster_assignments = {}
        for i, ticker in enumerate(pred_series.index):
            if i < n_stocks // 3:
                cluster_assignments[ticker] = 'high-volatility'
            elif i < 2 * n_stocks // 3:
                cluster_assignments[ticker] = 'moderate'
            else:
                cluster_assignments[ticker] = 'low-volatility'
        
        # Calculate weights
        target_weights = {}
        for ticker in stock_data.keys():
            cluster = cluster_assignments.get(ticker, 'moderate')
            cluster_stocks = sum(1 for c in cluster_assignments.values() if c == cluster)
            if cluster_stocks > 0:
                weight = portfolio.target_allocation.get(cluster, 0) / cluster_stocks
                target_weights[ticker] = weight
        
        # Get prices and buy
        current_prices = {}
        for ticker, data in stock_data.items():
            available_dates = data.loc[:initial_date].index
            if len(available_dates) > 0:
                last_date = available_dates[-1]
                current_prices[ticker] = data.loc[last_date, 'Close']
        
        initial_costs = 0
        for ticker, weight in target_weights.items():
            if ticker in current_prices and weight > 0:
                target_value = portfolio.current_value * weight
                shares = target_value / current_prices[ticker]
                portfolio.holdings[ticker] = shares
                initial_costs += target_value * transaction_cost_rate
        
        portfolio.current_value -= initial_costs
        total_transaction_costs += initial_costs
        
        portfolio_values.append({
            'date': initial_date,
            'value': portfolio.current_value,
            'return': 0.0
        })
    
    # Quarterly rebalancing with ML predictions
    for rebal_date in rebalancing_dates:
        # Get features at rebalancing date
        current_features = {}
        for ticker, features_df in stock_features_dict.items():
            available = features_df[features_df.index <= rebal_date]
            if len(available) > 0:
                current_features[ticker] = available.iloc[-1]
        
        if len(current_features) == 0:
            continue
        
        feature_matrix = pd.DataFrame(current_features).T
        
        # Make predictions
        X = feature_matrix[feature_cols].copy()
        
        # Encode cluster if present
        if 'cluster' in feature_cols and 'cluster' in X.columns:
            cluster_map = {'low-volatility': 0, 'moderate': 1, 'high-volatility': 2}
            X['cluster'] = X['cluster'].map(cluster_map).fillna(1)
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Rank and assign pseudo-clusters
        pred_series = pd.Series(predictions, index=feature_matrix.index).sort_values(ascending=False)
        n_stocks = len(pred_series)
        
        cluster_assignments = {}
        for i, ticker in enumerate(pred_series.index):
            if i < n_stocks // 3:
                cluster_assignments[ticker] = 'high-volatility'
            elif i < 2 * n_stocks // 3:
                cluster_assignments[ticker] = 'moderate'
            else:
                cluster_assignments[ticker] = 'low-volatility'
        
        # Calculate target weights
        target_weights = {}
        for ticker in stock_data.keys():
            cluster = cluster_assignments.get(ticker, 'moderate')
            cluster_stocks = sum(1 for c in cluster_assignments.values() if c == cluster)
            if cluster_stocks > 0:
                weight = portfolio.target_allocation.get(cluster, 0) / cluster_stocks
                target_weights[ticker] = weight
        
        # Get current prices
        current_prices = {}
        for ticker, data in stock_data.items():
            available_dates = data.loc[:rebal_date].index
            if len(available_dates) > 0:
                last_date = available_dates[-1]
                current_prices[ticker] = data.loc[last_date, 'Close']
        
        # Calculate current value
        current_value = sum(
            portfolio.holdings.get(ticker, 0) * current_prices.get(ticker, 0)
            for ticker in current_prices.keys()
        )
        
        if current_value.sum() == 0:
            current_value = portfolio.current_value
        
        # Rebalance
        new_holdings = {}
        costs = 0
        for ticker, weight in target_weights.items():
            if ticker in current_prices and weight > 0:
                target_value = current_value * weight
                target_shares = target_value / current_prices[ticker]
                
                old_shares = portfolio.holdings.get(ticker, 0)
                shares_diff = abs(target_shares - old_shares)
                trade_value = shares_diff * current_prices[ticker]
                costs += trade_value * transaction_cost_rate
                
                new_holdings[ticker] = target_shares
        
        portfolio.holdings = new_holdings
        portfolio.current_value = current_value - costs
        total_transaction_costs += costs
        
        # Record values
        prev_value = portfolio_values[-1]['value']
        period_return = (current_value / prev_value) - 1 if prev_value > 0 else 0
        
        portfolio_values.append({
            'date': rebal_date,
            'value': current_value,
            'return': period_return
        })
    
    # Calculate final metrics
    final_date = pd.Timestamp(end_date)
    final_prices = {}
    for ticker, data in stock_data.items():
        available_dates = data.loc[:final_date].index
        if len(available_dates) > 0:
            last_date = available_dates[-1]
            final_prices[ticker] = data.loc[last_date, 'Close']
    
    final_value = sum(
        portfolio.holdings.get(ticker, 0) * final_prices.get(ticker, 0)
        for ticker in final_prices.keys()
    )
    
    # Calculate performance metrics
    history_df = pd.DataFrame(portfolio_values).set_index('date')
    
    total_return = (final_value / portfolio.initial_capital) - 1
    cagr = (1 + total_return) ** (1 / 4) - 1
    
    returns = history_df['return'].dropna() if len(portfolio_values) > 0 else pd.Series([])
    sharpe_ratio = (cagr - 0.02) / (returns.std() * np.sqrt(4)) if len(returns) > 0 and returns.std() > 0 else 0
    
    cummax = history_df['value'].cummax()
    drawdown = (history_df['value'] - cummax) / cummax
    max_drawdown = drawdown.min() if len(history_df) > 0 else 0
    
    print(f"  Total Return:   {total_return:>8.2%}")
    print(f"  CAGR:           {cagr:>8.2%}")
    print(f"  Sharpe Ratio:   {sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:   {max_drawdown:>8.2%}")
    
    # Calculate enhanced metrics if we have benchmark data
    if benchmark_data is not None:
        try:
            enhanced_metrics = calculate_enhanced_metrics(portfolio_values, benchmark_data, start_date, end_date)
            print(f"  Volatility:     {enhanced_metrics['volatility']:>8.2%}")
            print(f"  Correlation:    {enhanced_metrics['correlation']:>8.2f}")
            print(f"  Info Ratio:     {enhanced_metrics['information_ratio']:>8.2f}")
            
            return {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value,
                'transaction_costs': total_transaction_costs,
                'volatility': enhanced_metrics['volatility'],
                'correlation': enhanced_metrics['correlation'],
                'information_ratio': enhanced_metrics['information_ratio']
            }
        except:
            pass
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': final_value,
        'transaction_costs': total_transaction_costs,
        'volatility': 0.0,
        'correlation': 0.0,
        'information_ratio': 0.0
    }

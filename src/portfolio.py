"""
Portfolio construction module.
"""

import pandas as pd
import numpy as np


class Portfolio:
    """Portfolio class to manage stock allocations and rebalancing."""
    
    def __init__(self, name, target_allocation, initial_capital=100000):
        self.name = name
        self.target_allocation = target_allocation
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.holdings = {}
        self.weights = {}
        self.history = []
        
    def calculate_weights(self, cluster_assignments):
        """Calculate stock weights based on target cluster allocations."""
        clusters = {}
        for ticker, cluster in cluster_assignments.items():
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(ticker)
        
        weights = {}
        for cluster, tickers in clusters.items():
            if cluster in self.target_allocation:
                cluster_weight = self.target_allocation[cluster]
                weight_per_stock = cluster_weight / len(tickers)
                for ticker in tickers:
                    weights[ticker] = weight_per_stock
        
        return weights
    
    def rebalance(self, prices, cluster_assignments, transaction_cost_pct=0.0015):
        """Rebalance portfolio to target allocations."""
        # Use current_value as the base (includes both holdings value and any cash)
        portfolio_value = self.current_value
        
        # Calculate target weights
        target_weights = self.calculate_weights(cluster_assignments)
        
        # Calculate current holdings value
        current_values = {ticker: shares * prices.get(ticker, 0) 
                         for ticker, shares in self.holdings.items()}
        
        # Calculate total value to be traded (for transaction costs)
        total_traded = 0
        
        # Calculate what we need to buy/sell
        for ticker in target_weights.keys():
            target_value = target_weights[ticker] * portfolio_value
            current_value = current_values.get(ticker, 0)
            trade_value = abs(target_value - current_value)
            total_traded += trade_value
        
        # Add value of stocks we're selling completely
        for ticker in list(self.holdings.keys()):
            if ticker not in target_weights:
                current_value = self.holdings[ticker] * prices.get(ticker, 0)
                total_traded += current_value
        
        # Calculate transaction costs
        transaction_costs = total_traded * transaction_cost_pct
        
        # Available capital after costs
        available_capital = portfolio_value - transaction_costs
        
        # Calculate target dollar amounts for each stock
        target_values = {ticker: weight * available_capital 
                        for ticker, weight in target_weights.items()}
        
        # Buy stocks
        new_holdings = {}
        for ticker, target_value in target_values.items():
            price = prices.get(ticker, 0)
            if price > 0:
                target_shares = target_value / price
                new_holdings[ticker] = target_shares
        
        # Update portfolio
        self.holdings = new_holdings
        self.weights = target_weights
        self.current_value = available_capital
        
        return transaction_costs
    
    def update_value(self, prices, date=None):
        """Update portfolio value based on current prices."""
        total_value = sum(shares * prices.get(ticker, 0) 
                         for ticker, shares in self.holdings.items())
        
        self.current_value = total_value
        
        if date is not None:
            self.history.append({
                'date': date,
                'value': total_value,
                'return': (total_value / self.initial_capital) - 1
            })
    
    def get_history_df(self):
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.history)


def create_portfolios(initial_capital=100000):
    """Create the three portfolio types."""
    portfolios = {
        'Conservative': Portfolio('Conservative', {'low-volatility': 0.6, 'moderate': 0.3, 'high-volatility': 0.1}, initial_capital),
        'Balanced': Portfolio('Balanced', {'low-volatility': 0.4, 'moderate': 0.4, 'high-volatility': 0.2}, initial_capital),
        'Aggressive': Portfolio('Aggressive', {'low-volatility': 0.2, 'moderate': 0.3, 'high-volatility': 0.5}, initial_capital)
    }
    return portfolios


if __name__ == "__main__":
    print("Portfolio module test - run portfolio construction test instead")

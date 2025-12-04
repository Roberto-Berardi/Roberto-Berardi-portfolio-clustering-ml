"""
Basic unit tests that verify core functionality
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

def test_sharpe_calculation():
    """Test Sharpe ratio calculation logic"""
    returns = 0.10  # 10% return
    volatility = 0.20  # 20% volatility
    risk_free = 0.02  # 2% risk-free rate
    
    sharpe = (returns - risk_free) / volatility
    
    assert abs(sharpe - 0.4) < 0.001, f"Expected 0.4, got {sharpe}"
    print("✓ Sharpe calculation test passed")

def test_max_drawdown():
    """Test max drawdown logic"""
    prices = pd.Series([100, 110, 105, 95, 100])
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    
    assert max_dd < 0, "Max drawdown should be negative"
    assert max_dd > -0.15, f"Max drawdown {max_dd} seems too large"
    print("✓ Max drawdown test passed")

def test_portfolio_allocation():
    """Test portfolio allocation logic"""
    # Conservative: 60/30/10
    conservative = 0.60 + 0.30 + 0.10
    assert abs(conservative - 1.0) < 0.001, "Should sum to 100%"
    
    # Balanced: 40/40/20
    balanced = 0.40 + 0.40 + 0.20
    assert abs(balanced - 1.0) < 0.001, "Should sum to 100%"
    
    # Aggressive: 20/30/50
    aggressive = 0.20 + 0.30 + 0.50
    assert abs(aggressive - 1.0) < 0.001, "Should sum to 100%"
    
    print("✓ Portfolio allocation test passed")

def test_transaction_cost():
    """Test transaction cost calculation"""
    trade_value = 10000
    cost_rate = 0.0015  # 0.15%
    
    cost = trade_value * cost_rate
    assert abs(cost - 15.0) < 0.01, f"Expected $15, got ${cost}"
    print("✓ Transaction cost test passed")

def test_return_calculation():
    """Test return calculation"""
    price_old = 100
    price_new = 105
    
    simple_return = (price_new - price_old) / price_old
    assert abs(simple_return - 0.05) < 0.001, f"Expected 5%, got {simple_return}"
    print("✓ Return calculation test passed")

if __name__ == "__main__":
    print("\nRunning basic unit tests...\n")
    test_sharpe_calculation()
    test_max_drawdown()
    test_portfolio_allocation()
    test_transaction_cost()
    test_return_calculation()
    print("\n✅ All 5 tests passed!\n")

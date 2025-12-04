# Dynamic Portfolio Clustering and Risk Profiling with Machine Learning

**Student:** Roberto Berardi  
**Student Number:** 25419094  
**Program:** MSc Finance, HEC Lausanne - UNIL  
**Course:** Advanced Programming - Fall 2025  
**Date:** November 3, 2025

## Research Question

Can risk-based clustering strategies outperform machine learning predictions for portfolio construction? This project compares simple clustering methods (K-means, GMM) against complex ML models (Ridge, Random Forest, XGBoost, Neural Network) for building investment portfolios.

## Data and Features

I will analyze **50 U.S. large-cap stocks** using daily price data from January 2015 to December 2024. For each stock, I will calculate the following risk-return metrics on a **rolling 12-month basis**:

- Annualized return and volatility
- Sharpe ratio (2% risk-free rate)
- Maximum drawdown
- Beta and correlation (vs. S&P 500)
- Momentum (1m, 3m, 6m, 12m)

## Methodology

### Unsupervised Learning (Clustering)
1. Standardize all features
2. Apply PCA for dimensionality reduction
3. Apply K-means and GMM clustering
4. Label clusters: "low-volatility", "moderate", "high-volatility"
5. Compare quality using silhouette scores

### Portfolio Construction
Based on cluster assignments, construct three portfolios with equal weighting within clusters:

- **Conservative:** 60% low-vol / 30% moderate / 10% high-vol
- **Balanced:** 40% low-vol / 40% moderate / 20% high-vol
- **Aggressive:** 20% low-vol / 30% moderate / 50% high-vol

**Rebalancing:** Quarterly (2021-2024) with 0.15% transaction costs, re-running clustering at each quarter using the most recent 12 months of data.

### Supervised Learning (ML Prediction)
Train four regression models to predict 3-month forward returns:
1. Ridge Regression
2. Random Forest
3. XGBoost
4. Neural Network

Each model trained in two versions:
- **Base:** 10 numeric features only
- **Enhanced:** Numeric features + cluster assignment

**Temporal validation:**
- Training: 2015-2020 (6 years)
- Testing: 2021-2024 (4 years)

## Evaluation Metrics

**ML Models:**
- R² (coefficient of determination)
- MSE (Mean Squared Error)
- Directional accuracy

**Portfolio Performance (vs. S&P 500):**
- Total return and CAGR
- Sharpe ratio
- Maximum drawdown
- Information ratio
- Volatility and correlation with benchmark
- Net returns after transaction costs

## Expected Outcomes

This project will determine:
1. Whether enhanced ML models (with cluster features) outperform base models
2. Whether clustering-based portfolios deliver superior risk-adjusted returns compared to ML-driven portfolios and the S&P 500 benchmark

The results will provide insights into when simpler, interpretable methods (clustering) may be more robust than complex predictive models (ML) for portfolio construction.

## Technical Implementation

- **Language:** Python 3.10+
- **Key Libraries:** pandas, numpy, scikit-learn, xgboost, yfinance
- **Initial Capital:** $100,000
- **Reproducibility:** Fixed random seeds throughout

---

**Word Count:** 490 words

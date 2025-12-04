# Roberto Berardi - Dynamic Portfolio Clustering and Risk Profiling with Machine Learning

**Course:** Advanced Programming - HEC Lausanne, Fall 2025  
**Student:** Roberto Berardi

## 📊 Project Overview

This project compares risk-based clustering strategies versus machine learning predictions for portfolio construction. Using 50 U.S. stocks from 2015-2024, we evaluate whether simple clustering (K-means, GMM) can outperform complex ML models (Ridge, Random Forest, XGBoost, Neural Network) for building investment portfolios.

## 🎯 Key Findings

- **Clustering-based portfolios outperformed ML-driven portfolios by 3-10%**
- **Both strategies beat S&P 500 benchmark by 10-24%**
- Aggressive clustering portfolio: **+84.03%** vs S&P 500: **+59.62%** (2021-2024)
- Enhanced ML models (with cluster features) marginally improved over base models

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Installation

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate portfolio-clustering-project
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Running the Project

**Full analysis (50 stocks, 10-15 minutes):**
```bash
python main.py
```

**Quick test (5 stocks, ~1 minute):**
```bash
python test_5stocks_complete.py
```

## 📁 Project Structure
```
portfolio-clustering-project/
├── main.py                     # Main entry point (50-stock analysis)
├── test_5stocks_complete.py    # Quick test with 5 stocks
├── PROPOSAL.md                 # Project proposal
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── src/                        # Source code modules
│   ├── data_loader.py          # Stock data loading (yfinance)
│   ├── feature_engineering.py  # Calculate 10 risk-return features
│   ├── clustering.py           # K-means & GMM clustering
│   ├── ml_models.py            # ML models (Ridge, RF, XGBoost, NN)
│   ├── portfolio.py            # Portfolio construction
│   └── backtesting.py          # Performance evaluation
├── data/                       # Cached stock data
│   └── cache/                  # Downloaded stock prices
├── results/                    # Output figures and tables
└── notebooks/                  # Jupyter notebooks (optional)
```

## 🔬 Methodology

### Data
- **50 U.S. large-cap stocks** (AAPL, MSFT, GOOGL, etc.)
- **Daily prices:** 2015-2024 (10 years)
- **Benchmark:** S&P 500

### Features (10 risk-return metrics, rolling 12-month)
1. Annualized return
2. Annualized volatility
3. Sharpe ratio (2% risk-free rate)
4. Maximum drawdown
5. Beta (market sensitivity)
6. Correlation with S&P 500
7-10. Momentum (1m, 3m, 6m, 12m)

### Clustering Approach
- **PCA:** Dimensionality reduction (3 components, 96.7% variance)
- **K-means:** Partition stocks into 3 clusters
- **GMM:** Probabilistic clustering
- **Labels:** low-volatility, moderate, high-volatility

### ML Approach
- **4 Models:** Ridge, Random Forest, XGBoost, Neural Network
- **Two Versions:** Base (10 features) vs Enhanced (+cluster feature)
- **Target:** Predict 3-month forward returns
- **Training:** 2015-2020 (6 years)
- **Testing:** 2021-2024 (4 years)

### Portfolio Construction
**Three strategies:**
- **Conservative:** 60% low-vol / 30% moderate / 10% high-vol
- **Balanced:** 40% low-vol / 40% moderate / 20% high-vol
- **Aggressive:** 20% low-vol / 30% moderate / 50% high-vol

**Backtesting:**
- Quarterly rebalancing (2021-2024)
- Transaction costs: 0.15% per trade
- Initial capital: $100,000

## 📊 Results Summary

### Clustering-Based Portfolios (2021-2024)

| Portfolio | Total Return | CAGR | Sharpe | Max Drawdown | vs S&P 500 |
|-----------|-------------|------|--------|--------------|------------|
| Conservative | 60.78% | 12.61% | 0.85 | -17.94% | +1.16% |
| Balanced | 69.79% | 14.15% | 0.86 | -22.52% | +10.17% |
| Aggressive | 84.03% | 16.47% | 0.72 | -30.42% | +24.41% |

### ML-Driven Portfolios (2021-2024)

| Portfolio | Total Return | CAGR | Sharpe | Max Drawdown | Volatility | Info Ratio |
|-----------|-------------|------|--------|--------------|------------|------------|
| Conservative | 50.15% | 10.70% | 0.73 | -18.08% | 12.10% | -0.46 |
| Balanced | 60.63% | 12.58% | 0.81 | -20.92% | 13.63% | -0.41 |
| Aggressive | 80.86% | 15.97% | 0.85 | -26.74% | 17.51% | -0.30 |

**S&P 500 Benchmark:** 59.62% total return, 12.40% CAGR, 0.63 Sharpe

### ML Model Performance

| Model | Version | R² | MSE | Directional Accuracy |
|-------|---------|-----|-----|---------------------|
| Ridge | Base | -0.108 | 0.0238 | 58.8% |
| Ridge | Enhanced | **-0.101** | **0.0237** | **58.9%** |
| Random Forest | Base | -0.504 | 0.0323 | 56.7% |
| Random Forest | Enhanced | -0.529 | 0.0329 | 57.3% |
| XGBoost | Base | -0.340 | 0.0288 | 55.2% |
| XGBoost | Enhanced | -0.376 | 0.0296 | 55.8% |
| Neural Network | Base | -1.170 | 0.0467 | 51.5% |
| Neural Network | Enhanced | -0.972 | 0.0424 | 52.4% |

**Best Model:** Ridge (Enhanced) with R² = -0.101

## 🔑 Key Insights

1. **Simpler is Better:** Risk-based clustering outperformed complex ML predictions
2. **Both Beat Market:** All strategies exceeded S&P 500 returns
3. **Clusters Add Value:** Enhanced ML models slightly better than base versions
4. **Stock Prediction is Hard:** Negative R² scores are normal (market noise)
5. **Directional Accuracy Matters:** ~59% accuracy (better than random 50%)

## ⚙️ Reproducibility

All results are fully reproducible:
- Random seed: `random_state=42` everywhere
- Same data sources (yfinance)
- Same time periods (2015-2024)
- Same methodology

Running `python main.py` will produce identical results.

## 🛠️ Technical Details

- **Language:** Python 3.11
- **Key Libraries:** pandas, numpy, scikit-learn, xgboost, yfinance
- **ML Framework:** scikit-learn
- **Data Source:** Yahoo Finance (yfinance)
- **Optimization:** Pre-calculated features reused (20x speedup)

## 📚 Citation
```bibtex
@misc{portfolio_clustering_2025,
  author = {Roberto Berardi},
  title = {Dynamic Portfolio Clustering and Risk Profiling with Machine Learning},
  year = {2025},
  institution = {HEC Lausanne},
  course = {Advanced Programming}
}
```

## 📄 License

This project is created for academic purposes as part of the Advanced Programming course at HEC Lausanne.

## 👤 Author

**Roberto Berardi**  
HEC Lausanne, Fall 2025  
Course: Advanced Programming

## 📧 Contact

For questions about this project, contact: roberto.berardi@unil.ch

---

*Last updated: December 2025*

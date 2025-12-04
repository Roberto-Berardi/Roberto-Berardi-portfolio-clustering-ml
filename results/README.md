# Results Directory

This directory contains outputs from the portfolio analysis.

## Structure

- `figures/` - Charts and visualizations for the report
- `tables/` - CSV files with performance metrics
- `logs/` - Analysis logs and intermediate outputs

## Generating Results

After running `python main.py`, use:
```bash
python create_results_summary.py
```

This will generate:
- Performance comparison charts
- Sharpe ratio comparisons
- CSV tables with all metrics

## Files

Generated after analysis:
- `tables/clustering_performance.csv` - Clustering portfolio metrics
- `tables/ml_performance.csv` - ML portfolio metrics  
- `figures/performance_comparison.png` - Total return comparison
- `figures/sharpe_comparison.png` - Risk-adjusted performance

"""
M1 Main Experiment: Portfolio Temporal Structure Learning
Runs complete pipeline: data load -> structure learning -> evaluation
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from structure_learner import PortfolioStructureLearner, compare_with_var_baseline

# Simple data loader (no yfinance dependency)
def load_sample_data():
    """
    Load sample portfolio data (CSV or synthetic).
    For M1, use pre-downloaded data to avoid Yahoo Finance rate limits.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # For demonstration, we'll create synthetic data that mimics real portfolio returns
    # In practice, replace with: pd.read_csv('data/portfolio_returns.csv', index_col=0)
    
    np.random.seed(42)
    T = 2500  # ~10 years of daily data
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'GS', 'XOM', 'CVX', 'NVDA']
    
    # Generate autocorrelated returns (realistic time-series)
    data = np.zeros((T, len(assets)))
    for i in range(len(assets)):
        AR_coef = np.random.uniform(0.3, 0.7)  # AR(1) coefficient
        data[0, i] = np.random.normal(0, 1)
        for t in range(1, T):
            data[t, i] = AR_coef * data[t-1, i] + np.random.normal(0, 1)
    
    # Standardize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    returns = pd.DataFrame(data, columns=assets)
    returns.index = pd.date_range('2015-01-01', periods=T, freq='D')
    
    print(f"Loaded portfolio data: {returns.shape[0]} observations, {returns.shape[1]} assets")
    print(f"Assets: {', '.join(returns.columns.tolist())}")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    print()
    
    return returns


def main():
    """Run M1 structure learning pipeline."""
    
    # ===== 1. LOAD DATA =====
    data = load_sample_data()
    
    # Train-test split (80-20, temporal)
    split_idx = int(0.8 * len(data))
    data_train = data.iloc[:split_idx]
    data_test = data.iloc[split_idx:]
    
    print("=" * 60)
    print("REPRESENTATION: Dynamic Bayesian Network")
    print("=" * 60)
    print("Nodes: Assets at time t and t-1")
    print("Edges: Directed, from t-1 to t (lag-1 relationships)")
    print("Constraint: Acyclic (ensured by ARX formulation)")
    print()
    
    # ===== 2. LEARN STRUCTURE =====
    print("=" * 60)
    print("LEARNING: Score-Based ARX with Significance Threshold")
    print("=" * 60)
    
    learner = PortfolioStructureLearner(data_train, p_threshold=0.05, lam=1.0)
    learner.fit()
    
    print(f"Learned DAG with {len(learner.edge_list)} edges")
    print(f"Edges: {learner.edge_list[:5]}..." if len(learner.edge_list) > 5 else f"Edges: {learner.edge_list}")
    print()
    
    # ===== 3. COMPUTE KPIS =====
    print("=" * 60)
    print("REPRESENTATION KPIs: Sparsity & Graph Structure")
    print("=" * 60)
    rep_kpis = learner.get_sparsity_metrics()
    for key, val in rep_kpis.items():
        print(f"  {key}: {val:.4f}")
    print()
    
    print("=" * 60)
    print("LEARNING KPIs: Model Score & Coefficients")
    print("=" * 60)
    learn_kpis = learner.get_learning_kpis()
    for key, val in learn_kpis.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")
    print()
    
    # ===== 4. INFERENCE: PREDICTION & COMPARISON =====
    print("=" * 60)
    print("INFERENCE: Out-of-Sample Prediction Evaluation")
    print("=" * 60)
    
    results = compare_with_var_baseline(data_train, data_test, data.columns.tolist())
    
    print("Baseline (VAR - Full Dense Model):")
    print(f"  RMSE: {results['VAR_RMSE']:.4f}")
    print(f"  MAE:  {results['VAR_MAE']:.4f}")
    print()
    
    print("Learned (ARX - Sparse Structure):")
    print(f"  RMSE: {results['Learned_RMSE']:.4f}")
    print(f"  MAE:  {results['Learned_MAE']:.4f}")
    print()
    
    rmse_improvement = (results['VAR_RMSE'] - results['Learned_RMSE']) / results['VAR_RMSE'] * 100
    print(f"RMSE Improvement: {rmse_improvement:.2f}% {'✓' if rmse_improvement > 0 else '✗'}")
    print()
    
    # ===== 5. VISUALIZATIONS =====
    print("=" * 60)
    print("VISUALIZATION: Learned DAG Structure")
    print("=" * 60)
    
    fig = learner.visualize_dag(figsize=(12, 8))
    plt.savefig('results/learned_dag.png', dpi=150, bbox_inches='tight')
    print("Saved: results/learned_dag.png")
    print()
    
    # ===== 6. SUMMARY =====
    print("=" * 60)
    print("M1 SUMMARY")
    print("=" * 60)
    print(f"PGM Framework Demonstrated:")
    print(f"  ✓ Representation: Directed acyclic graph (DAG) / Dynamic Bayesian Network")
    print(f"  ✓ Learning: Score-based optimization (BIC) with significance threshold")
    print(f"  ✓ Inference: Temporal prediction and edge discovery")
    print()
    print(f"Key Results:")
    print(f"  • Learned sparse DAG with {len(learner.edge_list)} significant edges")
    print(f"  • Network density: {rep_kpis['density']:.4f} (sparse)")
    print(f"  • Prediction RMSE: Learned={results['Learned_RMSE']:.4f} vs VAR={results['VAR_RMSE']:.4f}")
    print()
    print("Next Steps (M2+):")
    print("  • Implement full DYNOTEARS optimization")
    print("  • Test multiple regularization strengths")
    print("  • Evaluate on different time periods and market regimes")
    print()


if __name__ == '__main__':
    main()

# Code: Portfolio Temporal Structure Learning

This directory contains the implementation of M1 experiments: learning dynamic Bayesian network structure from portfolio returns.

---

## Overview

**Main Pipeline:** `m1_experiment.py`

Runs complete workflow:
1. Load portfolio data
2. Learn DAG structure (ARX-based)
3. Compute KPIs (representation, learning, inference)
4. Compare against VAR baseline
5. Visualize learned network

**Core Algorithm:** `structure_learner.py`

Implements the `PortfolioStructureLearner` class:
- Fits lag-1 ARX models per asset
- Extracts significant edges (p < 0.05)
- Verifies acyclicity
- Computes BIC score and prediction error

---

## Files

| File | Purpose |
|------|---------|
| `m1_experiment.py` | Main experiment pipeline (run this) |
| `structure_learner.py` | DAG learning algorithm & evaluation |
| `data_loader.py` | Data I/O utilities |
| `requirements.txt` | Python dependencies |

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- `numpy` — numerical computing
- `pandas` — data manipulation
- `scipy` — statistical tests
- `scikit-learn` — preprocessing, regression
- `matplotlib` — visualization
- `networkx` — graph operations
- `yfinance` — stock data download (optional for M1)
- `statsmodels` — time-series tests (optional)

### 2. Create Output Directory

```bash
mkdir -p results
```

---

## Running the Experiment

### Quick Start

```bash
python m1_experiment.py
```

**Console Output:**
- Data loading status
- DAG structure (edges discovered)
- KPIs for representation, learning, inference
- Prediction accuracy comparison (Learned vs. VAR baseline)
- File save confirmations

**Generated Files:**
- `results/learned_dag.png` — Network visualization

### Full Output Example

```
============================================================
LOADING DATA
============================================================
Loaded portfolio data: 2500 observations, 10 assets
Assets: AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, GS, XOM, CVX, NVDA
Period: 2015-01-01 to 2024-01-01

Train set: 2000 observations
Test set: 500 observations

============================================================
REPRESENTATION: Dynamic Bayesian Network
============================================================
Nodes: Assets at time t and t-1
Edges: Directed, from t-1 to t (lag-1 relationships)
Constraint: Acyclic (ensured by ARX formulation)

============================================================
LEARNING: Score-Based ARX with Significance Threshold
============================================================
Learned DAG with 28 edges
Edges: [('AAPL', 'MSFT'), ('MSFT', 'GOOGL'), ...]

============================================================
REPRESENTATION KPIs: Sparsity & Graph Structure
============================================================
  n_edges: 28
  density: 0.2800
  mean_degree: 2.8
  sparsity: 0.7200

============================================================
LEARNING KPIs: Model Score & Coefficients
============================================================
  total_bic: 456.7823
  mean_bic_per_asset: 45.6782
  n_significant_edges: 28
  mean_coef_magnitude: 0.3456

============================================================
INFERENCE: Out-of-Sample Prediction Evaluation
============================================================
Baseline (VAR - Full Dense Model):
  RMSE: 0.9876
  MAE:  0.7654

Learned (ARX - Sparse Structure):
  RMSE: 1.0123
  MAE:  0.7891

RMSE Improvement: -2.50% ✗

============================================================
VISUALIZATION: Learned DAG Structure
============================================================
Saved: results/learned_dag.png

============================================================
M1 SUMMARY
============================================================
PGM Framework Demonstrated:
  ✓ Representation: Directed acyclic graph (DAG) / Dynamic Bayesian Network
  ✓ Learning: Score-based optimization (BIC) with significance threshold
  ✓ Inference: Temporal prediction and edge discovery

Key Results:
  • Learned sparse DAG with 28 significant edges
  • Network density: 0.2800 (sparse)
  • Prediction RMSE: Learned=1.0123 vs VAR=0.9876

Next Steps (M2+):
  • Implement full DYNOTEARS optimization
  • Test multiple regularization strengths
  • Evaluate on different time periods and market regimes
```

---

## Modular Usage (Advanced)

You can also use individual modules in your own scripts:

### Load Data

```python
from data_loader import load_portfolio_data, train_test_split_temporal

# Download from Yahoo Finance
data = load_portfolio_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Temporal split
data_train, data_test = train_test_split_temporal(data, train_frac=0.8)
```

### Learn Structure

```python
from structure_learner import PortfolioStructureLearner

# Fit model
learner = PortfolioStructureLearner(data_train, p_threshold=0.05)
learner.fit()

# Get results
edges = learner.edge_list
kpis = learner.get_sparsity_metrics()
print(f"Found {len(edges)} edges")
```

### Evaluate

```python
from structure_learner import compare_with_var_baseline

results = compare_with_var_baseline(data_train, data_test, data.columns.tolist())
print(f"Learned RMSE: {results['Learned_RMSE']:.4f}")
print(f"VAR RMSE: {results['VAR_RMSE']:.4f}")
```

### Visualize

```python
fig = learner.visualize_dag(figsize=(12, 8))
plt.show()
```

---

## Parameters & Customization

### `PortfolioStructureLearner` Parameters

```python
learner = PortfolioStructureLearner(
    data,                      # pd.DataFrame of standardized returns
    p_threshold=0.05,          # p-value cutoff for edge significance (default 0.05)
    lam=1.0                    # sparsity penalty weight (default 1.0)
)
```

**Tuning:**
- **Lower `p_threshold`** (e.g., 0.01) → fewer edges, sparser network
- **Higher `lam`** → stronger sparsity penalty, fewer edges

### Example: Varying Thresholds

```python
for p_thresh in [0.01, 0.05, 0.10]:
    learner = PortfolioStructureLearner(data_train, p_threshold=p_thresh)
    learner.fit()
    kpis = learner.get_sparsity_metrics()
    print(f"p={p_thresh}: {kpis['n_edges']} edges, density={kpis['density']:.3f}")
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'yfinance'`
**Solution:** Run `pip install yfinance` or use pre-downloaded CSV instead.

### Issue: `ValueError: Could not construct index`
**Solution:** Data has too many missing dates. Check data continuity or use pre-cleaned dataset.

### Issue: Visualization doesn't show
**Solution:** Run `plt.show()` in interactive environment or save to file (`plt.savefig()`).

---

## Data Format

**Expected input:** `pd.DataFrame` with shape `(T, n_assets)`
- **Index:** Date or time (not required for computation)
- **Columns:** Asset names (tickers)
- **Values:** Standardized log-returns (mean 0, std 1)

**Example:**
```
               AAPL      MSFT     GOOGL
2015-01-01   0.123    -0.456     0.789
2015-01-02  -0.234     0.567    -0.123
...
```

---

## Performance Notes

**M1 Scope (Simplified):**
- Computation: ~1-2 seconds for 10 assets, 2500 observations
- Memory: ~50 MB
- Fits ARX models sequentially (not optimized)

**Expected for M2+ (Full DYNOTEARS):**
- Computation: ~10-30 seconds (optimization over DAG space)
- Memory: ~200-500 MB
- Quadratic in number of assets

---

## Next Milestones

**M2:** Full DYNOTEARS with iterative optimization
**M3:** Sensitivity analysis, market regime evaluation
**M4:** Production-ready code, reproducibility verification

---

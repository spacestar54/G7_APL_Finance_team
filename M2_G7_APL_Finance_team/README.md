# M2: Technical Foundation — DYNOTEARS on S&P 100 Stock Returns

**Course:** CSE 516 – Probabilistic Graphical Models  
**Project:** Major Project – Applied AI (Finance)  
**Group:** G7_APL_Finance_Team  

**Team Members:**
- Foram Gohel (AU2340054)
- Priyra Iyenger (AU2340263)

## 1. Project Overview

This milestone focuses on learning a Dynamic Bayesian Network (DBN) of S&P 100 stock returns using DYNOTEARS.

The objective is to investigate how daily stock returns influence one another through:

- **Intra-slice relationships (W):** Same-day dependencies between stocks, represented by a directed acyclic graph (DAG).
- **Inter-slice relationships (A):** Lagged dependencies, representing the influence of previous-day returns on current-day returns.

The DYNOTEARS optimization algorithm was implemented from scratch using an augmented Lagrangian approach and L-BFGS-B optimization.

## 2. Dataset

The project uses daily closing prices of S&P 100 companies obtained through Yahoo Finance.

**Dataset details:**

- Period: 1 July 2014 – 30 June 2019
- Original universe: 97 stock tickers
- Final dataset: 93 stocks
- Observations: 1,257 daily return rows
- Preprocessing: Log-return transformation, Augmented Dickey-Fuller (ADF) stationarity testing, and standardization.

Four companies were excluded because historical data was unavailable.

## 3. Methodology

The M2 implementation consists of the following components:

1. **Data Pipeline:** Collection, cleaning and preprocessing of stock price data.
2. **DYNOTEARS Implementation:** Learning same-day and lagged dependencies through constrained optimization.
3. **Simulation Experiments:** Testing graph recovery on synthetic datasets with known ground-truth structures.
4. **Baseline Comparison:** Comparing DYNOTEARS against NOTEARS + Lasso and evaluating forecasting performance.
5. **Visualization:** Generating learned network graphs, heatmaps and experimental plots.

## 4. Repository Structure

```text
M2_G7_APL_Finance_team/
│
├── README.md
│
├── Report/
│   └── M2_G7_APL_Finance_team_Report.pdf
│
├── Video/
│   └── M2_G7_APL_Finance_team_Video_Link.txt
│
├── Code/
│   ├── src/
│   │   ├── task1_data_pipeline.py
│   │   ├── task2_dynotears.py
│   │   ├── task3_simulation.py
│   │   ├── task5_6_baselines_rmse.py
│   │   ├── show_dyn.py
│   │   └── plot_basenet.py
│   │
│   ├── README.md
│   └── requirements.txt
│
├── Data/
│   ├── dataset.npz
│   ├── log_returns.csv
│   ├── prices_clean.csv
│   ├── prices_raw.csv
│   ├── returns_std.csv
│   └── README.md
│
└── Results/
    ├── basenet_graph_dynotears.png
    ├── basenet_heatmap_dynotears.png
    ├── fit_dynotears.npz
    ├── fit_static_notears.npz
    ├── task1_report.json
    ├── task2_real_fit.json
    ├── task3/
    ├── task6_forecast_rmse.csv
    └── task6_forecast_rmse.png
```

## 5. Experimental Results

The implementation was evaluated using simulated data and real S&P 100 stock returns.

**Key findings:**

- DYNOTEARS recovered the underlying graph structures more accurately than the independent NOTEARS + Lasso baseline in the simulation experiments.
- The learned same-day graph explained approximately 35% of the variance in held-out stock returns.
- The real-data model contained no meaningful lagged dependencies.
- None of the tested forecasting models consistently outperformed the zero-return predictor.

Detailed experimental results, statistical comparisons, visualizations and evaluation metrics are available in the `Results/` directory and the technical report.

## 6. Software and Dependencies

The implementation uses Python and the following libraries:

- NumPy
- SciPy
- Pandas
- Statsmodels
- Scikit-learn
- Matplotlib
- NetworkX
- yfinance

Install the dependencies listed in `Code/requirements.txt`, along with any additional libraries required by the source scripts.

```bash
pip install -r Code/requirements.txt
pip install scipy scikit-learn matplotlib networkx
```

## 7. Running the Experiments

Run commands from the M2 project directory.

### Data Pipeline

```bash
python Code/src/task1_data_pipeline.py
```

### DYNOTEARS Self-Test

```bash
python Code/src/task2_dynotears.py --selftest
```

### Simulation Experiments

```bash
python Code/src/task3_simulation.py
```

### Baseline Forecasting Experiments

```bash
python Code/src/task5_6_baselines_rmse.py
```

The scripts may require the corresponding input data and output paths to be configured according to their implementation.

## 8. Reproducibility

The source code, processed datasets, saved model outputs, experimental tables and plots are included in their respective directories.

The technical report documents the experimental setup, hyperparameters, evaluation methodology, results and limitations.

Refer to the individual README files inside `Code/` and `Data/` for additional implementation and dataset information.
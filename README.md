# M1: Portfolio Temporal Structure Learning with Dynamic Bayesian Networks

**Project:** CSE516 Probabilistic Graphical Models Major Project  
**Group:** G7 APL Finance team  
**Members:** Foram Gohel (AU2340054), Priya Iyengar (AU2340263)  
**Milestone:** M1 (Proposal & Scoping)  
**Submission Date:** September 15, 2026  

---

## Overview

We learn **directed acyclic graph (DAG) structures** from portfolio time-series data using dynamic Bayesian networks. The core question: **which assets directly predict which other assets tomorrow?**

**Base Paper:** Pamfil et al., "DYNOTEARS: Structure Learning from Time-Series Data," AISTATS 2020.

**Applied to:** Portfolio asset returns (S&P 500 constituents, 2015-2024)

---

## Problem Statement

Traditional portfolio models (VAR, GARCH) capture correlation but lack interpretability: you know prediction accuracy but not *which relationships matter*. 

We use **structure learning** to discover sparse, interpretable temporal causal structures. A directed edge from Apple→Microsoft means "yesterday's Apple returns predict today's Microsoft returns." This enables:
- Portfolio construction based on lead-lag relationships
- Risk management via contagion pathways
- Interpretable dependency networks

---

## PGM Framework: Representation, Learning, Inference

### **Representation: Dynamic Bayesian Network**

- **Nodes:** Assets at time t and t-1
- **Edges:** Directed, from t-1→t (lagged) or t→t (contemporaneous)
- **Acyclicity:** Guaranteed by formulation (edges point forward in time)
- **Distribution:** Factorizes according to DAG structure

### **Learning: Score-Based Structure Learning**

Minimize penalized score over acyclic DAGs:
```
min_G [ BIC(G) + λ|E| ]
```
where:
- **Score:** BIC (goodness-of-fit vs. model complexity)
- **Penalty:** Sparsity (number of edges)
- **Constraint:** Acyclicity via topological ordering

**M1 Approach:** Fit lag-1 ARX models, extract significant coefficients as edges, verify acyclicity.

### **Inference: Temporal Prediction & Edge Discovery**

Given learned DAG:
- Predict: $\hat{R}_j^t = \sum_{i \in \text{parents}(j)} \beta_{ij} R_i^{t-1}$
- Interpret: Directed edges reveal direct temporal influence
- Evaluate: Out-of-sample prediction error vs. baseline

---

## Baseline & KPIs

### **Baseline:** Vector Autoregressive (VAR) Model
- Fits full dense coefficient matrix (no structure learning)
- Comparison benchmark for prediction accuracy

### **KPIs:**

| Category | Metric | Meaning |
|----------|--------|---------|
| **Representation** | Sparsity, Density, Mean Degree | How sparse is learned network? |
| **Learning** | BIC, # Significant Edges | How good is the model score? |
| **Inference** | RMSE, MAE (out-of-sample) | How well do predictions work? |
| **Structure** | Edge list, acyclicity check | Interpretability & consistency |

---

## Data

**Primary Dataset:** S&P 500 daily returns
- **Assets:** 10 major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, GS, XOM, CVX, NVDA)
- **Period:** 2015-01-01 to 2024-01-01 (~2500 trading days)
- **Preprocessing:** Log-returns, standardized (mean 0, std 1)
- **Train-Test Split:** 80-20 temporal (train on 2015-2022, test on 2023-2024)

**Source:** Yahoo Finance (via yfinance Python package)

---

## Folder Structure

```
M1_G7_APL_Finance_team/
├── README.md                          # This file
├── Report/
│   └── M1_G7_Report.pdf              # Full technical proposal
├── Code/
│   ├── src/
│   │   ├── structure_learner.py      # Core DAG learning algorithm
│   │   ├── data_loader.py            # Data I/O utilities
│   │   └── m1_experiment.py          # Main experiment pipeline
│   └── README.md                      # Code documentation
├── Data/
│   └── README.md                      # Dataset info & download steps
├── Results/
│   ├── learned_dag.png               # Visualization of learned DAG
│   └── kpi_summary.txt               # Numerical results
└── Video/
    └── M1_video_link.txt             # Video walkthrough link/file
```

---

## How to Run

See `Code/README.md` for detailed instructions. Quick start:

```bash
cd Code
python src/m1_experiment.py
```

This runs the full pipeline:
1. Load portfolio data
2. Learn DAG structure via ARX models
3. Compute representation/learning/inference KPIs
4. Compare against VAR baseline
5. Visualize learned network
6. Print summary

**Output:** Console output + visualization saved to `Results/`

---

## Key M1 Contributions

✓ **Problem Definition:** Clear, domain-specific (portfolio returns, temporal dependencies)  
✓ **PGM Framework:** All three verticals demonstrated (representation, learning, inference)  
✓ **SOTA Positioning:** DYNOTEARS (Pamfil et al. 2020) identified and adapted  
✓ **Baseline & KPIs:** VAR baseline, 4 categories of metrics defined  
✓ **Reproducible Code:** Modular, commented, real data pipeline  
✓ **Simplicity:** Proof-of-concept (ARX) that avoids overfitting M1 scope  

---

## Team Responsibilities (M1)

- **Foram Gohel:** Data loading, preprocessing, VAR baseline, prediction evaluation
- **Priya Iyengar:** Structure learning (ARX models), DAG visualization, KPI computation
- **Both:** Code documentation, README, video walkthrough, interpretation

---

## Milestones Ahead

- **M2 (Oct 11):** Full DYNOTEARS optimization, regularization tuning
- **M3 (Nov 1):** Systematic comparison, sensitivity analysis, market regime analysis
- **M4 (Nov 22):** Final evaluation, reproducibility, limitations & future work

---

## References

[1] Pamfil, R. et al. (2020). DYNOTEARS: Structure Learning from Time-Series Data. *Proc. 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)*.

[2] Koller, D., Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

[3] Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

---

**GitHub:** https://github.com/spacestar54/G7_APL_Finance_team.git
=======
# G7_APL_Finance_team
>>>>>>> 94486683a36f5f0dc8309b9159ecf47e96498a0c

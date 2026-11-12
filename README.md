# Portfolio Temporal Structure Learning with Dynamic Bayesian Networks

**Project:** CSE516 – Probabilistic Graphical Models Major Project  
**Group:** G7 APL Finance Team  
**Members:** Foram Gohel (AU2340054), Priya Iyengar (AU2340263)  
**Milestones:** M1 (Proposal & Scoping) | M2 (Implementation & Evaluation) | M3 (Systematic Comparison) | M4 (Final Evaluation)

---

## Overview

We learn **directed acyclic graph (DAG) structures** from portfolio time-series data using dynamic Bayesian networks.

The core question is: **Which assets directly predict which other assets over time?**

**Base Paper:** Pamfil et al., "DYNOTEARS: Structure Learning from Time-Series Data," AISTATS 2020.

**Application:** Portfolio asset returns.

---

## Problem Statement

Traditional portfolio models such as VAR and GARCH capture statistical dependencies but provide limited insight into the underlying relationships between assets.

We use **structure learning** to discover sparse, interpretable temporal dependency structures.

A directed edge from Apple → Microsoft, for example, indicates that Apple's past returns help predict Microsoft's current returns.

This enables:
- Portfolio construction based on lead-lag relationships.
- Risk management through contagion pathways.
- Interpretable financial dependency networks.

---

## PGM Framework: Representation, Learning, and Inference

### Representation: Dynamic Bayesian Network

- **Nodes:** Assets at time *t* and *t-1*.
- **Edges:** Directed relationships between variables across time.
- **Acyclicity:** Enforced through the model formulation.
- **Distribution:** Factorizes according to the learned DAG structure.

### Learning: Score-Based Structure Learning

The objective is to learn a sparse, interpretable DAG while satisfying acyclicity constraints.

The original M1 approach used lag-1 ARX models to identify significant coefficients and construct the initial network. Subsequent milestones developed the implementation using DYNOTEARS.

**Base formulation:**

```text
min_G [ BIC(G) + λ|E| ]
```

Where:
- **Score:** BIC, balancing goodness-of-fit and model complexity.
- **Penalty:** Sparsity, controlled by the number of edges.
- **Constraint:** Acyclicity of the learned graph.

### Inference: Temporal Prediction and Edge Discovery

Given the learned DAG:

- **Prediction:** Use learned temporal relationships to estimate future asset returns.
- **Interpretation:** Directed edges reveal direct temporal dependencies.
- **Evaluation:** Compare out-of-sample predictive performance against baseline models.

---

## Baseline and KPIs

### Baseline

**Vector Autoregressive (VAR) Model**

- Fits a dense coefficient matrix.
- Does not explicitly perform sparse structure learning.
- Serves as a benchmark for predictive performance.

### Key Performance Indicators

| Category | Metric | Meaning |
|---|---|---|
| Representation | Sparsity, Density, Mean Degree | Measures network complexity |
| Learning | BIC, Number of Significant Edges | Evaluates model fit and learned structure |
| Inference | RMSE, MAE | Measures out-of-sample prediction accuracy |
| Structure | Edge List, Acyclicity Check | Evaluates interpretability and consistency |

---

## Data

**Primary Dataset:** Historical daily stock returns.

The initial M1 proposal used the following dataset configuration:

- **Assets:** 10 major stocks — AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, GS, XOM, CVX, NVDA.
- **Period:** January 2015 – January 2024.
- **Preprocessing:** Log returns and standardization.
- **Train-Test Split:** 80:20 temporal split.
- **Source:** Yahoo Finance, accessed through the `yfinance` Python package.

Refer to the individual milestone reports for the final experimental dataset and preprocessing configuration.

---

## Folder Structure

```text
G7_APL_Finance_team/
├── README.md
├── Report/
│   ├── M1_G7_Report.pdf
│   └── Final_Report.pdf
├── Code/
│   ├── src/
│   │   ├── structure_learner.py
│   │   ├── data_loader.py
│   │   └── m1_experiment.py
│   └── README.md
├── Data/
│   └── README.md
├── Results/
│   ├── learned_dag.png
│   └── kpi_summary.txt
└── Video/
    └── M4_video_link.txt
```

---

## How to Run

Refer to `Code/README.md` for detailed installation instructions and execution steps.

Quick start:

```bash
cd Code
python src/m1_experiment.py
```

The initial experiment pipeline:

1. Loads portfolio data.
2. Learns the DAG structure.
3. Computes representation, learning, and inference KPIs.
4. Compares results against the VAR baseline.
5. Visualizes the learned network.
6. Generates a summary of results.

**Output:** Console results and visualizations saved to the `Results/` directory.

---

## Key Project Contributions

- **Problem Definition:** Applied temporal structure learning to financial portfolio returns.
- **PGM Framework:** Demonstrated representation, learning, and inference.
- **SOTA Positioning:** Studied and implemented DYNOTEARS based on Pamfil et al. (2020).
- **Baseline and Evaluation:** Established benchmark models and quantitative evaluation metrics.
- **Reproducibility:** Developed modular code and an experimental pipeline.
- **Model Evaluation:** Investigated learned temporal dependencies, predictive performance, and network stability.

---

## Team Responsibilities

- **Foram Gohel:** Data loading, preprocessing, VAR baseline, and prediction evaluation.
- **Priya Iyengar:** Structure learning, DAG visualization, and KPI computation.
- **Both:** Code documentation, report preparation, video demonstration, and interpretation of results.

---

## Milestones

- **M1 – Proposal & Scoping:** Problem formulation, literature review, and initial methodology.
- **M2 – Implementation & Evaluation:** Initial implementation and experimental evaluation.
- **M3 – Systematic Comparison:** Model comparisons, sensitivity analysis, and market regime analysis.
- **M4 – Final Evaluation:** Final experiments, reproducibility, limitations, and future work.

---

## M2: Video Demonstration

**Project:** Portfolio Temporal Structure Learning with Dynamic Bayesian Networks

**Milestone:** M2 — DYNOTEARS Implementation and Evaluation

**Video Demonstration:** [Watch the M2 Project Video](https://drive.google.com/drive/folders/1M3tHWr31MaXgN-LWe8WreHrU115yPbEM)

---

## M4: Video Demonstration

**Project:** Portfolio Temporal Structure Learning with Dynamic Bayesian Networks

**Milestone:** M4 — Final Evaluation

**Video Demonstration:** [Watch the M4 Project Video](https://drive.google.com/drive/folders/1M3tHWr31MaXgN-LWe8WreHrU115yPbEM)

## References

[1] Pamfil, R. et al. (2020). *DYNOTEARS: Structure Learning from Time-Series Data*. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS).

[2] Koller, D., Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

[3] Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

---

**GitHub Repository:** https://github.com/spacestar54/G7_APL_Finance_team

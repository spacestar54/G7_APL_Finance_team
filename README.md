# Portfolio Temporal Structure Learning with Dynamic Bayesian Networks

**Course:** CSE516 – Probabilistic Graphical Models  
**Project:** Major Project – Applied AI (Finance)  
**Group:** G7 APL Finance Team  
**Members:**
- Foram Gohel (AU2340054)
- Priya Iyengar (AU2340263)

**Milestones:** M1 (Proposal & Scoping) | M2 (Technical Foundation) | M3 (Systematic Comparison) | M4 (Final Evaluation)

---

## 1. Project Overview

This project investigates the learning of directed graphical structures from financial time-series data using Dynamic Bayesian Networks (DBNs).

The central research question is:

**Which financial assets exhibit direct contemporaneous and temporal dependencies with other assets?**

We study the application of structure learning to stock returns, using the DYNOTEARS framework proposed by Pamfil et al. (2020).

The project evolved across four milestones, beginning with an initial proposal and simplified proof of concept, followed by a full DYNOTEARS implementation, systematic experimentation, and final evaluation.

**Base Paper:** Pamfil et al., "DYNOTEARS: Structure Learning from Time-Series Data," AISTATS 2020.

**Application Domain:** Financial time-series and stock market dependency analysis.

---

## 2. Project Evolution and Scope

The project underwent a refinement of its research scope after the initial proposal.

### M1 – Initial Proposal

The original M1 proposal focused on learning temporal dependencies among a small subset of major S&P 500 stocks.

The proposed experimental configuration included:
- 10–15 major stocks.
- Proposed period: 2015–2024.
- Log-return transformation and standardization.
- Initial lag-1 ARX-based structure learning.
- Comparison against a VAR baseline.

The initial implementation was intended as a simplified proof of concept to establish the research direction.

### M2–M4 – Refined Research Scope

From M2 onwards, the project was refined into a DYNOTEARS-based investigation using the S&P 100 universe.

The subsequent milestones focused on:
- Implementing DYNOTEARS from scratch.
- Learning contemporaneous and lagged dependency structures.
- Evaluating structure recovery on synthetic data.
- Applying the model to real-world financial returns.
- Comparing predictive and probabilistic performance against baseline models.
- Conducting sensitivity, ablation and robustness analyses.

The individual milestone folders contain the corresponding reports, code and experimental outputs.

---

## 3. Problem Statement

Financial time-series exhibit complex dependencies. Stock returns may be influenced by market-wide movements, industry relationships and temporal interactions between assets.

Traditional econometric models such as VAR and GARCH capture statistical dependencies but do not explicitly identify sparse directed graphical structures.

This project investigates structure learning to discover interpretable relationships between financial variables.

A directed edge from one asset to another represents a learned conditional dependency under the model.

Such structures can help investigate:
- Relationships between financial assets.
- Contemporaneous and lagged dependencies.
- Portfolio risk and interconnectedness.
- The interpretability and stability of learned financial networks.

The learned edges are statistical model relationships and should not automatically be interpreted as established causal effects.

---

## 4. Probabilistic Graphical Model Framework

### 4.1 Representation: Dynamic Bayesian Network

The project represents financial time-series using a Dynamic Bayesian Network.

The DYNOTEARS formulation separates two types of relationships:

- **W – Intra-slice structure:** Contemporaneous relationships between variables at the same time step. The corresponding graph must be acyclic.
- **A – Inter-slice structure:** Lagged relationships connecting previous observations to current observations.

The structural vector autoregressive formulation is:

```text
X = XW + YA + Z
```

Where:
- X represents the observed time-series.
- Y represents lagged observations.
- W represents contemporaneous relationships.
- A represents lagged relationships.
- Z represents residual noise.

### 4.2 Learning: DYNOTEARS

The project implements score-based structure learning with sparsity penalties and an acyclicity constraint.

The optimization objective is:

```text
minimize  (1/2n) ||X - XW - YA||²
          + λW ||W||₁
          + λA ||A||₁
```

Subject to the acyclicity constraint:

```text
h(W) = tr(exp(W ⊙ W)) - d = 0
```

The implementation uses an augmented Lagrangian optimization procedure and L-BFGS-B.

### 4.3 Inference and Evaluation

The learned structures are evaluated through:
- Predictive performance.
- Graph structure and sparsity.
- Synthetic ground-truth recovery.
- Forecasting and probabilistic scoring.
- Stability and robustness analysis.

---

## 5. Experimental Dataset

The final experimental work from M2 onwards uses historical daily stock returns from the S&P 100 universe.

The M1 dataset configuration was an initial proposal and should not be confused with the dataset used in the subsequent DYNOTEARS experiments.

### Final Experimental Configuration

| Attribute | Configuration |
|---|---|
| Market universe | S&P 100 |
| Usable stocks | 93 |
| Period | 1 July 2014 – 30 June 2019 |
| Observations | 1,257 daily log-return observations |
| Transformation | Log-returns |
| Scaling | Zero mean and unit variance |
| Data source | Yahoo Finance historical price data |

The experimental pipeline includes price retrieval, data cleaning, alignment, log-return computation and standardization.

The precise preprocessing and validation configurations are documented in the respective milestone reports.

Synthetic datasets are additionally used to evaluate structure recovery under controlled conditions with known ground-truth graphs.

---

## 6. Baselines and Evaluation Metrics

The project uses multiple baselines and evaluation metrics across its milestones.

### Baseline Models

- Vector Autoregression (VAR).
- Independent NOTEARS and Lasso.
- Market and sector factor models.
- Ridge regression.
- ARIMA.
- GARCH-based variance models.

The specific baselines vary according to the experiment and milestone.

### Evaluation Metrics

| Category | Metrics |
|---|---|
| Graph Structure | Edge count, density, sparsity, mean degree |
| Structure Recovery | Precision, Recall, F1, Structural Hamming Distance |
| Model Fit | Loss, BIC |
| Forecasting | RMSE, MAE |
| Probabilistic Evaluation | CRPS, Brier Score |
| Robustness | Bootstrap edge stability, selection frequency |
| Statistical Analysis | Confidence intervals and paired statistical tests |

---

## 7. Repository Organization

The repository contains separate submissions for each project milestone.

```text
G7_APL_Finance_team/
│
├── README.md
│
├── M1_G7_APL_Finance_team/
│   ├── README.md
│   ├── M1_Report.pdf
│   └── Code/
│       ├── README.md
│       ├── requirements.txt
│       └── src/
│           ├── data_loader.py
│           ├── m1_experiment.py
│           └── structure_learner.py
│
├── M2_G7_APL_Finance_team/
│   ├── README.md
│   ├── Code/
│   ├── Data/
│   ├── Report/
│   ├── Results/
│   └── Video/
│
├── M3_G7_APL_Finance_team/
│   ├── README.md
│   ├── Code/
│   ├── Data/
│   ├── Report/
│   └── Results/
│
└── M4_G7_APL_Finance_team/
    ├── README.md
    ├── code/
    ├── Data/
    ├── Report/
    ├── Results/
    ├── Presentation/
    └── Video/
```

Each milestone directory contains its corresponding submission materials.

---

## 8. Milestone Details

### M1 – Proposal & Scoping

Established the original research direction and initial methodology.

Key components:
- Problem formulation and project motivation.
- Literature review and SOTA positioning.
- Dynamic Bayesian Network representation.
- Initial ARX-based structure-learning approach.
- VAR baseline and proposed evaluation metrics.
- Initial experimental plan.

### M2 – Technical Foundation

Implemented the DYNOTEARS framework and established the technical foundation for subsequent experiments.

Key components:
- S&P 100 data pipeline.
- DYNOTEARS implementation from scratch.
- Synthetic simulation and validation.
- Reproduction of paper-based experiments.
- Baseline implementation.
- Initial forecasting and evaluation.
- Technical report and video demonstration.

### M3 – Systematic Comparison

Conducted systematic experiments on synthetic and real-world financial data.

Key components:
- Synthetic graph-recovery experiments.
- Sample-size and dimensionality analysis.
- Missing-data and measurement-noise experiments.
- Solver ablation.
- Rolling-origin forecasting evaluation.
- Comparison against baseline models.
- Statistical analysis and experimental results.

### M4 – Final Evaluation

Extended the experimental framework with additional evaluation, robustness and reproducibility analyses.

Key components:
- Rolling-origin forecasting.
- Distributional and probabilistic evaluation.
- Factor and sector-based baselines.
- Residual graph analysis.
- Bootstrap edge-stability experiments.
- ARIMA and GARCH comparisons.
- Error analysis and calibration.
- Final report, presentation and video demonstration.

---

## 9. Running the Project

Each milestone contains its own implementation and dependency documentation.

Refer to the corresponding milestone's README and requirements file for installation instructions and execution commands.

For the initial M1 implementation:

```bash
cd M1_G7_APL_Finance_team/Code
pip install -r requirements.txt
python src/m1_experiment.py
```

For M2, M3 and M4, follow the execution instructions documented in their respective code directories.

The M4 implementation also provides an experiment runner:

```bash
python run_all.py --dry-run
```

Refer to the M4 documentation for the complete execution sequence and reproducibility instructions.

---

## 10. Project Contributions

The project investigates several aspects of probabilistic graphical modelling and financial time-series analysis:

- **Dynamic Bayesian Networks:** Representation of contemporaneous and lagged dependencies.
- **Structure Learning:** Implementation of continuous optimization for DAG discovery.
- **Synthetic Validation:** Evaluation against known ground-truth structures.
- **Financial Applications:** Application of learned structures to S&P 100 stock returns.
- **Model Comparison:** Evaluation against statistical and factor-based baselines.
- **Robustness:** Investigation of sensitivity and stability of learned graph structures.
- **Reproducibility:** Modular code, archived datasets, experiment outputs and documentation.

---

## 11. Team Information

**Foram Gohel**  
Enrollment No.: AU2340054

**Priya Iyengar**  
Enrollment No.: AU2340263

Both members contributed to the project across its milestones. Specific individual contributions should be referred to in the relevant milestone documentation.

---

## 12. Video Demonstrations

### M2 – DYNOTEARS Implementation and Evaluation

[Watch the M2 Project Video](https://drive.google.com/drive/folders/1M3tHWr31MaXgN-LWe8WreHrU115yPbEM)

### M4 – Final Evaluation

[Watch the M4 Project Video](https://drive.google.com/drive/folders/1M3tHWr31MaXgN-LWe8WreHrU115yPbEM)

---

## 13. References

[1] Pamfil, R. et al. (2020). *DYNOTEARS: Structure Learning from Time-Series Data*. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS).

[2] Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). *DAGs with NO TEARS: Continuous Optimization for Structure Learning*. NeurIPS.

[3] Koller, D. and Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

[4] Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

---

**GitHub Repository:** https://github.com/spacestar54/G7_APL_Finance_team

"""
Task 3: Validate DYNOTEARS on simulated data with known ground truth.

Protocol follows Pamfil et al. (2020), Section 3 and Appendix B (B.1, B.3, B.4, B.7):
  - data from X = XW + YA + Z (ER intra-slice graph, ER inter-slice graph, Gaussian or exponential noise)
  - hyperparameters from Appendix B.3:
        n = 500: lambda_W = lambda_A = 0.05, tau_W = 0.3, tau_A = 0.1
        n = 50 : lambda_W = lambda_A = 0.2,  tau_W = 0.3, tau_A = 0.2
  - baseline: "NOTEARS + Lasso" (Murphy-style): static NOTEARS for W, Lasso regression for A (Appendix B.1)
  - metrics: precision, recall, F1, SHD, computed separately for W and A (Appendix B.7)
The paper used 5 runs per setting; this script uses 10 seeds by default (CSE 516 guidelines, Section 8.2)
and adds a paired Wilcoxon test (Section 8.4).

Usage
  python task3_simulation.py --quick      # d in {5,10}, 5 seeds, a few minutes
  python task3_simulation.py              # d in {5,10,20}, 10 seeds, Gaussian + exponential (longer)
  python task3_simulation.py --dims 5 10 20 50 --seeds 10     # closer to the paper (slow)
Outputs go to data_out/task3/: results_all_runs.csv, summary_table.csv, wilcoxon.csv, PNG figures.
Needs: numpy, pandas, scipy, matplotlib, scikit-learn  (pip install matplotlib scikit-learn)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task2_dynotears import fit_dynotears, simulate, edge_metrics

OUT = os.path.join("data_out", "task3")
HP = {500: dict(lam=0.05, tau_w=0.3, tau_a=0.1),     # Appendix B.3
      50: dict(lam=0.2, tau_w=0.3, tau_a=0.2)}


# ----------------------------------------------------------------------------
# STEP 1. The two methods
# ----------------------------------------------------------------------------
def run_dynotears(X, Y, hp):
    W, A, info = fit_dynotears(X, Y, hp["lam"], hp["lam"], hp["tau_w"], hp["tau_a"])
    return W, A, info["seconds"]


def run_notears_lasso(X, Y, hp):
    """Learn W and A independently: static NOTEARS on X, Lasso of X on lagged Y."""
    from sklearn.linear_model import Lasso
    t0 = time.time()
    n, d = X.shape
    W, _, _ = fit_dynotears(X, np.zeros((n, 0)), hp["lam"], hp["lam"], hp["tau_w"], 0.0)
    A = np.zeros((Y.shape[1], d))
    for j in range(d):
        A[:, j] = Lasso(alpha=hp["lam"], fit_intercept=False, max_iter=10000).fit(Y, X[:, j]).coef_
    A[np.abs(A) < hp["tau_a"]] = 0.0
    return W, A, time.time() - t0


METHODS = {"DYNOTEARS": run_dynotears, "NOTEARS+Lasso": run_notears_lasso}


# ----------------------------------------------------------------------------
# STEP 2. Experiment grid
# ----------------------------------------------------------------------------
def run_grid(dims, ns, noises, seeds, p=1, k_w=2.0, k_a=2.0):
    rows = []
    total = len(dims) * len(ns) * len(noises) * seeds
    done = 0
    for noise in noises:
        for n in ns:
            for d in dims:
                for seed in range(seeds):
                    X, Y, Wt, At = simulate(d=d, n=n, p=p, k_w=k_w, k_a=k_a,
                                            eta=1.0, noise=noise, seed=seed)
                    for name, fn in METHODS.items():
                        W, A, sec = fn(X, Y, HP[n])
                        mw, ma = edge_metrics(Wt, W), edge_metrics(At, A)
                        rows.append(dict(method=name, noise=noise, n=n, d=d, seed=seed, seconds=sec,
                                         f1_W=mw["f1"], prec_W=mw["precision"], rec_W=mw["recall"], shd_W=mw["shd"],
                                         f1_A=ma["f1"], prec_A=ma["precision"], rec_A=ma["recall"], shd_A=ma["shd"]))
                    done += 1
                    print(f"[{done}/{total}] noise={noise} n={n} d={d} seed={seed}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 3. Summary table (mean +/- SD), Wilcoxon test, plots
# ----------------------------------------------------------------------------
def summarize(df):
    g = df.groupby(["noise", "n", "d", "method"])
    cols = ["f1_W", "f1_A", "shd_W", "shd_A", "seconds"]
    mean, sd = g[cols].mean(), g[cols].std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.2f} +/- {s:.2f}" for m, s in zip(mean[c], sd[c].fillna(0))]
    return out.reset_index()


def wilcoxon_table(df):
    from scipy.stats import wilcoxon
    rows = []
    for (noise, n, d), sub in df.groupby(["noise", "n", "d"]):
        a = sub[sub.method == "DYNOTEARS"].sort_values("seed")
        b = sub[sub.method == "NOTEARS+Lasso"].sort_values("seed")
        for metric in ("f1_W", "f1_A"):
            diff = a[metric].values - b[metric].values
            try:
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                p = float("nan")
            rows.append(dict(noise=noise, n=n, d=d, metric=metric,
                             mean_diff=float(diff.mean()), p_value=float(p), n_pairs=len(diff)))
    return pd.DataFrame(rows)


def plot_f1(df, path):
    noises, ns = sorted(df.noise.unique()), sorted(df.n.unique(), reverse=True)
    fig, axes = plt.subplots(len(noises), len(ns), figsize=(5.2 * len(ns), 3.8 * len(noises)), squeeze=False)
    colors = {"DYNOTEARS": "tab:blue", "NOTEARS+Lasso": "tab:orange"}
    for i, noise in enumerate(noises):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            for m in METHODS:
                for col, ls in (("f1_W", "-"), ("f1_A", "--")):
                    s = df[(df.noise == noise) & (df.n == n) & (df.method == m)].groupby("d")[col].agg(["mean", "std"])
                    ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=colors[m], ls=ls,
                                marker="o", capsize=2, label=f"{m} ({'W' if col == 'f1_W' else 'A'})")
            ax.set_title(f"{noise} noise, n = {n}")
            ax.set_xlabel("Number of nodes d"); ax.set_ylabel("F1"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("F1 vs d (solid: W, dashed: A; mean +/- SD over seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def figure2_replica(path):
    """d=5, n=500, p=3, Gaussian noise, eta=1.5 (Pamfil et al., 2020, Figure 2)."""
    d, p = 5, 3
    X, Y, Wt, At = simulate(d=d, n=500, p=p, k_w=4.0, k_a=1.0, eta=1.5, noise="gaussian", seed=1)
    W, A, _ = fit_dynotears(X, Y, 0.05, 0.05, 0.01, 0.01)
    pairs = [("W (intra)", Wt, W)] + [(f"A{i + 1} (inter)", At[i * d:(i + 1) * d], A[i * d:(i + 1) * d]) for i in range(p)]
    vmax = max(np.abs(Wt).max(), np.abs(W).max())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for c, (title, T, E) in enumerate(pairs):
        for r, (M, lab) in enumerate(((T, "True"), (E, "Estimated"))):
            ax = axes[r][c]
            ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            for (a, b), v in np.ndenumerate(M):
                ax.text(b, a, f"{v:.2f}" if v else "0", ha="center", va="center", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{lab} {title}", fontsize=9)
    fig.suptitle("Replica of the Figure 2 setting (d=5, n=500, p=3). Row = source, column = target.")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return edge_metrics(Wt, W), edge_metrics(At, A)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 50])
    ap.add_argument("--noises", nargs="+", default=["gaussian", "exponential"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    if a.quick:
        a.dims, a.seeds = [5, 10], 5

    os.makedirs(OUT, exist_ok=True)
    df = run_grid(a.dims, a.ns, a.noises, a.seeds)
    df.to_csv(os.path.join(OUT, "results_all_runs.csv"), index=False)

    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, "summary_table.csv"), index=False)
    wil = wilcoxon_table(df)
    wil.to_csv(os.path.join(OUT, "wilcoxon.csv"), index=False)
    plot_f1(df, os.path.join(OUT, "f1_vs_d.png"))
    mw, ma = figure2_replica(os.path.join(OUT, "figure2_replica.png"))

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 200)
    print("\n=== Summary (mean +/- SD over seeds) ===")
    print(summ.to_string(index=False))
    print("\n=== Paired Wilcoxon, DYNOTEARS vs NOTEARS+Lasso (F1) ===")
    print(wil.round(4).to_string(index=False))
    print(f"\nFigure-2 replica: W F1={mw['f1']:.2f} SHD={mw['shd']} | A F1={ma['f1']:.2f} SHD={ma['shd']}")
    print(f"\nSaved tables and figures in {OUT}/")
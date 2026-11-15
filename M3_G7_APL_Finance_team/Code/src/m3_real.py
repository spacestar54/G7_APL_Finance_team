"""
M3, family B: real S&P 100 data, expanding-window evaluation (CSE516, DYNOTEARS project).

What it does
  1. Splits the 1257 return rows into K time-ordered folds (train on everything before the test block,
     2-day gap, test on the next block). Standardization uses TRAIN statistics only (no leakage).
  2. Fits on each fold: DYNOTEARS (lambda chosen on the last 25% of the training block), static NOTEARS,
     and the baselines: zero, train-mean, VAR(1) OLS, ridge-VAR, lasso-VAR.
  3. Turns every model into a Gaussian predictive distribution N(mu, sigma^2) (sigma from training residuals)
     and reports: RMSE, MAE, negative log-likelihood, 90% interval coverage, Brier score and ECE for the
     up/down direction, and a pooled reliability diagram.
  4. Tests each model against the zero predictor with a moving-block bootstrap (returns are autocorrelated
     in volatility, so i.i.d. resampling of days is too optimistic).
  5. Edge stability: how often each edge of W and A appears across folds.

Usage (run from the repo root; needs data_out/log_returns.csv from Task 1)
  python m3_real.py --quick                 # 20 stocks, 3 folds, one lambda: a few minutes
  python m3_real.py --jobs 2                # full run: 93 stocks, 5 folds, lambda grid (roughly an hour or more)
  python m3_real.py --lams 0.1 --jobs 2     # skip lambda tuning (much faster)
Outputs go to data_out/m3_real/.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from joblib import Parallel, delayed

from task2_dynotears import fit_dynotears, forecast

warnings.filterwarnings("ignore")
OUT = os.path.join("data_out", "m3_real")
MODELS = ["Zero predictor", "Train-mean predictor", "VAR(1) OLS", "Ridge-VAR(1)", "Lasso-VAR(1)", "DYNOTEARS"]
RIDGE_GRID = [1, 10, 100, 1e3, 1e4]
LASSO_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
N_BINS = 10


def make_xy(block):
    """Lag-1 design built inside one block, so no information crosses a split."""
    return block[1:], block[:-1]


def make_folds(n_rows, n_folds, test_len, gap):
    folds = []
    for k in range(n_folds):
        te_end = n_rows - (n_folds - 1 - k) * test_len
        te_start = te_end - test_len
        folds.append((te_start - gap, te_start, te_end))      # train end, test start, test end
    return folds


# ----------------------------------------------------------------------------
# Baseline fitters (all linear maps Y -> X)
# ----------------------------------------------------------------------------
def fit_ridge(Y, X, a):
    return np.linalg.solve(Y.T @ Y + a * np.eye(Y.shape[1]), Y.T @ X)


def fit_lasso(Y, X, a):
    from sklearn.linear_model import Lasso
    B = np.zeros((Y.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        B[:, j] = Lasso(alpha=a, fit_intercept=False, max_iter=5000).fit(Y, X[:, j]).coef_
    return B


def rmse(X, P):
    return float(np.sqrt(np.mean((X - P) ** 2)))


def pick(fitter, grid, Xtr, Ytr):
    """Choose a penalty on the last 25% of the training block (time order kept)."""
    cut = int(0.75 * len(Xtr))
    errs = [rmse(Xtr[cut:], Ytr[cut:] @ fitter(Ytr[:cut], Xtr[:cut], a)) for a in grid]
    return grid[int(np.argmin(errs))]


# ----------------------------------------------------------------------------
# Probability-quality metrics
# ----------------------------------------------------------------------------
def prob_metrics(X, mu, sig):
    z = (X - mu) / sig
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2))
    cover = float(np.mean(np.abs(z) < 1.645))
    p = norm.cdf(mu / sig).ravel()
    y = (X > 0).astype(float).ravel()
    brier = float(np.mean((p - y) ** 2))
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    cnt = np.bincount(bins, minlength=N_BINS).astype(float)
    sp = np.bincount(bins, weights=p, minlength=N_BINS)
    sy = np.bincount(bins, weights=y, minlength=N_BINS)
    ece = float(np.sum(np.abs(sp - sy)) / len(p))
    return dict(nll=nll, cover90=cover, brier=brier, ece=ece), (sp, sy, cnt)


# ----------------------------------------------------------------------------
# One fold
# ----------------------------------------------------------------------------
def run_fold(k, fold, rets, args):
    tr_end, te_start, te_end = fold
    arr = rets.values
    mu_s, sd_s = arr[:tr_end].mean(axis=0), arr[:tr_end].std(axis=0)
    Z = (arr - mu_s) / sd_s                                    # train-only standardization
    Xtr, Ytr = make_xy(Z[:tr_end])
    Xte, Yte = make_xy(Z[te_start:te_end])
    n_tr, d = Xtr.shape

    preds = {}        # name -> (test mean, train mean, fit seconds)
    preds["Zero predictor"] = (np.zeros_like(Xte), np.zeros_like(Xtr), 0.0)
    m = Xtr.mean(axis=0)
    preds["Train-mean predictor"] = (np.tile(m, (len(Xte), 1)), np.tile(m, (n_tr, 1)), 0.0)

    t0 = time.time()
    B = np.linalg.lstsq(Ytr, Xtr, rcond=None)[0]
    preds["VAR(1) OLS"] = (Yte @ B, Ytr @ B, time.time() - t0)

    t0 = time.time()
    a = pick(fit_ridge, RIDGE_GRID, Xtr, Ytr)
    B = fit_ridge(Ytr, Xtr, a)
    preds["Ridge-VAR(1)"] = (Yte @ B, Ytr @ B, time.time() - t0)

    t0 = time.time()
    al = pick(fit_lasso, LASSO_GRID, Xtr, Ytr)
    B = fit_lasso(Ytr, Xtr, al)
    preds["Lasso-VAR(1)"] = (Yte @ B, Ytr @ B, time.time() - t0)

    # DYNOTEARS: choose lambda on the last 25% of the training block, then refit on all of it
    lam_best = args.lams[0]
    tune_s = 0.0
    if len(args.lams) > 1:
        t0 = time.time()
        cut = int(0.75 * n_tr)
        best = np.inf
        for lam in sorted(args.lams, reverse=True):            # ties go to the larger (sparser) lambda
            W, A, _ = fit_dynotears(Xtr[:cut], Ytr[:cut], lam, lam, args.tau, args.tau)
            e = rmse(Xtr[cut:], forecast(Ytr[cut:], W, A))
            if e < best - 1e-12:
                best, lam_best = e, lam
        tune_s = time.time() - t0
    W, A, info = fit_dynotears(Xtr, Ytr, lam_best, lam_best, args.tau, args.tau)
    inv = np.linalg.inv(np.eye(d) - W)
    preds["DYNOTEARS"] = (Yte @ A @ inv, Ytr @ A @ inv, info["seconds"])
    Ws, _, info_s = fit_dynotears(Xtr, np.zeros((n_tr, 0)), lam_best, lam_best, args.tau, args.tau)

    rows, day_mse, rel = [], {}, {}
    for name in MODELS:
        mu_te, mu_tr, sec = preds[name]
        sig = np.sqrt(np.mean((Xtr - mu_tr) ** 2, axis=0))     # per-stock sigma from TRAIN residuals
        pm, r = prob_metrics(Xte, mu_te, sig)
        rows.append(dict(model=name, fold=k, rmse=rmse(Xte, mu_te), mae=float(np.mean(np.abs(Xte - mu_te))),
                         fit_seconds=sec, **pm))
        day_mse[name] = np.mean((Xte - mu_te) ** 2, axis=1)
        rel[name] = r

    corr = np.corrcoef(Xtr.T)
    iu = ~np.eye(d, dtype=bool)
    connected = ((W != 0) | (W.T != 0)) & iu
    struct = dict(fold=k, lam=lam_best, tune_seconds=tune_s, nnz_W=int((W != 0).sum()), nnz_A=int((A != 0).sum()),
                  nnz_W_static=int((Ws != 0).sum()), pos_frac_W=float((W[W != 0] > 0).mean()) if (W != 0).any() else np.nan,
                  edges_removed_for_dag=info["edges_removed_for_dag"], dyn_seconds=info["seconds"],
                  static_seconds=info_s["seconds"],
                  mean_corr_connected=float(corr[connected].mean()) if connected.any() else np.nan,
                  mean_corr_other=float(corr[iu & ~connected].mean()),
                  pc1_variance_share=float(np.linalg.eigvalsh(np.cov(Xtr.T))[-1] / np.trace(np.cov(Xtr.T))),
                  ridge_alpha=a, lasso_alpha=al,
                  train_dates=f"{rets.index[0].date()}..{rets.index[tr_end - 1].date()}",
                  test_dates=f"{rets.index[te_start].date()}..{rets.index[te_end - 1].date()}")
    print(f"  fold {k + 1} done: lam={lam_best}, edges W={struct['nnz_W']} A={struct['nnz_A']}, "
          f"DYNOTEARS RMSE={rows[-1]['rmse']:.4f} vs zero {rows[0]['rmse']:.4f}", flush=True)
    return dict(rows=rows, day_mse=day_mse, rel=rel, struct=struct, W=W, A=A)


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def block_boot(diff, rng, block=10, B=2000):
    """Moving-block bootstrap of the mean of a per-day loss difference."""
    n = len(diff)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - block + 1, nb)
        means[b] = diff[(st[:, None] + np.arange(block)).ravel()[:n]].mean()
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(min(p, 1.0))


def main(args):
    os.makedirs(OUT, exist_ok=True)
    rets = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True)
    if args.max_d:
        rets = rets.iloc[:, :args.max_d]
    folds = make_folds(len(rets), args.folds, args.test_len, args.gap)
    print(f"{rets.shape[0]} days x {rets.shape[1]} stocks, {len(folds)} folds, test length {args.test_len}, "
          f"lambda grid {args.lams}, tau {args.tau}", flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(k, f, rets, args) for k, f in enumerate(folds))
    tickers = list(rets.columns)

    # ---- per-fold table and summary (mean +/- SD, 95% CI over folds) ----
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    df.to_csv(os.path.join(OUT, "forecast_by_fold.csv"), index=False)
    cols = ["rmse", "mae", "nll", "cover90", "brier", "ece", "fit_seconds"]
    K = args.folds
    rows = []
    for name in MODELS:
        s = df[df.model == name]
        row = dict(model=name)
        for c in cols:
            sd = s[c].std(ddof=1) if K > 1 else 0.0
            row[c] = f"{s[c].mean():.4f} +/- {sd:.4f}"
            row[c + "_ci95"] = float(stats.t.ppf(0.975, K - 1) * sd / np.sqrt(K)) if K > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "forecast_summary.csv"), index=False)

    # ---- block-bootstrap test against the zero predictor (all test days pooled) ----
    rng = np.random.default_rng(0)
    zero = np.concatenate([x["day_mse"]["Zero predictor"] for x in res])
    trows = []
    for name in MODELS[1:]:
        mod = np.concatenate([x["day_mse"][name] for x in res])
        lo, hi, p = block_boot(zero - mod, rng)
        trows.append(dict(model=name, mean_mse_gain_vs_zero=float((zero - mod).mean()), ci95_low=lo, ci95_high=hi,
                          block_bootstrap_p=p, note="positive gain = better than zero predictor"))
    pd.DataFrame(trows).to_csv(os.path.join(OUT, "tests_vs_zero.csv"), index=False)

    # ---- structure table and edge stability ----
    st = pd.DataFrame([x["struct"] for x in res])
    st.to_csv(os.path.join(OUT, "structure_by_fold.csv"), index=False)
    Wb = np.mean([(x["W"] != 0).astype(float) for x in res], axis=0)
    Ab = np.mean([(x["A"] != 0).astype(float) for x in res], axis=0)
    erows = []
    for kind, M, W_all in (("W", Wb, [x["W"] for x in res]), ("A", Ab, [x["A"] for x in res])):
        for i, j in zip(*np.nonzero(M)):
            erows.append(dict(matrix=kind, source=tickers[i], target=tickers[j], freq=float(M[i, j]),
                              mean_weight=float(np.mean([w[i, j] for w in W_all]))))
    ed = pd.DataFrame(erows)
    if len(ed):
        ed = ed.sort_values("freq", ascending=False)
    ed.to_csv(os.path.join(OUT, "edge_stability.csv"), index=False)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 4))
    s = df.groupby("model").rmse.agg(["mean", "std"]).reindex(MODELS)
    ax.barh(range(len(MODELS)), s["mean"], xerr=s["std"].fillna(0), color="tab:blue", capsize=3)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.invert_yaxis()
    lo = (s["mean"] - s["std"].fillna(0)).min()
    ax.set_xlim(lo * 0.98, (s["mean"] + s["std"].fillna(0)).max() * 1.02)
    ax.axvline(s.loc["Zero predictor", "mean"], color="k", ls="--", lw=.8)
    ax.set_xlabel("Next-day RMSE (mean +/- SD over folds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_rmse.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect calibration")
    for name in MODELS:
        sp = sum(x["rel"][name][0] for x in res)
        sy = sum(x["rel"][name][1] for x in res)
        cnt = sum(x["rel"][name][2] for x in res)
        ok = cnt > 0
        ax.plot(sp[ok] / cnt[ok], sy[ok] / cnt[ok], marker="o", ms=4, lw=1, label=name)
    ax.set_xlabel("Predicted P(return > 0)")
    ax.set_ylabel("Observed frequency of up days")
    ax.set_title("Reliability diagram (pooled over folds and stocks)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_reliability.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = df.groupby("model")[["fit_seconds", "rmse"]].mean().reindex(MODELS)
    ax.scatter(np.maximum(g.fit_seconds, 1e-3), g.rmse)
    for name, r in g.iterrows():
        ax.annotate(name, (max(r.fit_seconds, 1e-3), r.rmse), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Fit time (s, log scale; 0 plotted at 1e-3)")
    ax.set_ylabel("Next-day RMSE")
    ax.set_title("Inference error vs computational time")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_rmse_vs_time.png"), dpi=150)
    plt.close(fig)

    if Wb.any():
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Wb, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("Edge stability of W: share of folds with edge (row -> column)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plot_edge_stability_W.png"), dpi=150)
        plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== Forecast summary (mean +/- SD over folds) ===")
    print(pd.DataFrame(rows)[["model"] + cols].to_string(index=False))
    print("\n=== Block-bootstrap test vs zero predictor (positive gain = better) ===")
    print(pd.DataFrame(trows).drop(columns="note").round(5).to_string(index=False))
    print("\n=== Structure per fold ===")
    print(st[["fold", "lam", "nnz_W", "nnz_A", "pos_frac_W", "mean_corr_connected", "mean_corr_other",
              "pc1_variance_share", "dyn_seconds"]].round(3).to_string(index=False))
    print(f"\nSaved tables and plots to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 stocks, 3 folds, one lambda")
    ap.add_argument("--max-d", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-len", type=int, default=150)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.quick:
        a.max_d, a.folds, a.lams = 20, 3, [0.1]
    main(a)

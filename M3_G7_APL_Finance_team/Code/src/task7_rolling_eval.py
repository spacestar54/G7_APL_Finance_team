"""
Task 7 (M3, experiments 1 and 2): rolling-origin forecast evaluation + Ljung-Box tables.

Method sources
  Rolling forecasting origin ("time series cross-validation"), forecast accuracy measures (RMSE, RMSSE),
  skill scores, benchmark methods, residual diagnostics and the Ljung-Box portmanteau test:
      Hyndman, R.J. and Athanasopoulos, G. (2021), Forecasting: Principles and Practice, 3rd ed., OTexts,
      https://otexts.com/fpp3/  (Sections 5.2, 5.4, 5.8, 5.9, 5.10)
  DYNOTEARS forecast X_hat = Y A (I - W)^(-1):  Pamfil et al. (2020), AISTATS, Section 2.3.

Design (leakage-free)
  - Uses the RAW log-returns (data_out/log_returns.csv). At every origin the data are standardized with
    the mean and SD of the training window only.
  - Origins start at --start rows, then move forward by --step rows (expanding training window).
    For each origin, every model is fitted on rows [0, origin) and makes ONE-STEP-AHEAD forecasts for the next
    --step rows, using the actual previous-day returns as inputs (so the models are refitted every --step days,
    not every day, which keeps DYNOTEARS affordable).
  - Models: zero predictor (= naive forecast of the log price), training-mean (= drift), naive-on-returns
    (tomorrow = today, the literal FPP3 naive applied to returns), AR(1) per stock (ARIMA(1,0,0)), VAR(1) OLS,
    ridge-VAR(1) (penalty chosen inside the training window), optional Lasso-VAR(1), and DYNOTEARS
    (refitted every --dyn-every origins; reused in between, which is still using only past data).

Usage
  python task7_rolling_eval.py --no-dyn            # fast run without DYNOTEARS
  python task7_rolling_eval.py                     # includes DYNOTEARS (about half an hour or more)
  python task7_rolling_eval.py --lasso --dyn-every 3
Outputs in data_out/task7/: task7_block_errors.csv, task7_skill_table.csv, task7_ljungbox.csv, PNG plots.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

OUT = os.path.join("data_out", "task7")
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Model fitters (training window only)
# ----------------------------------------------------------------------------
def fit_ridge(Y, X, alpha):
    return np.linalg.solve(Y.T @ Y + alpha * np.eye(Y.shape[1]), Y.T @ X)


def fit_lasso(Y, X, alpha):
    from sklearn.linear_model import Lasso
    B = np.zeros((Y.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        B[:, j] = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000).fit(Y, X[:, j]).coef_
    return B


def choose_alpha(fitter, grid, Ytr, Xtr):
    """Penalty chosen on the last 25% of the training window (time order kept)."""
    cut = int(0.75 * len(Xtr))
    errs = [np.mean((Xtr[cut:] - Ytr[cut:] @ fitter(Ytr[:cut], Xtr[:cut], a)) ** 2) for a in grid]
    return grid[int(np.argmin(errs))]


# ----------------------------------------------------------------------------
# Rolling-origin evaluation
# ----------------------------------------------------------------------------
def rolling_eval(R, start, step, use_dyn, dyn_every, use_lasso, tau):
    n, d = R.shape
    origins = list(range(start, n - step + 1, step))
    print(f"{len(origins)} origins from row {origins[0]} to {origins[-1]} (step {step}), d = {d}")
    rows, W_dyn, A_dyn = [], None, None
    for k, o in enumerate(origins):
        mu, sd = R[:o].mean(0), R[:o].std(0)
        Z = (R - mu) / sd                                   # training-window scaling only
        Xtr, Ytr = Z[1:o], Z[:o - 1]
        Xte, Yte = Z[o:o + step], Z[o - 1:o + step - 1]
        s2 = ((Z[1:o] - Z[:o - 1]) ** 2).mean(0)            # naive-on-returns scale for RMSSE (FPP3 5.8)

        preds = {"Zero predictor": np.zeros_like(Xte),
                 "Training mean (drift)": np.tile(Xtr.mean(0), (step, 1)),
                 "Naive on returns": Yte.copy()}
        b = (Ytr * Xtr).sum(0) / (Ytr ** 2).sum(0)
        preds["AR(1) per stock"] = Yte * b
        preds["VAR(1) OLS"] = Yte @ np.linalg.lstsq(Ytr, Xtr, rcond=None)[0]
        preds["Ridge-VAR(1)"] = Yte @ fit_ridge(Ytr, Xtr, choose_alpha(fit_ridge, [1, 10, 100, 1e3, 1e4], Ytr, Xtr))
        if use_lasso:
            preds["Lasso-VAR(1)"] = Yte @ fit_lasso(Ytr, Xtr, choose_alpha(fit_lasso, [0.003, 0.01, 0.03, 0.1, 0.3], Ytr, Xtr))
        if use_dyn:
            if W_dyn is None or k % dyn_every == 0:
                from task2_dynotears import fit_dynotears
                t0 = time.time()
                print(f"  origin {o}: refitting DYNOTEARS (about 5 minutes for d=93; progress lines follow) ...", flush=True)
                W_dyn, A_dyn, _ = fit_dynotears(Xtr, Ytr, 0.1, 0.1, tau, tau, verbose=True)
                print(f"  origin {o}: DYNOTEARS refit in {time.time() - t0:.0f}s "
                      f"(edges W={int((W_dyn != 0).sum())}, A={int((A_dyn != 0).sum())})", flush=True)
            preds["DYNOTEARS"] = Yte @ A_dyn @ np.linalg.inv(np.eye(d) - W_dyn)

        for name, P in preds.items():
            e2 = (Xte - P) ** 2
            rows.append(dict(origin=o, model=name, sse=float(e2.sum()), count=int(e2.size),
                             mse=float(e2.mean()), rmsse=float(np.sqrt((e2 / s2).mean()))))
        if k % 5 == 0:
            print(f"  done origin {k + 1}/{len(origins)}", flush=True)
    return pd.DataFrame(rows)


def skill_table(df):
    piv = df.pivot(index="origin", columns="model", values="mse")
    zero = piv["Zero predictor"]
    rows = []
    for m in piv.columns:
        sub = df[df.model == m]
        pooled = float(np.sqrt(sub.sse.sum() / sub["count"].sum()))
        skill = 1 - np.sqrt(piv[m] / zero)                   # per-block RMSE skill vs zero predictor
        diff = zero - piv[m]                                 # positive = better than zero
        try:
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
        except ValueError:
            p = np.nan
        boot = [skill.values[rng.integers(0, len(skill), len(skill))].mean() for _ in range(2000)]
        rows.append(dict(model=m, pooled_RMSE=pooled,
                         block_RMSE_mean=float(np.sqrt(sub.mse).mean()), block_RMSE_sd=float(np.sqrt(sub.mse).std(ddof=1)),
                         RMSSE_mean=float(sub.rmsse.mean()),
                         skill_vs_zero_mean=float(skill.mean()), skill_sd=float(skill.std(ddof=1)),
                         skill_CI95_low=float(np.percentile(boot, 2.5)), skill_CI95_high=float(np.percentile(boot, 97.5)),
                         wilcoxon_p_vs_zero=float(p), n_blocks=len(skill)))
    return pd.DataFrame(rows).sort_values("pooled_RMSE")


# ----------------------------------------------------------------------------
# Ljung-Box tables (FPP3 Section 5.4)
# ----------------------------------------------------------------------------
def lb_pvalues(M, lags):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return np.array([acorr_ljungbox(M[:, j], lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
                     for j in range(M.shape[1])])


def ljung_box_table(R, lags):
    from statsmodels.stats.multitest import multipletests
    Z = (R - R.mean(0)) / R.std(0)
    series = {"Returns": Z, "Squared returns": Z ** 2}
    b = (Z[:-1] * Z[1:]).sum(0) / (Z[:-1] ** 2).sum(0)
    series["AR(1) residuals"] = Z[1:] - Z[:-1] * b
    fit_p, data_p = os.path.join("data_out", "fit_dynotears.npz"), os.path.join("data_out", "dataset.npz")
    if os.path.exists(fit_p) and os.path.exists(data_p):
        F, D = np.load(fit_p, allow_pickle=True), np.load(data_p, allow_pickle=True)
        W, A = F["W"].copy(), F["A"].copy()
        W[np.abs(W) < 0.05] = 0
        A[np.abs(A) < 0.05] = 0
        Xt, Yt = D["X_train"], D["Y_train"]
        series["DYNOTEARS structural residuals (train)"] = Xt - Xt @ W - Yt @ A
    rows = []
    for name, M in series.items():
        p = lb_pvalues(M, lags)
        rows.append(dict(series=name, n_stocks=len(p), lag=lags,
                         n_p_below_0_05=int((p < 0.05).sum()), expected_by_chance=round(0.05 * len(p), 1),
                         n_p_below_0_01=int((p < 0.01).sum()),
                         n_rejected_after_BH_FDR_5pct=int(multipletests(p, alpha=0.05, method="fdr_bh")[0].sum()),
                         median_p=float(np.median(p))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main(a):
    os.makedirs(OUT, exist_ok=True)
    R = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True)
    print(f"Raw log-returns: {R.shape[0]} rows x {R.shape[1]} stocks")
    Rv = R.values

    df = rolling_eval(Rv, a.start, a.step, not a.no_dyn, a.dyn_every, a.lasso, a.tau)
    df.to_csv(os.path.join(OUT, "task7_block_errors.csv"), index=False)
    tab = skill_table(df)
    tab.to_csv(os.path.join(OUT, "task7_skill_table.csv"), index=False)

    lb = ljung_box_table(Rv, a.lags)
    lb.to_csv(os.path.join(OUT, "task7_ljungbox.csv"), index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== Rolling-origin forecast accuracy (lower RMSE is better; skill > 0 beats the zero predictor) ===")
    print(tab.round(4).to_string(index=False))
    print(f"\n=== Ljung-Box tests (lag {a.lags}) across stocks ===")
    print(lb.round(3).to_string(index=False))

    piv = df.pivot(index="origin", columns="model", values="mse")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in piv.columns:
        ax.plot(piv.index, np.sqrt(piv[m]), label=m, lw=1.6 if m in ("Zero predictor", "DYNOTEARS") else 1)
    ax.set_xlabel("Forecast origin (row index)"); ax.set_ylabel("Block RMSE")
    ax.set_title("One-step-ahead RMSE per rolling-origin block"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task7_rmse_by_origin.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    t = tab.set_index("model")
    err = np.vstack([t.skill_vs_zero_mean - t.skill_CI95_low, t.skill_CI95_high - t.skill_vs_zero_mean])
    ax.barh(t.index[::-1], t.skill_vs_zero_mean[::-1], xerr=err[:, ::-1], color="tab:blue", capsize=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Skill vs zero predictor (mean over blocks, 95% bootstrap CI over blocks)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task7_skill_scores.png"), dpi=150); plt.close(fig)
    print(f"\nSaved tables and figures in {OUT}/")
    print("Note: blocks use overlapping training windows, so block-level tests and CIs are approximate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=600, help="rows in the first training window")
    ap.add_argument("--step", type=int, default=20, help="test block length and refit interval")
    ap.add_argument("--no-dyn", action="store_true", help="skip DYNOTEARS (fast run)")
    ap.add_argument("--dyn-every", type=int, default=5, help="refit DYNOTEARS every k origins")
    ap.add_argument("--lasso", action="store_true", help="also run Lasso-VAR(1)")
    ap.add_argument("--tau", type=float, default=0.05, help="threshold for DYNOTEARS weights")
    ap.add_argument("--lags", type=int, default=10, help="Ljung-Box lag (FPP3 suggests 10 for non-seasonal data)")
    main(ap.parse_args())
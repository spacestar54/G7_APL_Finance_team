"""
Tasks 5 and 6: baselines and next-day RMSE (the first inference result).

Inference query: predict tomorrow's standardized returns for all stocks from today's returns.
  DYNOTEARS forecast (reduced form of X = XW + YA + Z):   X_hat = Y A (I - W)^(-1)
        (Pamfil et al., 2020, Section 2.3, Eqs. 8-9). Uses ONLY past information.
  Static NOTEARS has no lagged term, so its forecast is the zero vector.
  Baselines: zero predictor, training-mean predictor, VAR(1) by OLS, ridge-VAR, Lasso-VAR.
  Ridge/Lasso penalties are chosen on the last 25% of the TRAINING block, never on validation.

Also reported (NOT a forecast): the structural conditional error ||X - XW - YA||, which uses same-day returns
of a stock's parents. It is the paper's validation loss (Figure 12) expressed per entry.

Statistics: per-day squared errors on the validation block, paired Wilcoxon test against the zero predictor,
and a bootstrap 95% CI for the RMSE difference (i.i.d. resampling of days; it ignores autocorrelation, so treat
it as approximate).

Usage:  python task5_6_baselines_rmse.py [--tau 0.05]
Needs data_out/dataset.npz (Task 1) and data_out/fit_dynotears.npz, fit_static_notears.npz (Task 2).
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

OUT = "data_out"
rng = np.random.default_rng(0)


def rmse(X, Xh):
    return float(np.sqrt(np.mean((X - Xh) ** 2)))


def day_mse(X, Xh):
    return np.mean((X - Xh) ** 2, axis=1)           # one number per validation day


# ---------------------------------------------------------------------------
# Baseline fitters (all use only the training block)
# ---------------------------------------------------------------------------
def fit_ridge(Y, X, alpha):
    return np.linalg.solve(Y.T @ Y + alpha * np.eye(Y.shape[1]), Y.T @ X)


def fit_lasso(Y, X, alpha):
    from sklearn.linear_model import Lasso
    B = np.zeros((Y.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        B[:, j] = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000).fit(Y, X[:, j]).coef_
    return B


def choose_alpha(fitter, grid, Ytr, Xtr):
    """Pick the penalty on the last 25% of the training block (time order kept)."""
    cut = int(0.75 * len(Xtr))
    best, best_a = np.inf, grid[0]
    for a in grid:
        B = fitter(Ytr[:cut], Xtr[:cut], a)
        e = rmse(Xtr[cut:], Ytr[cut:] @ B)
        if e < best:
            best, best_a = e, a
    return best_a


# ---------------------------------------------------------------------------
def main(tau):
    D = np.load(os.path.join(OUT, "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    n_va, d = Xva.shape
    print(f"Train {Xtr.shape}, validation {Xva.shape}")

    F = np.load(os.path.join(OUT, "fit_dynotears.npz"), allow_pickle=True)
    W, A = F["W"].copy(), F["A"].copy()
    W[np.abs(W) < tau] = 0.0
    A[np.abs(A) < tau] = 0.0
    S = np.load(os.path.join(OUT, "fit_static_notears.npz"), allow_pickle=True)
    Ws = S["W"].copy()
    Ws[np.abs(Ws) < tau] = 0.0
    print(f"DYNOTEARS at tau={tau}: edges W = {(W != 0).sum()}, A = {(A != 0).sum()}")

    preds, times = {}, {}

    preds["Zero predictor"] = np.zeros_like(Xva)
    preds["Train-mean predictor"] = np.tile(Xtr.mean(axis=0), (n_va, 1))

    t0 = time.time()
    inv = np.linalg.inv(np.eye(d) - W)
    preds["DYNOTEARS forecast"] = Yva @ A @ inv
    times["DYNOTEARS forecast"] = time.time() - t0
    preds["Static NOTEARS forecast"] = np.zeros_like(Xva)       # no lagged term

    t0 = time.time(); B_ols = np.linalg.lstsq(Ytr, Xtr, rcond=None)[0]
    preds["VAR(1) OLS"] = Yva @ B_ols; times["VAR(1) OLS"] = time.time() - t0

    grid_r = [1, 10, 100, 1e3, 1e4]
    a_r = choose_alpha(fit_ridge, grid_r, Ytr, Xtr)
    preds["Ridge-VAR(1)"] = Yva @ fit_ridge(Ytr, Xtr, a_r)

    grid_l = [0.003, 0.01, 0.03, 0.1, 0.3]
    a_l = choose_alpha(fit_lasso, grid_l, Ytr, Xtr)
    B_l = fit_lasso(Ytr, Xtr, a_l)
    preds["Lasso-VAR(1)"] = Yva @ B_l
    print(f"Chosen penalties: ridge alpha = {a_r}, lasso alpha = {a_l} "
          f"(Lasso-VAR nonzeros: {(B_l != 0).sum()})")

    # ---- forecast table with statistics ----
    zero_day = day_mse(Xva, preds["Zero predictor"])
    rows = []
    for name, P in preds.items():
        dm = day_mse(Xva, P)
        diff = zero_day - dm                       # positive = better than zero predictor
        boot = [np.sqrt(zero_day[i].mean()) - np.sqrt(dm[i].mean())
                for i in (rng.integers(0, n_va, n_va) for _ in range(2000))]
        try:
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = float("nan")
        rows.append(dict(model=name, RMSE=rmse(Xva, P),
                         gain_vs_zero=np.sqrt(zero_day.mean()) - np.sqrt(dm.mean()),
                         CI95_low=np.percentile(boot, 2.5), CI95_high=np.percentile(boot, 97.5),
                         wilcoxon_p=p, latency_s=times.get(name, np.nan)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "task6_forecast_rmse.csv"), index=False)

    # ---- structural conditional error (not a forecast) ----
    Dn, Ds = Xva - Xva @ W - Yva @ A, Xva - Xva @ Ws
    struct = {"DYNOTEARS (structural, uses same-day parents)": rmse(Xva, Xva - Dn),
              "Static NOTEARS (structural, uses same-day parents)": rmse(Xva, Xva - Ds),
              "Zero model": rmse(Xva, 0 * Xva)}
    print("\n=== Next-day FORECAST RMSE on validation block (lower is better) ===")
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    print("\n=== Structural conditional RMSE (NOT a forecast; relates to the paper's validation loss) ===")
    for k, v in struct.items():
        print(f"  {k}: {v:.4f}")
    print("  (Frobenius loss = RMSE * sqrt(n*d); for comparison, the paper's values are 167.387 / 167.784 / 217.596.)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(tab.model[::-1], tab.RMSE[::-1], color="tab:blue")
    ax.axvline(tab.RMSE[0], color="k", ls="--", lw=.8, label="zero predictor")
    ax.set_xlim(tab.RMSE.min() * 0.97, tab.RMSE.max() * 1.03)
    ax.set_xlabel("Next-day RMSE (standardized returns)"); ax.legend(); ax.set_title("Forecast RMSE, validation block")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task6_forecast_rmse.png"), dpi=150)
    print("\nSaved task6_forecast_rmse.csv and .png in data_out/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05, help="threshold applied to saved W and A")
    main(ap.parse_args().tau)
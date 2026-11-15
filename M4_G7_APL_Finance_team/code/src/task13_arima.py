"""
Task 13 (M3/M4): the ARIMA family, done properly.

What was missing so far: only AR(1) per stock (= ARIMA(1,0,0)) and the zero / mean benchmarks. This script adds
  1. ARIMA(p,0,q) per stock with the order chosen by AICc over p, q in {0..maxpq} (FPP3 Sections 9.6-9.7 style
     automatic selection; returns are already stationary so d = 0, no constant because the data are standardized);
  2. ARIMAX / dynamic regression: AR(1) errors with the lagged equal-weight market return as a regressor
     (FPP3 Chapter 10);
  3. ARIMA mean + GARCH(1,1) variance on the ARIMA residuals (probabilistic forecasts, scored by CRPS).
Hyndman, R.J. and Athanasopoulos, G. (2021), Forecasting: Principles and Practice, 3rd ed., https://otexts.com/fpp3/ ;
Bollerslev (1986) for GARCH.

Design = the same rolling-origin protocol as task7_rolling_eval.py / task8_distributional.py:
  first training window --start rows, origin moves by --step rows, standardization with the training window only,
  models fitted on past data, ONE-STEP-AHEAD forecasts for the next --step days with the actual past data as input
  (fitted parameters are held fixed inside the block). Orders are re-selected every --select-every origins and
  parameters are re-estimated at every origin.

Outputs (data_out/task13/): point-forecast skill table, distribution table, chosen orders per stock and a summary of the
selected orders, plus a PNG. Needs: statsmodels, scipy, pandas, matplotlib; task7_rolling_eval.py and
task8_distributional.py in the same folder.

Usage:
  python task13_arima.py --max-stocks 10        # quick check (a few minutes)
  python task13_arima.py                        # full run (roughly 15-40 minutes; progress is printed)
  python task13_arima.py --no-arimax            # skip the regression model
"""
import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task7_rolling_eval import skill_table
from task8_distributional import fit_garch, garch_var, score_normal, summarize_blocks

warnings.filterwarnings("ignore")
OUT = os.path.join("data_out", "task13")


def arima_fit(z, order, exog=None):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():                      # statsmodels is very chatty about start parameters
        warnings.simplefilter("ignore")
        return ARIMA(z, order=(order[0], 0, order[1]), exog=exog, trend="n").fit()


def select_orders(Zs, maxpq):
    """AICc-optimal (p, q) for every stock on the training window Zs (rows x stocks)."""
    grid = list(itertools.product(range(maxpq + 1), range(maxpq + 1)))
    best = []
    for j in range(Zs.shape[1]):
        scores = []
        for od in grid:
            try:
                scores.append(arima_fit(Zs[:, j], od).aicc)
            except Exception:
                scores.append(np.inf)
        best.append(grid[int(np.argmin(scores))])
    return best


def one_step(res, z_full, o, step, exog_full=None):
    """One-step-ahead predictions for rows o..o+step-1 and in-sample predictions for all rows (fixed parameters)."""
    r2 = res.apply(z_full, exog=exog_full, refit=False)
    return np.asarray(r2.predict(start=0, end=len(z_full) - 1))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    R = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True).values
    if a.max_stocks:
        R = R[:, :a.max_stocks]
    n, d = R.shape
    origins = list(range(a.start, n - a.step + 1, a.step))
    print(f"{len(origins)} origins (start {a.start}, step {a.step}), d = {d}, order grid p,q in 0..{a.maxpq}", flush=True)

    point_rows, dist_rows, order_rows = [], [], []
    orders = None
    t_start = time.time()
    for k, o in enumerate(origins):
        mu, sd = R[:o].mean(0), R[:o].std(0)
        Z = (R - mu) / sd
        Zfull = Z[:o + a.step]
        X = Z[o:o + a.step]
        s2 = ((Z[1:o] - Z[:o - 1]) ** 2).mean(0)
        M = Zfull.mean(1)
        mlag = np.concatenate([[0.0], M[:-1]])[:, None]           # market return of the previous day

        if orders is None or k % a.select_every == 0:
            orders = select_orders(Z[:o], a.maxpq)
            for j, od in enumerate(orders):
                order_rows.append(dict(origin=o, stock=j, p=od[0], q=od[1]))
            print(f"  origin {o}: orders selected; share ARIMA(0,0) = {np.mean([od == (0, 0) for od in orders]):.2f}", flush=True)

        P_arima, P_x = np.zeros((a.step, d)), np.zeros((a.step, d))
        E_arima, E_x = np.zeros((o + a.step, d)), np.zeros((o + a.step, d))
        for j in range(d):
            try:
                res = arima_fit(Z[:o, j], orders[j])
                pred = one_step(res, Zfull[:, j], o, a.step)
            except Exception:
                pred = np.zeros(o + a.step)
            P_arima[:, j] = pred[o:]
            E_arima[:, j] = Zfull[:, j] - pred
            if not a.no_arimax:
                try:
                    resx = arima_fit(Z[:o, j], (1, 0), exog=mlag[:o])
                    predx = one_step(resx, Zfull[:, j], o, a.step, exog_full=mlag)
                except Exception:
                    predx = np.zeros(o + a.step)
                P_x[:, j] = predx[o:]
                E_x[:, j] = Zfull[:, j] - predx
        b_ar1 = (Z[:o - 1] * Z[1:o]).sum(0) / (Z[:o - 1] ** 2).sum(0)

        points = {"Zero predictor": np.zeros_like(X),
                  "AR(1) per stock": Z[o - 1:o + a.step - 1] * b_ar1,
                  "ARIMA(p,0,q), AICc order": P_arima}
        if not a.no_arimax:
            points["ARIMAX: AR(1) + lagged market"] = P_x
        for name, P in points.items():
            e2 = (X - P) ** 2
            point_rows.append(dict(origin=o, model=name, sse=float(e2.sum()), count=int(e2.size),
                                   mse=float(e2.mean()), rmsse=float(np.sqrt((e2 / s2).mean()))))

        # probabilistic: mean + variance models, Normal predictive distributions
        def garch_sigma(E):
            params = np.array([fit_garch(E[:o, j]) for j in range(d)])
            return np.sqrt(garch_var(E, params)[o:])
        dists = {"Zero mean + constant variance": (np.zeros_like(X), np.ones((a.step, d))),
                 "Zero mean + GARCH(1,1)": (np.zeros_like(X), garch_sigma(Zfull.copy())),
                 "ARIMA mean + constant variance": (P_arima, np.tile(E_arima[:o].std(0), (a.step, 1))),
                 "ARIMA mean + GARCH(1,1)": (P_arima, garch_sigma(E_arima))}
        if not a.no_arimax:
            dists["ARIMAX mean + GARCH(1,1)"] = (P_x, garch_sigma(E_x))
        for name, (mu_, sg) in dists.items():
            sc = score_normal(X, mu_, sg)
            dist_rows.append(dict(origin=o, model=name, crps=sc["crps"].mean(), nll=sc["nll"].mean(),
                                  in80=sc["in80"].mean(), in95=sc["in95"].mean()))
        print(f"  origin {o} done ({k + 1}/{len(origins)}), elapsed {time.time() - t_start:.0f}s", flush=True)

    pts = pd.DataFrame(point_rows)
    pts.to_csv(os.path.join(OUT, "task13_point_block_errors.csv"), index=False)
    tab_pt = skill_table(pts)
    tab_pt.to_csv(os.path.join(OUT, "task13_point_skill_table.csv"), index=False)

    dst = pd.DataFrame(dist_rows)
    dst.to_csv(os.path.join(OUT, "task13_distribution_block_scores.csv"), index=False)
    tab_ds = summarize_blocks(dst, base="Zero mean + constant variance")
    tab_ds.to_csv(os.path.join(OUT, "task13_distribution_table.csv"), index=False)

    od = pd.DataFrame(order_rows)
    od.to_csv(os.path.join(OUT, "task13_orders.csv"), index=False)
    od["order"] = "(" + od.p.astype(str) + "," + od.q.astype(str) + ")"
    counts = od.order.value_counts().sort_index()
    summ = pd.DataFrame([dict(selections=len(od), share_0_0=float((od.p + od.q == 0).mean()),
                              mean_p=float(od.p.mean()), mean_q=float(od.q.mean()),
                              share_p_plus_q_ge_3=float((od.p + od.q >= 3).mean()))])
    summ.to_csv(os.path.join(OUT, "task13_order_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(counts.index, counts.values, color="tab:blue")
    ax.set_xlabel("Selected ARIMA(p,0,q) order (AICc)"); ax.set_ylabel("Number of stock selections")
    ax.set_title("Orders chosen by AICc across stocks and selection points")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task13_order_counts.png"), dpi=150); plt.close(fig)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== POINT forecasts, rolling origins (skill > 0 beats the zero predictor) ===")
    print(tab_pt.round(4).to_string(index=False))
    print("\n=== PROBABILISTIC forecasts (lower CRPS/NLL better; coverage near 0.80 / 0.95); skill vs zero mean + constant variance ===")
    print(tab_ds.round(4).to_string(index=False))
    print("\n=== Selected orders ===")
    print(summ.round(3).to_string(index=False))
    print(counts.to_string())
    print(f"\nSaved tables and figure in {OUT}/")
    print("Notes: training windows overlap across blocks (tests and CIs approximate); parameters are held fixed inside each "
          "block; Gaussian predictive distributions; orders restricted to p,q <= maxpq; the market regressor is the "
          "previous-day equal-weight mean of all stocks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--maxpq", type=int, default=2)
    ap.add_argument("--select-every", type=int, default=4, help="re-select orders every k origins")
    ap.add_argument("--max-stocks", type=int, default=0, help="debug: first k stocks only")
    ap.add_argument("--no-arimax", action="store_true")
    main(ap.parse_args())
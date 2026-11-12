"""
Task 16 (M4): choose lambda and tau for the market + sector residual graph WITHOUT touching the validation block.

Why: lambda and tau were chosen earlier with the validation block, so the validation scores were slightly
optimistic for the graph. Here the choice is made by a split INSIDE the training block, and the official
validation block is used exactly once, for the final evaluation of the chosen setting.

Steps
  1. Split the training rows (time order): first 70% = inner-train, 2-row gap, rest = inner-validation.
  2. For each (lambda, tau) in the grid: fit the factor regression (market + sector) on inner-train, learn
     DYNOTEARS on the standardized residuals, score same-day conditional CRPS on inner-validation.
  3. Choose the pair with the lowest inner-validation CRPS (ties within 0.1% -> fewer edges).
  4. Refit on the FULL training block with the chosen pair and evaluate once on the official validation block,
     against the market + sector baseline (paired CRPS gain, bootstrap CI, Wilcoxon).
These are SAME-DAY conditional scores, not forecasts.

Usage (run from the Finance folder; needs task2_dynotears.py, task8_distributional.py,
       task11_sector_residual_graph.py, plot_basenet.py side by side):
  python task16_inner_selection.py                        # 9 fits + 1 final fit, roughly 15-25 minutes
  python task16_inner_selection.py --lams 0.1 0.3 --taus 0.05   # smaller grid
Outputs in data_out/task16/ and data_out/fit_sector_residual_inner.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from task2_dynotears import fit_dynotears
from task8_distributional import score_normal
from task11_sector_residual_graph import loo_features
from plot_basenet import SECTORS

OUT = os.path.join("data_out", "task16")
rng = np.random.default_rng(0)


def prepare(Xa, Ya, Xb, Yb, groups):
    """Fit the market+sector regression on (Xa, Ya) rows; return residual data for both blocks."""
    d = Xa.shape[1]
    Ma, Sa = loo_features(Xa, groups)
    Mb, Sb = loo_features(Xb, groups)
    Mya, Sya = loo_features(Ya, groups)
    Myb, Syb = loo_features(Yb, groups)
    coef = np.zeros((d, 2))
    for j in range(d):
        coef[j] = np.linalg.lstsq(np.column_stack([Ma[:, j], Sa[:, j]]), Xa[:, j], rcond=None)[0]
    fac = lambda M, S: M * coef[:, 0] + S * coef[:, 1]
    se = (Xa - fac(Ma, Sa)).std(0)
    out = dict(
        fa=fac(Ma, Sa), fb=fac(Mb, Sb),
        Za=(Xa - fac(Ma, Sa)) / se, Zb=(Xb - fac(Mb, Sb)) / se,
        Yza=(Ya - fac(Mya, Sya)) / se, Yzb=(Yb - fac(Myb, Syb)) / se,
        se=se, coef=coef)
    return out


def fit_and_score(Xa, Ya, Xb, Yb, groups, lam, tau):
    P = prepare(Xa, Ya, Xb, Yb, groups)
    t0 = time.time()
    W, A, info = fit_dynotears(P["Za"], P["Yza"], lam, lam, tau, tau, verbose=False)
    secs = time.time() - t0
    mu_a = P["fa"] + (P["Za"] @ W + P["Yza"] @ A) * P["se"]
    mu_b = P["fb"] + (P["Zb"] @ W + P["Yzb"] @ A) * P["se"]
    sig = (Xa - mu_a).std(0)
    sc = score_normal(Xb, mu_b, np.tile(sig, (len(Xb), 1)))
    sig0 = (Xa - P["fa"]).std(0)
    sc0 = score_normal(Xb, P["fb"], np.tile(sig0, (len(Xb), 1)))
    return dict(W=W, A=A, P=P, secs=secs, dag=info["is_dag"],
                crps=sc["crps"].mean(), nll=sc["nll"].mean(), day=sc["crps"].mean(1),
                crps_base=sc0["crps"].mean(), day_base=sc0["crps"].mean(1))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = [str(t) for t in D["tickers"]]
    d = len(tickers)
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    lab = np.array([sector_of.get(t, "Other") for t in tickers])
    groups = [list(np.where(lab == s)[0]) for s in dict.fromkeys(lab)]

    n = len(Xtr)
    cut = int(round(0.7 * n))
    Xa, Ya = Xtr[:cut], Ytr[:cut]
    Xb, Yb = Xtr[cut + 2:], Ytr[cut + 2:]
    print(f"Inner split of the training block: inner-train {len(Xa)} rows, gap 2, inner-validation {len(Xb)} rows",
          flush=True)

    rows = []
    for lam in a.lams:
        for tau in a.taus:
            r = fit_and_score(Xa, Ya, Xb, Yb, groups, lam, tau)
            e = int((r["W"] != 0).sum() + (r["A"] != 0).sum())
            rows.append(dict(lam=lam, tau=tau, inner_CRPS=r["crps"], inner_NLL=r["nll"],
                             inner_CRPS_market_sector=r["crps_base"],
                             gain_vs_market_sector=r["crps_base"] - r["crps"],
                             edges_W=int((r["W"] != 0).sum()), edges_A=int((r["A"] != 0).sum()),
                             is_dag=r["dag"], fit_seconds=r["secs"]))
            print(f"  lam={lam:<5} tau={tau:<5} inner CRPS={r['crps']:.4f} (market+sector {r['crps_base']:.4f}) "
                  f"edges={e} fit {r['secs']:.0f}s", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(OUT, "task16_inner_grid.csv"), index=False)

    # choice: lowest inner CRPS; among pairs within 0.1% of the best, the one with fewer edges
    best = grid.inner_CRPS.min()
    cand = grid[grid.inner_CRPS <= best * 1.001].copy()
    cand["edges"] = cand.edges_W + cand.edges_A
    ch = cand.sort_values(["edges", "inner_CRPS"]).iloc[0]
    lam, tau = float(ch.lam), float(ch.tau)
    print(f"\nChosen by inner validation: lambda = {lam}, tau = {tau}", flush=True)

    # final: refit on the full training block, evaluate ONCE on the official validation block
    r = fit_and_score(Xtr, Ytr, Xva, Yva, groups, lam, tau)
    diff = r["day_base"] - r["day"]                       # positive = graph helps
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    final = pd.DataFrame([dict(
        lam=lam, tau=tau, edges_W=int((r["W"] != 0).sum()), edges_A=int((r["A"] != 0).sum()),
        CRPS_market_sector=r["crps_base"], CRPS_residual_graph=r["crps"],
        mean_CRPS_gain=diff.mean(), CI95_low=np.percentile(boot, 2.5), CI95_high=np.percentile(boot, 97.5),
        graph_better_share_of_days=float((diff > 0).mean()),
        wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan,
        is_dag=r["dag"], fit_seconds=r["secs"])])
    final.to_csv(os.path.join(OUT, "task16_final_validation.csv"), index=False)
    np.savez(os.path.join("data_out", "fit_sector_residual_inner.npz"), W=r["W"], A=r["A"],
             coef=r["P"]["coef"], tickers=np.array(tickers), lam=lam, tau=tau)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== Inner-validation grid (training block only) ===")
    print(grid.round(4).to_string(index=False))
    print("\n=== Final evaluation on the official validation block, used once (same-day conditional, NOT forecasts) ===")
    print(final.round(5).to_string(index=False))
    print(f"\nSaved to {OUT}/ and data_out/fit_sector_residual_inner.npz")
    print("Notes: one inner split (70/30), so the choice is itself noisy; the grid is coarse; the market+sector "
          "regression was refitted inside each split; CIs and tests are approximate (day-level resampling).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    main(ap.parse_args())
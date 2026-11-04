"""
Task 11 (M3, experiment 4c): what does the graph add BEYOND the market + sector factor model?

Task 10 showed that market + sector (279 parameters, 46.1% of validation variance) beats the factor + residual
graph (357 parameters, 43.9%). The fair question for the PGM is therefore:
    after the market and sector factors are removed, does a sparse directed graph on the residuals still
    improve same-day conditional prediction?

Steps
  1. Per stock j, regress x_j on [leave-one-out market return, leave-one-out sector mean]  (training block).
  2. Standardize the residuals and re-learn DYNOTEARS on them (Pamfil et al., 2020, AISTATS), same lambda and tau.
  3. Combine: x_j ~ factors + (graph prediction of the residual), Normal with training residual SD.
  4. Compare on the validation block with the market + sector baseline (CRPS, NLL, RMSE, coverage, variance
     explained, parameter count, paired Wilcoxon test and bootstrap CI over days; scoring as in FPP3
     Sections 5.5 and 5.9, Hyndman & Athanasopoulos, 2021). These are SAME-DAY conditional distributions, not forecasts.
  5. List the strongest cross-sector edges that remain (candidates for interpretation).

Usage (from the folder containing data_out/; needs task2_dynotears.py, task8_distributional.py, plot_basenet.py):
  python task11_sector_residual_graph.py                # about 5 minutes
  python task11_sector_residual_graph.py --lam 0.03     # denser graph, slower
Outputs in data_out/task11/ and data_out/fit_sector_residual.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from task2_dynotears import fit_dynotears
from task8_distributional import score_normal

OUT = os.path.join("data_out", "task11")
TAU = 0.05
rng = np.random.default_rng(0)

try:
    from plot_basenet import SECTORS
except Exception as e:
    raise SystemExit(f"Cannot import SECTORS from plot_basenet.py ({e}). Copy plot_basenet.py into this folder.")


def loo_features(X, groups):
    """Per-stock features [leave-one-out market, leave-one-out sector mean]; single-member sectors get 0."""
    n, d = X.shape
    M = (X.sum(1, keepdims=True) - X) / (d - 1)
    S = np.zeros_like(X)
    for g in groups:
        if len(g) > 1:
            tot = X[:, g].sum(1, keepdims=True)
            S[:, g] = (tot - X[:, g]) / (len(g) - 1)
    return M, S


def main(a):
    os.makedirs(OUT, exist_ok=True)
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = [str(t) for t in D["tickers"]]
    d = len(tickers)
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    lab = np.array([sector_of.get(t, "Other") for t in tickers])
    groups = [list(np.where(lab == s)[0]) for s in dict.fromkeys(lab)]
    if len(groups) == 1:
        raise SystemExit("All stocks fell into one sector: tickers do not match plot_basenet.py SECTORS.")

    # 1. factor regression (market + sector), fitted on training rows only
    Mtr, Str = loo_features(Xtr, groups)
    Mva, Sva = loo_features(Xva, groups)
    My_tr, Sy_tr = loo_features(Ytr, groups)
    My_va, Sy_va = loo_features(Yva, groups)
    coef = np.zeros((d, 2))
    for j in range(d):
        coef[j] = np.linalg.lstsq(np.column_stack([Mtr[:, j], Str[:, j]]), Xtr[:, j], rcond=None)[0]
    fac = lambda M, S: M * coef[:, 0] + S * coef[:, 1]
    Etr = Xtr - fac(Mtr, Str)
    se = Etr.std(0)

    # 2. DYNOTEARS on standardized residuals (current day and lagged day use the same factor coefficients)
    Ztr = Etr / se
    Zva = (Xva - fac(Mva, Sva)) / se
    Yz_tr = (Ytr - fac(My_tr, Sy_tr)) / se
    Yz_va = (Yva - fac(My_va, Sy_va)) / se
    print(f"Train {Xtr.shape}, validation {Xva.shape}; fitting DYNOTEARS on market+sector residuals "
          f"(lambda = {a.lam}, tau = {TAU}); about 5 minutes ...", flush=True)
    t0 = time.time()
    W, A, info = fit_dynotears(Ztr, Yz_tr, a.lam, a.lam, TAU, TAU, verbose=True)
    secs = time.time() - t0
    np.savez(os.path.join("data_out", "fit_sector_residual.npz"), W=W, A=A, coef=coef, tickers=np.array(tickers))
    print(f"Fit took {secs:.0f}s; edges W = {int((W != 0).sum())}, A = {int((A != 0).sum())}, DAG = {info['is_dag']}")

    # 3-4. distributions on the validation block
    models = {}
    models["Constant variance"] = (np.zeros_like(Xtr), np.zeros_like(Xva), 0)
    models["Market + sector factor"] = (fac(Mtr, Str), fac(Mva, Sva), 2 * d)
    mu_tr = fac(Mtr, Str) + (Ztr @ W + Yz_tr @ A) * se
    mu_va = fac(Mva, Sva) + (Zva @ W + Yz_va @ A) * se
    models["Market + sector + residual graph"] = (mu_tr, mu_va, 2 * d + int((W != 0).sum() + (A != 0).sum()))

    rows, day = [], {}
    base_rmse = None
    for name, (mtr, mva, npar) in models.items():
        sig = (Xtr - mtr).std(0)
        sc = score_normal(Xva, mva, np.tile(sig, (len(Xva), 1)))
        rmse = float(np.sqrt(np.mean((Xva - mva) ** 2)))
        base_rmse = rmse if base_rmse is None else base_rmse
        day[name] = sc["crps"].mean(1)
        rows.append(dict(model=name, n_params=npar + d, CRPS=sc["crps"].mean(), NLL=sc["nll"].mean(), RMSE=rmse,
                         var_explained=1 - (rmse / base_rmse) ** 2, coverage80=sc["in80"].mean(), coverage95=sc["in95"].mean()))
    tab = pd.DataFrame(rows)
    tab["CRPS_skill_vs_const"] = 1 - tab.CRPS / tab.CRPS.iloc[0]
    tab.to_csv(os.path.join(OUT, "task11_scores.csv"), index=False)

    diff = day["Market + sector factor"] - day["Market + sector + residual graph"]   # positive = graph helps
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    test = pd.DataFrame([dict(comparison="residual graph vs market + sector alone", mean_CRPS_gain=diff.mean(),
                              CI95_low=np.percentile(boot, 2.5), CI95_high=np.percentile(boot, 97.5),
                              graph_better_on_share_of_days=float((diff > 0).mean()),
                              wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan)])
    test.to_csv(os.path.join(OUT, "task11_paired_test.csv"), index=False)

    # 5. graph statistics and edge list
    i, j = np.nonzero(W)
    chance = sum((lab == s).sum() * ((lab == s).sum() - 1) for s in set(lab)) / (d * (d - 1))
    stats = dict(edges_W=len(i), edges_A=int((A != 0).sum()),
                 positive_share=float((W[i, j] > 0).mean()) if len(i) else np.nan,
                 within_sector_edge_share=float((lab[i] == lab[j]).mean()) if len(i) else np.nan,
                 within_sector_by_chance=float(chance), fit_seconds=secs)
    pd.DataFrame([stats]).to_csv(os.path.join(OUT, "task11_graph_stats.csv"), index=False)
    edges = pd.DataFrame(dict(source=[tickers[k] for k in i], target=[tickers[k] for k in j], weight=W[i, j],
                              source_sector=lab[i], target_sector=lab[j], same_sector=lab[i] == lab[j]))
    edges = edges.reindex(edges.weight.abs().sort_values(ascending=False).index)
    edges.to_csv(os.path.join(OUT, "task11_edges.csv"), index=False)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== Same-day conditional distributions, validation block (NOT forecasts) ===")
    print(tab.round(4).to_string(index=False))
    print("\n=== Paired test: does the residual graph add anything to market + sector? (daily CRPS) ===")
    print(test.round(5).to_string(index=False))
    print("\n=== Residual graph statistics ===")
    print(pd.DataFrame([stats]).round(3).to_string(index=False))
    cross = edges[~edges.same_sector].head(15)
    if len(cross):
        print("\n=== Strongest cross-sector edges (row = source stock, same-day; not causal claims) ===")
        print(cross.round(3).to_string(index=False))
    print(f"\nSaved tables in {OUT}/ and the fit in data_out/fit_sector_residual.npz")
    print("Notes: same-day conditional scores, not forecasts; sector labels are approximate; lambda and tau were chosen "
          "with the validation block in earlier experiments, which slightly favours the graph.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.1)
    main(ap.parse_args())
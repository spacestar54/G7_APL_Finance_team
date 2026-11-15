"""
Task 10 (M3, experiment 4b): does the sparse DYNOTEARS graph beat simple same-day structure?

Task 9 found: one-factor model 34.3% of validation variance, original graph 35.2%, factor + residual graph 43.9%.
Before claiming the graph adds value, compare it with other simple models of SAME-DAY structure (not forecasts):

  Constant variance
  One-factor (leave-one-out market)
  Market + sector factor       x_j ~ market_(-j) + mean of the other stocks in j's sector
  PCA factors (K = 1, 3, 5, 10)  x_j ~ K principal components computed from the OTHER stocks (leave-one-out, no leakage)
  Dense ridge                  x_j ~ all other stocks (ridge penalty chosen on the last 25% of the training block)
  Original DYNOTEARS graph     (data_out/fit_dynotears.npz)
  Factor + residual graph      (data_out/fit_factor_residual.npz, created by task9_factor_ablation.py --run factor)

Every model gives x_j a Normal(mu, sigma_j^2) distribution; sigma_j is the training residual SD.
Scores: CRPS, NLL, RMSE, coverage, variance explained (CSE 516 KPIs; FPP3 Sections 5.5 and 5.9, Hyndman &
Athanasopoulos, 2021) plus a parameter count (model complexity, guidelines Section 6.1).
DYNOTEARS: Pamfil et al. (2020), AISTATS.

Usage (from the folder that contains data_out/; needs task8_distributional.py and plot_basenet.py there):
  python task10_factor_baselines.py
Outputs in data_out/task10/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task8_distributional import score_normal

OUT = os.path.join("data_out", "task10")
TAU = 0.05
rng = np.random.default_rng(0)

try:
    from plot_basenet import SECTORS
except Exception as e:                       # do not fail silently: sector labels are required here
    raise SystemExit(f"Cannot import SECTORS from plot_basenet.py ({e}). "
                     f"Copy plot_basenet.py into the folder you run this script from.")


def loo_market(X):
    return (X.sum(1, keepdims=True) - X) / (X.shape[1] - 1)


def ols_columns(Ftr, Fva, Xtr, Xva):
    """Per-stock OLS (no intercept) with per-stock feature matrices Ftr[j], Fva[j]. Returns mu_tr, mu_va, nparams."""
    d = Xtr.shape[1]
    mu_tr, mu_va, npar = np.zeros_like(Xtr), np.zeros_like(Xva), 0
    for j in range(d):
        b = np.linalg.lstsq(Ftr[j], Xtr[:, j], rcond=None)[0]
        mu_tr[:, j], mu_va[:, j] = Ftr[j] @ b, Fva[j] @ b
        npar += len(b)
    return mu_tr, mu_va, npar


def main():
    os.makedirs(OUT, exist_ok=True)
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = [str(t) for t in D["tickers"]]
    n, d = Xtr.shape
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    groups = {}
    for k, t in enumerate(tickers):
        groups.setdefault(sector_of.get(t, "Other"), []).append(k)
    print(f"Train {Xtr.shape}, validation {Xva.shape}; sectors: " + ", ".join(f"{s}={len(v)}" for s, v in groups.items()))
    if len(groups) == 1:
        raise SystemExit("All stocks fell into one sector: tickers do not match plot_basenet.py SECTORS.")

    preds = {}                                       # name -> (mu_tr, mu_va, n_params)

    # constant variance
    preds["Constant variance"] = (np.zeros_like(Xtr), np.zeros_like(Xva), 0)

    # one-factor
    Mtr, Mva = loo_market(Xtr), loo_market(Xva)
    beta = (Mtr * Xtr).sum(0) / (Mtr ** 2).sum(0)
    preds["One-factor (market)"] = (Mtr * beta, Mva * beta, d)

    # market + sector factor (leave-one-out within the sector; single-member sectors get a zero column)
    Str, Sva = np.zeros_like(Xtr), np.zeros_like(Xva)
    for g in groups.values():
        for j in g:
            if len(g) > 1:
                Str[:, j] = (Xtr[:, g].sum(1) - Xtr[:, j]) / (len(g) - 1)
                Sva[:, j] = (Xva[:, g].sum(1) - Xva[:, j]) / (len(g) - 1)
    Ftr = [np.column_stack([Mtr[:, j], Str[:, j]]) for j in range(d)]
    Fva = [np.column_stack([Mva[:, j], Sva[:, j]]) for j in range(d)]
    preds["Market + sector factor"] = ols_columns(Ftr, Fva, Xtr, Xva)

    # PCA factors from the other stocks (leave-one-out)
    Ks = (1, 3, 5, 10)
    svd = {}
    for j in range(d):
        others = [k for k in range(d) if k != j]
        _, _, Vt = np.linalg.svd(Xtr[:, others], full_matrices=False)
        svd[j] = (others, Vt[:max(Ks)].T)
    for K in Ks:
        Ftr = [Xtr[:, svd[j][0]] @ svd[j][1][:, :K] for j in range(d)]
        Fva = [Xva[:, svd[j][0]] @ svd[j][1][:, :K] for j in range(d)]
        preds[f"PCA factors (K={K})"] = ols_columns(Ftr, Fva, Xtr, Xva)

    # dense ridge on all other stocks
    cut = int(0.75 * n)
    mu_tr, mu_va, npar = np.zeros_like(Xtr), np.zeros_like(Xva), 0
    for j in range(d):
        o = [k for k in range(d) if k != j]
        A, y = Xtr[:, o], Xtr[:, j]
        errs = []
        for a in (1, 10, 100, 1e3, 1e4):
            b = np.linalg.solve(A[:cut].T @ A[:cut] + a * np.eye(d - 1), A[:cut].T @ y[:cut])
            errs.append(np.mean((y[cut:] - A[cut:] @ b) ** 2))
        alpha = (1, 10, 100, 1e3, 1e4)[int(np.argmin(errs))]
        b = np.linalg.solve(A.T @ A + alpha * np.eye(d - 1), A.T @ y)
        mu_tr[:, j], mu_va[:, j] = A @ b, Xva[:, o] @ b
        npar += d - 1
    preds["Dense ridge (all other stocks)"] = (mu_tr, mu_va, npar)

    # DYNOTEARS graphs
    F0 = np.load(os.path.join("data_out", "fit_dynotears.npz"), allow_pickle=True)
    W0 = np.where(np.abs(F0["W"]) >= TAU, F0["W"], 0.0)
    A0 = np.where(np.abs(F0["A"]) >= TAU, F0["A"], 0.0)
    preds["Original DYNOTEARS graph"] = (Xtr @ W0 + Ytr @ A0, Xva @ W0 + Yva @ A0, int((W0 != 0).sum() + (A0 != 0).sum()))

    fp = os.path.join("data_out", "fit_factor_residual.npz")
    if os.path.exists(fp):
        F2 = np.load(fp, allow_pickle=True)
        W2 = np.where(np.abs(F2["W"]) >= TAU, F2["W"], 0.0)
        A2 = np.where(np.abs(F2["A"]) >= TAU, F2["A"], 0.0)
        Etr = Xtr - Mtr * beta
        se = Etr.std(0)
        Ztr, Zva = Etr / se, (Xva - Mva * beta) / se
        Yz_tr = (Ytr - loo_market(Ytr) * beta) / se
        Yz_va = (Yva - loo_market(Yva) * beta) / se
        preds["Factor + residual graph (DYNOTEARS)"] = (
            Mtr * beta + (Ztr @ W2 + Yz_tr @ A2) * se, Mva * beta + (Zva @ W2 + Yz_va @ A2) * se,
            d + int((W2 != 0).sum() + (A2 != 0).sum()))
    else:
        print("Missing data_out/fit_factor_residual.npz: run  python task9_factor_ablation.py --run factor  first.")

    # scoring
    rows, day = [], {}
    base_rmse = None
    for name, (mtr, mva, npar) in preds.items():
        sig = (Xtr - mtr).std(0)
        sc = score_normal(Xva, mva, np.tile(sig, (len(Xva), 1)))
        rmse = float(np.sqrt(np.mean((Xva - mva) ** 2)))
        base_rmse = rmse if base_rmse is None else base_rmse
        day[name] = sc["crps"].mean(1)
        rows.append(dict(model=name, n_params=npar + d, CRPS=sc["crps"].mean(), NLL=sc["nll"].mean(), RMSE=rmse,
                         var_explained=1 - (rmse / base_rmse) ** 2, coverage80=sc["in80"].mean(), coverage95=sc["in95"].mean()))
    tab = pd.DataFrame(rows)
    tab["CRPS_skill_vs_const"] = 1 - tab.CRPS / tab.loc[tab.model == "Constant variance", "CRPS"].iloc[0]
    tab = tab.sort_values("CRPS").reset_index(drop=True)
    tab.to_csv(os.path.join(OUT, "task10_sameday_baselines.csv"), index=False)

    # paired tests against the factor + residual graph (if present)
    key = "Factor + residual graph (DYNOTEARS)"
    if key in day:
        rows = []
        for name, v in day.items():
            if name == key:
                continue
            diff = v - day[key]                          # positive = graph model has lower CRPS (better)
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
            rows.append(dict(baseline=name, mean_CRPS_gain_of_graph_model=diff.mean(),
                             CI95_low=np.percentile(boot, 2.5), CI95_high=np.percentile(boot, 97.5),
                             graph_better_on_share_of_days=float((diff > 0).mean()), wilcoxon_p=p))
        tests = pd.DataFrame(rows).sort_values("mean_CRPS_gain_of_graph_model")
        tests.to_csv(os.path.join(OUT, "task10_paired_tests.csv"), index=False)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== SAME-DAY conditional distributions, validation block (NOT forecasts); sorted by CRPS ===")
    print(tab.round(4).to_string(index=False))
    if key in day:
        print("\n=== Paired tests: is the factor + residual graph better than each baseline? (daily CRPS) ===")
        print(tests.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(tab.n_params, tab.var_explained, color="tab:blue")
    for _, r in tab.iterrows():
        ax.annotate(r.model, (r.n_params, r.var_explained), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=100); ax.set_xlabel("Number of parameters (weights + residual SDs)")
    ax.set_ylabel("Validation variance explained"); ax.grid(alpha=.3)
    ax.set_title("Accuracy vs model size, same-day conditional models")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task10_accuracy_vs_size.png"), dpi=150); plt.close(fig)
    print(f"\nSaved tables and figure in {OUT}/")
    print("Notes: same-day conditional scores, not forecasts. Sector labels are approximate (plot_basenet.py). "
          "n_params counts regression/edge weights plus d residual SDs; PCA loadings are not counted. "
          "tau = 0.05 and lambda = 0.1 were chosen with the validation block, so graph results are slightly optimistic.")


if __name__ == "__main__":
    main()
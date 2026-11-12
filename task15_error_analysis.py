"""
Task 15 (M3 Section II.6 / M4): error analysis of the same-day conditional models on the validation block.

Answers, with the numbers from your own run:
  1. Which stocks are well / poorly explained by the market + sector + residual graph model, and does it depend on
     the (approximate) sector or on the stock's volatility?
  2. Which stocks have residual autocorrelation that survives multiple-testing correction (Ljung-Box, lag 10, BH-FDR 5%)
     for the original DYNOTEARS graph (the '3 stocks' of the Ljung-Box table)?
  3. When are the errors large? Monthly error and variance explained over the validation period, share of total squared
     error in the 10 worst days, rolling 20-day RMSE of the constant model vs the graph model (with dates).
  4. What are the weights of the lagged (inter-slice) edges that survive in the residual graph?

Same-day conditional prediction (NOT forecasts). Ljung-Box and residual diagnostics: Hyndman & Athanasopoulos (2021),
FPP3 Section 5.4. DYNOTEARS: Pamfil et al. (2020), AISTATS.

Usage (folder containing data_out/; needs task11_sector_residual_graph.py and plot_basenet.py there; needs
data_out/dataset.npz, fit_dynotears.npz, fit_sector_residual.npz and log_returns.csv):
  python task15_error_analysis.py
Outputs in data_out/task15/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task11_sector_residual_graph import loo_features
from plot_basenet import SECTORS

OUT = os.path.join("data_out", "task15")
TAU = 0.05
VAL_LEN, GAP = 400, 2


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = [str(t) for t in D["tickers"]]
    d = len(tickers)
    dates = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True).index
    val_dates = dates[len(dates) - VAL_LEN + 1:]                    # X_val = block[1:], so one date less than 400
    assert len(val_dates) == len(Xva), f"date alignment: {len(val_dates)} dates vs {len(Xva)} validation rows"
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    lab = np.array([sector_of.get(t, "Other") for t in tickers])
    groups = [list(np.where(lab == s)[0]) for s in dict.fromkeys(lab)]
    if len(groups) == 1:
        raise SystemExit("All stocks in one sector: tickers do not match plot_basenet.py SECTORS.")

    # ---- reproduce the models' validation predictions --------------------------------------------------
    Mtr, Str = loo_features(Xtr, groups)
    Mva, Sva = loo_features(Xva, groups)
    coef = np.zeros((d, 2))
    for j in range(d):
        coef[j] = np.linalg.lstsq(np.column_stack([Mtr[:, j], Str[:, j]]), Xtr[:, j], rcond=None)[0]
    fac = lambda M, S: M * coef[:, 0] + S * coef[:, 1]
    mu_ms = fac(Mva, Sva)

    F2 = np.load(os.path.join("data_out", "fit_sector_residual.npz"), allow_pickle=True)
    W2, A2 = F2["W"], F2["A"]
    My_tr, Sy_tr = loo_features(Ytr, groups)
    My_va, Sy_va = loo_features(Yva, groups)
    se = (Xtr - fac(Mtr, Str)).std(0)
    Zva = (Xva - fac(Mva, Sva)) / se
    Yz_va = (Yva - fac(My_va, Sy_va)) / se
    mu_hy = mu_ms + (Zva @ W2 + Yz_va @ A2) * se

    # ---- 1. per-stock errors ----------------------------------------------------------------------------
    sse0, sse_ms, sse_hy = (Xva ** 2).sum(0), ((Xva - mu_ms) ** 2).sum(0), ((Xva - mu_hy) ** 2).sum(0)
    per = pd.DataFrame(dict(stock=tickers, sector=lab, rmse_zero=np.sqrt(sse0 / len(Xva)), rmse_market_sector=np.sqrt(sse_ms / len(Xva)),
                            rmse_hybrid=np.sqrt(sse_hy / len(Xva)), var_explained_market_sector=1 - sse_ms / sse0,
                            var_explained_hybrid=1 - sse_hy / sse0, train_volatility_rank=pd.Series(Xtr.std(0)).rank().values,
                            val_vs_train_sd=Xva.std(0) / Xtr.std(0)))
    per["graph_gain_in_var_explained"] = per.var_explained_hybrid - per.var_explained_market_sector
    per = per.sort_values("var_explained_hybrid", ascending=False)
    per.to_csv(os.path.join(OUT, "task15_per_stock.csv"), index=False)
    by_sector = per.groupby("sector")[["var_explained_market_sector", "var_explained_hybrid", "graph_gain_in_var_explained"]].mean().round(3)
    by_sector["n_stocks"] = per.groupby("sector").size()
    by_sector.to_csv(os.path.join(OUT, "task15_by_sector.csv"))
    corr = per[["var_explained_hybrid", "val_vs_train_sd", "graph_gain_in_var_explained"]].corr(method="spearman").round(2)

    # ---- 2. Ljung-Box failures for the original graph ---------------------------------------------------
    lb = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.stats.multitest import multipletests
        F0 = np.load(os.path.join("data_out", "fit_dynotears.npz"), allow_pickle=True)
        W0 = np.where(np.abs(F0["W"]) >= TAU, F0["W"], 0.0)
        A0 = np.where(np.abs(F0["A"]) >= TAU, F0["A"], 0.0)
        res = Xtr - Xtr @ W0 - Ytr @ A0
        p = np.array([acorr_ljungbox(res[:, j], lags=[10], return_df=True)["lb_pvalue"].iloc[0] for j in range(d)])
        rej = multipletests(p, alpha=0.05, method="fdr_bh")[0]
        lb = pd.DataFrame(dict(stock=tickers, sector=lab, ljungbox_p=p, rejected_after_BH=rej)).sort_values("ljungbox_p")
        lb.to_csv(os.path.join(OUT, "task15_ljungbox_original_graph.csv"), index=False)
    except Exception as e:
        print("Ljung-Box part skipped:", e)

    # ---- 3. errors over time ----------------------------------------------------------------------------
    e0, e_hy = (Xva ** 2).mean(1), ((Xva - mu_hy) ** 2).mean(1)
    daily = pd.DataFrame(dict(date=val_dates, mse_zero=e0, mse_hybrid=e_hy)).set_index("date")
    monthly = daily.resample("MS").mean()
    monthly["var_explained"] = 1 - monthly.mse_hybrid / monthly.mse_zero
    monthly["rmse_zero"], monthly["rmse_hybrid"] = np.sqrt(monthly.mse_zero), np.sqrt(monthly.mse_hybrid)
    monthly["days"] = daily.resample("MS").size()
    monthly = monthly[["days", "rmse_zero", "rmse_hybrid", "var_explained"]]
    monthly.to_csv(os.path.join(OUT, "task15_monthly.csv"))
    worst = daily.assign(excess=daily.mse_hybrid).sort_values("mse_hybrid", ascending=False).head(10)
    worst.to_csv(os.path.join(OUT, "task15_worst_days.csv"))
    top10_share = float(daily.mse_hybrid.nlargest(10).sum() / daily.mse_hybrid.sum())

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=False)
    r0, rh = np.sqrt(daily.mse_zero.rolling(20).mean()), np.sqrt(daily.mse_hybrid.rolling(20).mean())
    ax[0].plot(r0.index, r0, label="Zero model"); ax[0].plot(rh.index, rh, label="Market + sector + residual graph")
    ax[0].set_ylabel("Rolling 20-day RMSE"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Same-day conditional error over the validation period")
    top = per.head(15).iloc[::-1]
    ax[1].barh(top.stock, top.var_explained_hybrid, color="tab:blue")
    ax[1].set_xlabel("Validation variance explained (best 15 stocks)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task15_error_over_time_and_stocks.png"), dpi=150); plt.close(fig)

    # ---- 4. lagged edges --------------------------------------------------------------------------------
    i, j = np.nonzero(A2)
    lag_edges = pd.DataFrame(dict(source_yesterday=[tickers[k] for k in i], target_today=[tickers[k] for k in j], weight=A2[i, j],
                                  source_sector=lab[i], target_sector=lab[j])).sort_values("weight", key=np.abs, ascending=False)
    lag_edges.to_csv(os.path.join(OUT, "task15_lagged_edges.csv"), index=False)

    # ---- print ------------------------------------------------------------------------------------------
    print("\n=== 1a. Best 10 and worst 10 stocks (validation variance explained by the hybrid graph model) ===")
    cols = ["stock", "sector", "var_explained_market_sector", "var_explained_hybrid", "graph_gain_in_var_explained", "val_vs_train_sd"]
    print(per[cols].head(10).round(3).to_string(index=False))
    print(per[cols].tail(10).round(3).to_string(index=False))
    print("\n=== 1b. By sector (means) ===")
    print(by_sector.to_string())
    print("\n=== 1c. Spearman correlations across stocks ===")
    print(corr.to_string())
    if lb is not None:
        print("\n=== 2. Ljung-Box (lag 10) residuals of the ORIGINAL graph, training block: stocks rejected after BH correction ===")
        print(lb[lb.rejected_after_BH].round(5).to_string(index=False) if lb.rejected_after_BH.any() else "none")
        print(f"(uncorrected p < 0.05: {int((lb.ljungbox_p < 0.05).sum())} of {d})")
    print("\n=== 3a. Monthly error and variance explained, validation period ===")
    print(monthly.round(3).to_string())
    print(f"\n=== 3b. Share of the hybrid model's total squared error in its 10 worst days: {top10_share:.1%} "
          f"(10 of {len(daily)} days = {10 / len(daily):.1%}) ===")
    print(worst[["mse_zero", "mse_hybrid"]].round(3).to_string())
    print("\n=== 4. Lagged edges of the residual graph (yesterday's stock -> today's stock) ===")
    print(lag_edges.round(4).to_string(index=False) if len(lag_edges) else "none")
    print(f"\nSaved tables and figure in {OUT}/")
    print("Notes: same-day conditional errors, not forecasts; sector labels are approximate; one validation block of "
          f"{len(daily)} days; descriptive, not causal.")


if __name__ == "__main__":
    main()
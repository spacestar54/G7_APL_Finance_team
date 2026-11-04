"""
Task 9 (M3, experiment 4): why is every edge positive? Market-factor removal + ablations.

Question: how much of the learned graph is just the common market factor, and does a graph learned on the
factor-free residuals add anything to a one-factor model?

  FACTOR   One-factor model x_j = beta_j * m_(-j) + e_j (m_(-j) = leave-one-out equal-weight market return,
           beta fitted on the training block). DYNOTEARS is re-learned on the standardized residuals e.
           Compared on the validation block (same-day conditional distributions, NOT forecasts) with:
             constant variance, one-factor only, original DYNOTEARS graph, and factor + residual graph.
           Also reports graph statistics: edges, share of positive edges, within-sector edge share vs chance,
           Louvain NMI against the (approximate) sector labels.
  LAMBDA   Sensitivity of the static graph to lambda_W in {0.03, 0.1, 0.3}  (0.1 reused from the earlier fit).
  LAG      Lag order p = 2 compared with p = 1 (the saved fit).

DYNOTEARS: Pamfil et al. (2020), AISTATS. Scoring: CRPS, NLL, coverage as in task8_distributional.py
(Hyndman & Athanasopoulos, 2021, FPP3, Sections 5.5 and 5.9).

Usage (from the folder containing data_out/; needs task2_dynotears.py, task8_distributional.py, plot_basenet.py):
  python task9_factor_ablation.py --run factor          # about 5-8 minutes, do this first
  python task9_factor_ablation.py --run lambda lag      # slower (several fits)
  python task9_factor_ablation.py                       # everything
Outputs in data_out/task9/.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from task2_dynotears import fit_dynotears
from task8_distributional import score_normal

OUT = os.path.join("data_out", "task9")
TAU = 0.05

try:                                   # approximate sector labels from plot_basenet.py
    from plot_basenet import SECTORS
    SECTOR_OF = {t: s for s, ms in SECTORS.items() for t in ms}
except Exception:
    SECTOR_OF = {}


# ----------------------------------------------------------------------------
def load():
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    F = np.load(os.path.join("data_out", "fit_dynotears.npz"), allow_pickle=True)
    return D["X_train"], D["Y_train"], D["X_val"], D["Y_val"], [str(t) for t in D["tickers"]], F["W"], F["A"]


def graph_stats(W, tickers, tau=TAU):
    W = np.where(np.abs(W) >= tau, W, 0.0)
    i, j = np.nonzero(W)
    lab = np.array([SECTOR_OF.get(t, "Other") for t in tickers])
    d = len(tickers)
    chance = sum((lab == s).sum() * ((lab == s).sum() - 1) for s in set(lab)) / (d * (d - 1))
    out = dict(edges=len(i), positive_share=float((W[i, j] > 0).mean()) if len(i) else np.nan,
               median_abs_w=float(np.median(np.abs(W[i, j]))) if len(i) else np.nan,
               within_sector_edge_share=float((lab[i] == lab[j]).mean()) if len(i) else np.nan,
               within_sector_by_chance=float(chance), nmi_vs_sector=np.nan)
    if len(i) == 0:                    # an empty graph has no communities: NMI is not meaningful
        return out
    try:                               # Louvain communities vs sector labels (symmetrized |w|)
        import networkx as nx
        from sklearn.metrics import normalized_mutual_info_score as nmi
        U = nx.Graph()
        U.add_nodes_from(range(d))
        for a, b in zip(i, j):
            w = abs(W[a, b]) + abs(W[b, a])
            U.add_edge(a, b, weight=w)
        comms = nx.community.louvain_communities(U, weight="weight", seed=0)
        pred = np.zeros(d, dtype=int)
        for k, c in enumerate(comms):
            for v in c:
                pred[v] = k
        out["nmi_vs_sector"] = float(nmi(lab, pred))
    except Exception:
        pass
    return out


def loo_market(X):
    return (X.sum(1, keepdims=True) - X) / (X.shape[1] - 1)


def rebuild_block(X, Y):
    """Recover the original contiguous block from X = block[1:], Y = block[:-1]."""
    return np.vstack([Y[:1], X])


# ----------------------------------------------------------------------------
def run_factor(Xtr, Ytr, Xva, Yva, tickers, W0, A0):
    W0 = np.where(np.abs(W0) >= TAU, W0, 0.0)
    A0 = np.where(np.abs(A0) >= TAU, A0, 0.0)
    Mtr, Mva = loo_market(Xtr), loo_market(Xva)
    beta = (Mtr * Xtr).sum(0) / (Mtr ** 2).sum(0)
    Etr = Xtr - Mtr * beta
    se = Etr.std(0)
    Ztr = Etr / se
    Yz_tr = (Ytr - loo_market(Ytr) * beta) / se
    Yz_va = (Yva - loo_market(Yva) * beta) / se
    Zva = (Xva - Mva * beta) / se

    print("[FACTOR] re-learning DYNOTEARS on factor-free residuals (about 5 minutes) ...", flush=True)
    t0 = time.time()
    W2, A2, info = fit_dynotears(Ztr, Yz_tr, 0.1, 0.1, TAU, TAU, verbose=True)
    print(f"[FACTOR] fit took {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join("data_out", "fit_factor_residual.npz"), W=W2, A=A2, beta=beta, tickers=np.array(tickers))

    # graph statistics
    gs = pd.DataFrame([dict(graph="Original DYNOTEARS (raw returns)", edges_A=int((A0 != 0).sum()), **graph_stats(W0, tickers)),
                       dict(graph="DYNOTEARS on factor-free residuals", edges_A=int((A2 != 0).sum()), **graph_stats(W2, tickers))])
    gs.to_csv(os.path.join(OUT, "task9_factor_graph_stats.csv"), index=False)

    # same-day conditional scores on the validation block
    models = {}
    models["Constant variance"] = (np.zeros_like(Xva), np.tile(Xtr.std(0), (len(Xva), 1)))
    models["One-factor only"] = (Mva * beta, np.tile(Etr.std(0), (len(Xva), 1)))
    s0 = (Xtr - Xtr @ W0 - Ytr @ A0).std(0)
    models["Original DYNOTEARS graph"] = (Xva @ W0 + Yva @ A0, np.tile(s0, (len(Xva), 1)))
    mu_tr = Mtr * beta + (Ztr @ W2 + Yz_tr @ A2) * se
    s2 = (Xtr - mu_tr).std(0)
    models["Factor + residual graph"] = (Mva * beta + (Zva @ W2 + Yz_va @ A2) * se, np.tile(s2, (len(Xva), 1)))

    rows, day = [], {}
    for name, (mu, sig) in models.items():
        sc = score_normal(Xva, mu, sig)
        day[name] = sc["crps"].mean(1)
        rows.append(dict(model=name, CRPS=sc["crps"].mean(), NLL=sc["nll"].mean(), RMSE=float(np.sqrt(np.mean((Xva - mu) ** 2))),
                         coverage80=sc["in80"].mean(), coverage95=sc["in95"].mean()))
    tab = pd.DataFrame(rows)
    base = tab.loc[tab.model == "Constant variance", "CRPS"].iloc[0]
    tab["CRPS_skill_vs_const"] = 1 - tab.CRPS / base
    tab["var_explained"] = 1 - (tab.RMSE / tab.RMSE.iloc[0]) ** 2
    tab.to_csv(os.path.join(OUT, "task9_factor_sameday_scores.csv"), index=False)

    def paired(a, b):
        d = day[a] - day[b]
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else np.nan
    tests = pd.DataFrame([
        dict(comparison="Factor + residual graph vs One-factor only", p=paired("One-factor only", "Factor + residual graph"),
             better_share=float((day["Factor + residual graph"] < day["One-factor only"]).mean())),
        dict(comparison="Factor + residual graph vs Original DYNOTEARS graph", p=paired("Original DYNOTEARS graph", "Factor + residual graph"),
             better_share=float((day["Factor + residual graph"] < day["Original DYNOTEARS graph"]).mean()))])
    tests.to_csv(os.path.join(OUT, "task9_factor_paired_tests.csv"), index=False)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== FACTOR: graph statistics (|w| >= 0.05) ===")
    print(gs.round(3).to_string(index=False))
    print("\n=== FACTOR: same-day conditional distributions, validation block (NOT forecasts) ===")
    print(tab.round(4).to_string(index=False))
    print("\n=== FACTOR: paired Wilcoxon on daily CRPS ===")
    print(tests.round(6).to_string(index=False))


def run_lambda(Xtr, Ytr, Xva, Yva, tickers):
    rows = []
    for lam in (0.03, 0.1, 0.3):
        if lam == 0.1 and os.path.exists(os.path.join("data_out", "fit_static_notears.npz")):
            W = np.load(os.path.join("data_out", "fit_static_notears.npz"), allow_pickle=True)["W"]
            W = np.where(np.abs(W) >= TAU, W, 0.0)
            secs = float("nan")
        else:
            print(f"[LAMBDA] static fit, lambda_W = {lam} ...", flush=True)
            t0 = time.time()
            W, _, _ = fit_dynotears(Xtr, np.zeros((len(Xtr), 0)), lam, lam, TAU, TAU, verbose=False)
            secs = time.time() - t0
        loss = float(np.linalg.norm(Xva - Xva @ W))
        rows.append(dict(lambda_W=lam, loss_ratio_to_zero=loss / np.linalg.norm(Xva), val_RMSE=loss / np.sqrt(Xva.size),
                         fit_seconds=secs, **graph_stats(W, tickers)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "task9_lambda_sensitivity.csv"), index=False)
    print("\n=== LAMBDA sensitivity (static graph, validation block) ===")
    print(tab.round(3).to_string(index=False))


def run_lag(Xtr, Ytr, Xva, Yva, tickers, W0, A0):
    Btr, Bva = rebuild_block(Xtr, Ytr), rebuild_block(Xva, Yva)
    def xy(B, p):
        return B[p:], np.hstack([B[p - i:len(B) - i] for i in range(1, p + 1)])
    rows = []
    for p in (1, 2):
        Xt, Yt = xy(Btr, p)
        Xv, Yv = xy(Bva, p)
        if p == 1:
            W = np.where(np.abs(W0) >= TAU, W0, 0.0); A = np.where(np.abs(A0) >= TAU, A0, 0.0); secs = float("nan")
        else:
            print("[LAG] fitting p = 2 (several minutes) ...", flush=True)
            t0 = time.time()
            W, A, _ = fit_dynotears(Xt, Yt, 0.1, 0.1, TAU, TAU, verbose=False)
            secs = time.time() - t0
        loss = float(np.linalg.norm(Xv - Xv @ W - Yv @ A))
        d = W.shape[0]
        edges_by_lag = [int((A[i * d:(i + 1) * d] != 0).sum()) for i in range(p)]
        rows.append(dict(p=p, val_rows=len(Xv), loss_ratio_to_zero=loss / np.linalg.norm(Xv), val_RMSE=loss / np.sqrt(Xv.size),
                         edges_W=int((W != 0).sum()), edges_A_by_lag=str(edges_by_lag), fit_seconds=secs))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "task9_lag_order.csv"), index=False)
    print("\n=== LAG ORDER: p = 1 vs p = 2 ===")
    print(tab.round(4).to_string(index=False))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=["factor", "lambda", "lag"], choices=["factor", "lambda", "lag"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    Xtr, Ytr, Xva, Yva, tickers, W0, A0 = load()
    print(f"Train {Xtr.shape}, validation {Xva.shape}; sector labels are approximate (plot_basenet.py)")
    if "factor" in a.run:
        run_factor(Xtr, Ytr, Xva, Yva, tickers, W0, A0)
    if "lambda" in a.run:
        run_lambda(Xtr, Ytr, Xva, Yva, tickers)
    if "lag" in a.run:
        run_lag(Xtr, Ytr, Xva, Yva, tickers, W0, A0)
    print(f"\nSaved tables in {OUT}/")
    print("Notes: same-day conditional scores are not forecasts; tau = 0.05 was chosen earlier from validation loss; "
          "sector labels are approximate; Louvain NMI depends on them.")
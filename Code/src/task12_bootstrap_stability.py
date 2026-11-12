"""
Task 12 (M3, experiment 5): which edges of the learned graph are reliable? Block-bootstrap edge stability.

Resample the TRAINING block in short blocks of consecutive days (moving-block bootstrap, so day-to-day dependence
and the cross-sectional correlation of stocks are preserved), refit the whole pipeline on every replicate, and count how
often each edge of the reference graph is selected again.
  residual mode (default): market + sector factors refitted on the replicate, then DYNOTEARS on the standardized
                           residuals (the Task 11 graph).
  raw mode:                DYNOTEARS directly on the standardized returns (the Task 9 "original" graph).
Lag pairs (x_t, x_(t-1)) are built inside each resampled block, so no artificial pair crosses a block boundary.

Reports: number of edges per replicate (mean +/- SD), average Jaccard similarity of edge sets between replicates,
selection frequency of every reference edge (directed and direction-free), sign consistency, the share of stable
edges, and whether stable edges are more often within-sector. Block bootstrap for dependent data is a standard
tool; the block length (default 20 days) is a choice, not a tuned value.
Supports robustness analysis (CSE 516 M3 item 6). DYNOTEARS: Pamfil et al. (2020), AISTATS.
Time-series evaluation background: Hyndman & Athanasopoulos (2021), FPP3, https://otexts.com/fpp3/ (Section 12.5 on bootstrapping).

Usage (from the folder containing data_out/; needs task2_dynotears.py, task11_sector_residual_graph.py,
plot_basenet.py there, and data_out/fit_sector_residual.npz from Task 11 for the reference graph):
  python task12_bootstrap_stability.py --B 6           # quick check (a few minutes)
  python task12_bootstrap_stability.py --B 30          # full run (about an hour; safe to stop, use --resume to continue)
  python task12_bootstrap_stability.py --model raw --B 30
Outputs in data_out/task12/.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task2_dynotears import fit_dynotears
from task11_sector_residual_graph import loo_features

OUT = os.path.join("data_out", "task12")
TAU = 0.05

try:
    from plot_basenet import SECTORS
except Exception as e:
    raise SystemExit(f"Cannot import SECTORS from plot_basenet.py ({e}). Copy plot_basenet.py into this folder.")


def make_bootstrap(block, L, rng, n_target):
    """Moving-block bootstrap of consecutive-day blocks; returns X (current) and Y (lagged) with n_target rows."""
    m = len(block)
    Xs, Ys, total = [], [], 0
    while total < n_target:
        s = int(rng.integers(0, m - L + 1))
        seg = block[s:s + L]
        Xs.append(seg[1:])
        Ys.append(seg[:-1])
        total += L - 1
    return np.vstack(Xs)[:n_target], np.vstack(Ys)[:n_target]


def fit_replicate(Xb, Yb, groups, mode, lam):
    if mode == "raw":
        Z, Yz = Xb, Yb
    else:
        d = Xb.shape[1]
        M, S = loo_features(Xb, groups)
        My, Sy = loo_features(Yb, groups)
        coef = np.zeros((d, 2))
        for j in range(d):
            coef[j] = np.linalg.lstsq(np.column_stack([M[:, j], S[:, j]]), Xb[:, j], rcond=None)[0]
        fac = lambda M_, S_: M_ * coef[:, 0] + S_ * coef[:, 1]
        E = Xb - fac(M, S)
        se = E.std(0)
        Z, Yz = E / se, (Yb - fac(My, Sy)) / se
    W, A, _ = fit_dynotears(Z, Yz, lam, lam, TAU, TAU, verbose=False)
    return W, A


def main(a):
    os.makedirs(OUT, exist_ok=True)
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr = D["X_train"], D["Y_train"]
    tickers = [str(t) for t in D["tickers"]]
    d = len(tickers)
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    lab = np.array([sector_of.get(t, "Other") for t in tickers])
    groups = [list(np.where(lab == s)[0]) for s in dict.fromkeys(lab)]
    if len(groups) == 1:
        raise SystemExit("All stocks fell into one sector: tickers do not match plot_basenet.py SECTORS.")

    ref_file = "fit_sector_residual.npz" if a.model == "residual" else "fit_dynotears.npz"
    ref_path = os.path.join("data_out", ref_file)
    if not os.path.exists(ref_path):
        raise SystemExit(f"Reference graph {ref_path} not found (run the matching earlier task first).")
    Wref = np.load(ref_path, allow_pickle=True)["W"]
    Wref = np.where(np.abs(Wref) >= TAU, Wref, 0.0)
    print(f"Mode = {a.model}; reference graph {ref_file} has {int((Wref != 0).sum())} edges; "
          f"block length {a.block}; B = {a.B}", flush=True)

    block = np.vstack([Ytr[:1], Xtr])               # recover the contiguous training block
    store = os.path.join(OUT, f"boot_{a.model}.npz")
    Ws, As = [], []
    if a.resume and os.path.exists(store):
        S = np.load(store)
        Ws, As = list(S["W"]), list(S["A"])
        print(f"Resuming from {len(Ws)} saved replicates", flush=True)

    for b in range(len(Ws), a.B):
        rng = np.random.default_rng(a.seed + b)     # replicate b always uses the same resample
        t0 = time.time()
        Xb, Yb = make_bootstrap(block, a.block, rng, len(Xtr))
        W, A = fit_replicate(Xb, Yb, groups, a.model, a.lam)
        Ws.append(W); As.append(A)
        np.savez(store, W=np.array(Ws), A=np.array(As))
        print(f"  replicate {b + 1}/{a.B}: {time.time() - t0:.0f}s, edges W = {int((W != 0).sum())}, "
              f"A = {int((A != 0).sum())}", flush=True)

    Wb, Ab = np.array(Ws), np.array(As)
    B = len(Wb)
    sel = Wb != 0
    freq = sel.mean(0)                              # directed selection frequency
    und = (sel | sel.transpose(0, 2, 1)).mean(0)    # direction-free frequency

    n_edges = sel.reshape(B, -1).sum(1)
    flat = sel.reshape(B, -1)
    jac = []
    for p in range(B):
        for q in range(p + 1, B):
            inter, union = (flat[p] & flat[q]).sum(), (flat[p] | flat[q]).sum()
            jac.append(inter / union if union else np.nan)

    i, j = np.nonzero(Wref)
    rows = []
    for u, v in zip(i, j):
        w = Wb[:, u, v]
        chosen = w != 0
        rows.append(dict(source=tickers[u], target=tickers[v], ref_weight=Wref[u, v],
                         freq_directed=freq[u, v], freq_undirected=und[u, v],
                         reverse_freq=freq[v, u],
                         sign_consistency=float((np.sign(w[chosen]) == np.sign(Wref[u, v])).mean()) if chosen.any() else np.nan,
                         source_sector=lab[u], target_sector=lab[v], same_sector=bool(lab[u] == lab[v])))
    edges = pd.DataFrame(rows).sort_values("freq_directed", ascending=False)
    edges.to_csv(os.path.join(OUT, f"task12_edge_stability_{a.model}.csv"), index=False)
    np.save(os.path.join(OUT, f"task12_freq_matrix_{a.model}.npy"), freq)

    summ = dict(model=a.model, replicates=B, block_length=a.block,
                ref_edges=len(edges), edges_per_replicate_mean=float(n_edges.mean()), edges_per_replicate_sd=float(n_edges.std(ddof=1)) if B > 1 else np.nan,
                A_edges_per_replicate_mean=float((Ab != 0).reshape(B, -1).sum(1).mean()),
                mean_pairwise_jaccard=float(np.nanmean(jac)) if jac else np.nan,
                median_freq_directed=float(edges.freq_directed.median()),
                share_freq_ge_0_9=float((edges.freq_directed >= 0.9).mean()),
                share_freq_ge_0_7=float((edges.freq_directed >= 0.7).mean()),
                share_freq_ge_0_5=float((edges.freq_directed >= 0.5).mean()),
                share_undirected_ge_0_7=float((edges.freq_undirected >= 0.7).mean()),
                within_sector_share_stable=float(edges[edges.freq_directed >= 0.7].same_sector.mean()) if (edges.freq_directed >= 0.7).any() else np.nan,
                within_sector_share_unstable=float(edges[edges.freq_directed < 0.5].same_sector.mean()) if (edges.freq_directed < 0.5).any() else np.nan)
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, f"task12_summary_{a.model}.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(edges.freq_directed, bins=np.linspace(0, 1, 11), alpha=.8, label="directed")
    ax[0].hist(edges.freq_undirected, bins=np.linspace(0, 1, 11), alpha=.5, label="direction ignored")
    ax[0].set_xlabel("Selection frequency over bootstrap replicates"); ax[0].set_ylabel("Reference edges"); ax[0].legend()
    ax[1].scatter(edges.ref_weight.abs(), edges.freq_directed, s=14)
    ax[1].set_xlabel("|weight| in reference graph"); ax[1].set_ylabel("Selection frequency"); ax[1].grid(alpha=.3)
    fig.suptitle(f"Edge stability ({a.model} graph, {B} block-bootstrap replicates, block length {a.block})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"task12_stability_{a.model}.png"), dpi=150); plt.close(fig)

    pd.set_option("display.width", 230, "display.max_columns", 30)
    print("\n=== Stability summary ===")
    print(pd.DataFrame([summ]).T.round(3).to_string(header=False))
    print("\n=== Most stable edges (top 15) ===")
    print(edges.head(15).round(3).to_string(index=False))
    print("\n=== Least stable reference edges (bottom 10) ===")
    print(edges.tail(10).round(3).to_string(index=False))
    print(f"\nSaved tables, matrix and figure in {OUT}/")
    print("Notes: with few replicates the frequencies are coarse; the block length is a choice; the reference graph was "
          "fitted on the full training block with lambda and tau chosen earlier; the sector labels are approximate; "
          "edge directions are not causal claims.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["residual", "raw"], default="residual")
    ap.add_argument("--B", type=int, default=30, help="number of bootstrap replicates")
    ap.add_argument("--block", type=int, default=20, help="block length in days")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from saved replicates")
    main(ap.parse_args())
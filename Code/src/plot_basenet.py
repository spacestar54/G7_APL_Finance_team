"""
Draw the learned base net (intra-slice matrix W) from the Task 2 fit on S&P 100 returns.
Figure 4 of Pamfil et al. (2020) shows W with stocks sorted by sector (row = source, column = target).

Usage
  python plot_basenet.py                 # uses data_out/fit_dynotears.npz, tau = 0.05
  python plot_basenet.py --tau 0.1
  python plot_basenet.py --fit data_out/fit_static_notears.npz --tag static

Outputs: data_out/basenet_heatmap_<tag>.png, data_out/basenet_graph_<tag>.png (+ NMI printed)
Needs: numpy, matplotlib, networkx, scikit-learn  (pip install networkx scikit-learn matplotlib)

SECTOR LABELS: the paper does not publish a ticker-to-sector table. The groups below follow the ORDER of
tickers along the axes of Figure 4 and the nine sector names printed there. This is an approximation read
from the figure (for example COST appears in the first group, which a standard GICS list would not do).
Check it and edit SECTORS if your instructor wants a different classification.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECTORS = {
    "Consumer Cyclicals": "COST DIS CMCSA NKE SBUX CHTR F AMZN MCD TGT LOW GM HD BKNG".split(),
    "Consumer Non-Cyclicals": "CL PG PM MDLZ MO WBA PEP WMT KO".split(),
    "Energy": "KMI CVX SLB OXY XOM COP".split(),
    "Financials": "BK MET C ALL JPM MS SPG AIG AXP USB WFC BLK COF BAC GS".split(),
    "Healthcare": "LLY AMGN MRK AGN ABT ABBV PFE MDT BIIB CVS JNJ DHR BMY UNH CELG GILD".split(),
    "Industrials": "DD UTX RTN EMR UPS CAT GD GE MMM UNP HON BA LMT FDX".split(),
    "Technology": "QCOM TXN V AAPL NVDA ACN ADBE CSCO FB ORCL GOOGL GOOG MA MSFT NFLX INTC IBM".split(),
    "Telecommunications": "T VZ".split(),
    "Utilities": "EXC DUK SO NEE".split(),
}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def sector_order(tickers):
    """Return (ordered column indices, sector name per ordered node, sector boundaries)."""
    order, labels, bounds = [], [], []
    for s, members in SECTORS.items():
        idx = [tickers.index(t) for t in members if t in tickers]
        if idx:
            order += idx
            labels += [s] * len(idx)
            bounds.append((s, len(order) - len(idx), len(order)))
    missing = [t for t in tickers if all(t not in m for m in SECTORS.values())]
    if missing:
        print("Tickers without a sector (appended last as 'Other'):", missing)
        order += [tickers.index(t) for t in missing]
        labels += ["Other"] * len(missing)
        bounds.append(("Other", len(order) - len(missing), len(order)))
    return order, labels, bounds


def heatmap(W, names, bounds, path, tau):
    d = len(names)
    fig, ax = plt.subplots(figsize=(11, 10))
    vmax = np.abs(W).max()
    im = ax.imshow(np.where(W != 0, W, np.nan), cmap="Reds" if (W >= 0).all() else "RdBu_r",
                   vmin=-vmax if not (W >= 0).all() else 0, vmax=vmax)
    ax.set_facecolor("#f7f7f7")
    ax.set_xticks(range(d)); ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticks(range(d)); ax.set_yticklabels(names, fontsize=5)
    for s, a, b in bounds:
        ax.add_patch(plt.Rectangle((a - .5, a - .5), b - a, b - a, fill=False, ec="k", lw=1))
        ax.text(b - .5, a - .5, s, fontsize=7, ha="right", va="bottom")
    ax.set_xlabel("target stock"); ax.set_ylabel("source stock")
    ax.set_title(f"Learned intra-slice W, stocks sorted by sector (|w| >= {tau}); row = source, column = target")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def graph(W, names, sec_labels, path, tau):
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))
    for i, j in zip(*np.nonzero(W)):
        G.add_edge(i, j, w=float(W[i, j]))
    sec_names = list(dict.fromkeys(sec_labels))
    col = {s: COLORS[k % 10] for k, s in enumerate(sec_names)}
    pos = nx.spring_layout(G.to_undirected(), seed=0, k=1.2 / np.sqrt(len(names)), iterations=200)
    wts = np.array([G[u][v]["w"] for u, v in G.edges()]) if G.number_of_edges() else np.array([])
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.4 + 3 * np.abs(wts) / (np.abs(wts).max() or 1),
                           edge_color="#888888", alpha=.5, arrowsize=6, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[s] for s in sec_labels], node_size=170,
                           edgecolors="k", linewidths=.4)
    nx.draw_networkx_labels(G, pos, labels=dict(enumerate(names)), font_size=5, ax=ax)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col[s], label=s) for s in sec_names],
              fontsize=7, loc="lower left")
    ax.set_title(f"Learned base net: {G.number_of_edges()} directed edges with |w| >= {tau} "
                 f"({len(names)} stocks, colored by sector)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return G


def louvain_nmi(G, sec_labels):
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as nmi
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    for u, v, dct in G.edges(data=True):          # symmetrize: sum of |w| over both directions
        w = abs(dct["w"]) + (abs(G[v][u]["w"]) if G.has_edge(v, u) else 0)
        U.add_edge(u, v, weight=w)
    comms = nx.community.louvain_communities(U, weight="weight", seed=0)
    pred = np.zeros(len(sec_labels), dtype=int)
    for k, c in enumerate(comms):
        for n in c:
            pred[n] = k
    return nmi(sec_labels, pred), len(comms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join("data_out", "fit_dynotears.npz"))
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tag", default="dynotears")
    a = ap.parse_args()

    F = np.load(a.fit, allow_pickle=True)
    W, tickers = F["W"].copy(), [str(t) for t in F["tickers"]]
    W[np.abs(W) < a.tau] = 0.0
    order, sec_labels, bounds = sector_order(tickers)
    Wo = W[np.ix_(order, order)]
    names = [tickers[i] for i in order]

    print(f"{(Wo != 0).sum()} edges kept at |w| >= {a.tau}; positive share = "
          f"{(Wo[Wo != 0] > 0).mean():.1%}")
    heatmap(Wo, names, bounds, os.path.join("data_out", f"basenet_heatmap_{a.tag}.png"), a.tau)
    G = graph(Wo, names, sec_labels, os.path.join("data_out", f"basenet_graph_{a.tag}.png"), a.tau)
    try:
        score, k = louvain_nmi(G, sec_labels)
        print(f"Louvain found {k} communities; NMI vs sector labels = {score:.2f} "
              f"(paper: about 0.79 with its own 97-stock graph and sector table)")
    except Exception as e:
        print("NMI skipped:", e)
    print("Saved PNGs in data_out/")
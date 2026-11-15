"""
Task 14 (M4): probability quality for EVENTS - Brier score, log-loss, calibration (ECE), ROC-AUC, reliability diagrams.
CSE 516 guidelines Sections 8.1 and 6: a PGM should say P(event | evidence) = 0.82, not just a class label.

Every model in this project gives each stock a Normal(mu, sigma^2) predictive distribution, so event probabilities follow:
  direction  y = 1 if the return is positive:      p = Phi(mu / sigma)
  large move y = 1 if |return| > c  (c = the 90th percentile of |return| in the TRAINING window, per stock):
                                                    p = 1 - [Phi((c-mu)/sigma) - Phi((-c-mu)/sigma)]
Climatology (the training-window frequency of the event) is the reference forecast for the Brier skill score.

PART A  Next-day, rolling origins (same design as task7/task8): zero mean with constant, EWMA and GARCH(1,1) variance.
        Zero-mean models always give P(direction) = 0.5, so the direction event shows whether anything has skill;
        the large-move event is where volatility models can help.
PART B  SAME-DAY conditional (NOT forecasts) on the validation block: constant variance, one-factor, market + sector,
        original DYNOTEARS graph, market + sector + residual graph. Direction is the natural 'class' to predict.

Metrics: Brier score (lower is better), log-loss, expected calibration error (10 equal-width bins), ROC-AUC,
accuracy for direction, Brier skill score vs climatology (pooled: 1 - mean Brier / mean climatology Brier; the mean of block-level ratios is also given); mean +/- SD over rolling blocks (Part A), bootstrap 95% CIs
over blocks or days (approximate: dependence is ignored), paired Wilcoxon tests.
References: Hyndman & Athanasopoulos (2021) FPP3 Sections 5.5, 5.9; Pamfil et al. (2020) for DYNOTEARS.

Usage (folder containing data_out/; needs task7/task8/task11 scripts and plot_basenet.py there; Part B needs
data_out/dataset.npz, fit_dynotears.npz and fit_sector_residual.npz):
  python task14_brier_calibration.py            # a few minutes
  python task14_brier_calibration.py --skip-b   # Part A only
Outputs in data_out/task14/.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, wilcoxon
from sklearn.metrics import roc_auc_score

from task8_distributional import fit_garch, garch_var, ewma_var

OUT = os.path.join("data_out", "task14")
rng = np.random.default_rng(0)
EPS = 1e-6


# ---------------------------------------------------------------------------
def p_direction(mu, sig):
    return norm.cdf(mu / sig)


def p_large(mu, sig, c):
    return 1 - (norm.cdf((c - mu) / sig) - norm.cdf((-c - mu) / sig))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(dict(bin=b, mean_pred=float(p[m].mean()), obs_freq=float(y[m].mean()), count=int(m.sum())))
    return pd.DataFrame(rows)


def ece(p, y, bins=10):
    r = reliability(p, y, bins)
    return float((r["count"] / r["count"].sum() * (r.mean_pred - r.obs_freq).abs()).sum())


def auc(p, y):
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------
def part_a(R, start, step, lam):
    n, d = R.shape
    origins = list(range(start, n - step + 1, step))
    print(f"[Part A] {len(origins)} origins, d = {d}", flush=True)
    blocks, pool = [], {}
    for k, o in enumerate(origins):
        mu, sd = R[:o].mean(0), R[:o].std(0)
        Z = (R - mu) / sd
        X = Z[o:o + step]
        c = np.quantile(np.abs(Z[:o]), 0.9, axis=0)
        params = np.array([fit_garch(Z[:o, j]) for j in range(d)])
        sig = {"Constant variance": np.ones((step, d)),
               "EWMA (0.94)": np.sqrt(ewma_var(Z[:o + step], lam)[o:]),
               "GARCH(1,1)": np.sqrt(garch_var(Z[:o + step], params)[o:])}
        probs = {}
        for name, s in sig.items():
            probs[(name, "large move")] = p_large(0.0, s, c)
            probs[(name, "direction")] = p_direction(0.0, s)
        clim_large = np.tile((np.abs(Z[:o]) > c).mean(0), (step, 1))
        clim_dir = np.tile((Z[:o] > 0).mean(0), (step, 1))
        probs[("Climatology", "large move")] = clim_large
        probs[("Climatology", "direction")] = clim_dir
        ys = {"large move": (np.abs(X) > c).astype(float), "direction": (X > 0).astype(float)}
        for (name, ev), P in probs.items():
            y = ys[ev]
            blocks.append(dict(origin=o, model=name, event=ev, brier=brier(P, y), logloss=logloss(P, y)))
            pool.setdefault((name, ev), []).append((P.ravel(), y.ravel()))
        if k % 8 == 0:
            print(f"  origin {o} ({k + 1}/{len(origins)})", flush=True)
    return pd.DataFrame(blocks), {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for k, v in pool.items()}


def summarize_a(blocks, pool):
    rows = []
    for ev in ("large move", "direction"):
        sub = blocks[blocks.event == ev]
        piv = sub.pivot(index="origin", columns="model", values="brier")
        for m in piv.columns:
            p_, y_ = pool[(m, ev)]
            bss_blocks = 1 - piv[m] / piv["Climatology"]                       # mean of block-level ratios
            bss = 1 - piv[m].mean() / piv["Climatology"].mean()                # pooled: ratio of mean Brier scores
            diff = piv["Constant variance"] - piv[m]
            try:
                pw = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
            except ValueError:
                pw = np.nan
            bm, bc = piv[m].values, piv["Climatology"].values
            boot = []
            for _ in range(2000):
                ii = rng.integers(0, len(bm), len(bm))
                boot.append(1 - bm[ii].mean() / bc[ii].mean())
            rows.append(dict(event=ev, model=m, brier_mean=piv[m].mean(), brier_sd=piv[m].std(ddof=1),
                             logloss_mean=sub[sub.model == m].logloss.mean(), ECE=ece(p_, y_), AUC=auc(p_, y_),
                             BSS_vs_climatology=bss, BSS_mean_of_block_ratios=bss_blocks.mean(), BSS_CI95_low=np.percentile(boot, 2.5),
                             BSS_CI95_high=np.percentile(boot, 97.5), wilcoxon_p_vs_constant=pw, n_blocks=len(piv)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------
def part_b(tau):
    from task11_sector_residual_graph import loo_features
    from plot_basenet import SECTORS
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = [str(t) for t in D["tickers"]]
    d = len(tickers)
    sector_of = {t: s for s, ms in SECTORS.items() for t in ms}
    lab = np.array([sector_of.get(t, "Other") for t in tickers])
    groups = [list(np.where(lab == s)[0]) for s in dict.fromkeys(lab)]
    if len(groups) == 1:
        raise SystemExit("All stocks in one sector: tickers do not match plot_basenet.py SECTORS.")

    Mtr, Str = loo_features(Xtr, groups)
    Mva, Sva = loo_features(Xva, groups)
    models = {"Constant variance": (np.zeros_like(Xtr), np.zeros_like(Xva))}
    beta = (Mtr * Xtr).sum(0) / (Mtr ** 2).sum(0)
    models["One-factor (market)"] = (Mtr * beta, Mva * beta)
    coef = np.zeros((d, 2))
    for j in range(d):
        coef[j] = np.linalg.lstsq(np.column_stack([Mtr[:, j], Str[:, j]]), Xtr[:, j], rcond=None)[0]
    fac = lambda M, S: M * coef[:, 0] + S * coef[:, 1]
    models["Market + sector factor"] = (fac(Mtr, Str), fac(Mva, Sva))

    F0 = np.load(os.path.join("data_out", "fit_dynotears.npz"), allow_pickle=True)
    W0 = np.where(np.abs(F0["W"]) >= tau, F0["W"], 0.0)
    A0 = np.where(np.abs(F0["A"]) >= tau, F0["A"], 0.0)
    models["Original DYNOTEARS graph"] = (Xtr @ W0 + Ytr @ A0, Xva @ W0 + Yva @ A0)

    fp = os.path.join("data_out", "fit_sector_residual.npz")
    if os.path.exists(fp):
        F2 = np.load(fp, allow_pickle=True)
        W2, A2 = F2["W"], F2["A"]
        My_tr, Sy_tr = loo_features(Ytr, groups)
        My_va, Sy_va = loo_features(Yva, groups)
        se = (Xtr - fac(Mtr, Str)).std(0)
        Ztr, Zva = (Xtr - fac(Mtr, Str)) / se, (Xva - fac(Mva, Sva)) / se
        Yz_tr, Yz_va = (Ytr - fac(My_tr, Sy_tr)) / se, (Yva - fac(My_va, Sy_va)) / se
        models["Market + sector + residual graph"] = (fac(Mtr, Str) + (Ztr @ W2 + Yz_tr @ A2) * se,
                                                      fac(Mva, Sva) + (Zva @ W2 + Yz_va @ A2) * se)
    else:
        print("[Part B] fit_sector_residual.npz missing: run task11_sector_residual_graph.py first (that model is skipped).")

    c = np.quantile(np.abs(Xtr), 0.9, axis=0)
    ys = {"direction": (Xva > 0).astype(float), "large move": (np.abs(Xva) > c).astype(float)}
    clim = {"direction": np.tile((Xtr > 0).mean(0), (len(Xva), 1)),
            "large move": np.tile((np.abs(Xtr) > c).mean(0), (len(Xva), 1))}
    rows, rel, daily = [], {}, {}
    for ev in ("direction", "large move"):
        y = ys[ev]
        pc = clim[ev]
        for name, (mtr, mva) in models.items():
            sig = (Xtr - mtr).std(0)
            P = p_direction(mva, sig) if ev == "direction" else p_large(mva, sig, c)
            daily[(ev, name)] = ((P - y) ** 2).mean(1)
            bs_clim = brier(pc, y)
            rows.append(dict(event=ev, model=name, brier=brier(P, y), logloss=logloss(P, y), ECE=ece(P.ravel(), y.ravel()),
                             AUC=auc(P.ravel(), y.ravel()),
                             accuracy=float(((P > 0.5) == (y > 0.5)).mean()) if ev == "direction" else np.nan,
                             BSS_vs_climatology=1 - brier(P, y) / bs_clim))
            rel[(ev, name)] = reliability(P.ravel(), y.ravel())
        rows.append(dict(event=ev, model="Climatology", brier=brier(pc, y), logloss=logloss(pc, y), ECE=ece(pc.ravel(), y.ravel()),
                         AUC=np.nan, accuracy=np.nan, BSS_vs_climatology=0.0))
    tab = pd.DataFrame(rows)

    tests = []
    for ev in ("direction", "large move"):
        for a, b in [("Market + sector factor", "One-factor (market)"),
                     ("Market + sector + residual graph", "Market + sector factor"),
                     ("Market + sector + residual graph", "Original DYNOTEARS graph")]:
            if (ev, a) in daily and (ev, b) in daily:
                diff = daily[(ev, b)] - daily[(ev, a)]                  # positive = first model has lower Brier
                boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                tests.append(dict(event=ev, model=a, versus=b, mean_brier_gain=diff.mean(), CI95_low=np.percentile(boot, 2.5),
                                  CI95_high=np.percentile(boot, 97.5), better_share_of_days=float((diff > 0).mean()),
                                  wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan))
    return tab, pd.DataFrame(tests), rel


def reliability_plot(items, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect calibration")
    for name, r in items.items():
        ax.plot(r.mean_pred, r.obs_freq, marker="o", ms=4, label=name)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 230, "display.max_columns", 30)
    R = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True).values
    blocks, pool = part_a(R, a.start, a.step, a.lam)
    blocks.to_csv(os.path.join(OUT, "task14_rolling_block_scores.csv"), index=False)
    ta = summarize_a(blocks, pool)
    ta.to_csv(os.path.join(OUT, "task14_rolling_event_table.csv"), index=False)
    print("\n=== PART A: next-day event probabilities, rolling origins ===")
    print(ta.round(4).to_string(index=False))
    reliability_plot({m: reliability(*pool[(m, "large move")]) for m in ("Constant variance", "EWMA (0.94)", "GARCH(1,1)")},
                     "Part A: P(large move) reliability (next day)", os.path.join(OUT, "task14_reliability_rolling_large_move.png"))

    if not a.skip_b:
        tb, tests, rel = part_b(a.tau)
        tb.to_csv(os.path.join(OUT, "task14_sameday_event_table.csv"), index=False)
        tests.to_csv(os.path.join(OUT, "task14_sameday_paired_tests.csv"), index=False)
        print("\n=== PART B: SAME-DAY conditional event probabilities (NOT forecasts), validation block ===")
        print(tb.round(4).to_string(index=False))
        print("\n=== PART B: paired tests on daily Brier score ===")
        print(tests.round(5).to_string(index=False))
        for ev in ("direction", "large move"):
            reliability_plot({n: r for (e, n), r in rel.items() if e == ev}, f"Part B: P({ev}) reliability (same day)",
                             os.path.join(OUT, f"task14_reliability_sameday_{ev.replace(' ', '_')}.png"))
    print(f"\nSaved tables and figures in {OUT}/")
    print("Notes: Gaussian predictive distributions (returns are heavy-tailed, which can distort tail-event probabilities); "
          "the large-move threshold is the training 90th percentile of |return| per stock; overlapping training windows and "
          "day-level resampling make CIs and tests approximate; Part B scores are same-day conditional, not forecasts; "
          "lambda and tau of the graph were chosen with the validation block, which slightly favors it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA decay")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--skip-b", action="store_true")
    main(ap.parse_args())
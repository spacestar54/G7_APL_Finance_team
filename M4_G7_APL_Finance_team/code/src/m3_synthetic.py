"""
M3, family A: synthetic experiments with known ground truth (CSE516, DYNOTEARS project).

What it does
  Runs DYNOTEARS and the M2 baseline (NOTEARS + Lasso) under identical conditions on simulated data,
  over many seeds, while varying ONE thing at a time away from a base setting:
      size     sample size n            (Sample Size vs SHD plot)
      dim      number of nodes d
      density  intra-slice graph density k_w
      noise    gaussian vs exponential noise
      missing  fraction of entries missing (mean-imputed afterwards)
      meas     measurement noise added to the observed series
      ablation DYNOTEARS variants (no lag term, no sparsity, no threshold, different lambda)
  Same seed = same dataset for every method, so comparisons are paired.

Usage (run from the repo root, the folder that contains task2_dynotears.py)
  python m3_synthetic.py --quick                  # tiny smoke test, ~1-2 min
  python m3_synthetic.py --exps size --seeds 10   # one experiment
  python m3_synthetic.py --seeds 10 --jobs 4      # everything (tens of minutes)
  python m3_synthetic.py --analyze                # tables, statistics and plots from saved results
  (add --quick to --analyze to analyze the smoke-test file instead)

Results are appended to data_out/m3_synth/results.csv after every batch, so you can stop and re-run:
finished (experiment, setting, seed) jobs are skipped.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from task2_dynotears import fit_dynotears, simulate, edge_metrics
from task3_simulation import run_notears_lasso

warnings.filterwarnings("ignore")
OUT = os.path.join("data_out", "m3_synth")
RESULTS = os.path.join(OUT, "results.csv")

BASE = dict(noise="gaussian", n=500, d=10, k_w=2.0, k_a=2.0, miss=0.0, meas=0.0)
EXPS = {
    "size": ("n", [50, 100, 200, 500, 1000]),
    "dim": ("d", [5, 10, 20]),
    "density": ("k_w", [1.0, 2.0, 4.0]),
    "noise": ("noise", ["gaussian", "exponential"]),
    "missing": ("miss", [0.0, 0.05, 0.1, 0.2]),
    "meas": ("meas", [0.0, 0.25, 0.5, 1.0]),
}
ABLATION_VARIANTS = ["full", "no_lag", "no_sparsity", "no_threshold", "lam_x4", "lam_x0.25"]
ABLATION_NS = [50, 500]


# ----------------------------------------------------------------------------
# Hyperparameters. The paper (Appendix B.3) only gives n = 500 and n = 50.
# For other n we interpolate on a log scale between those two settings (n >= 500 uses the n = 500 values).
# ----------------------------------------------------------------------------
def hp_for(n):
    if n >= 500:
        return dict(lam=0.05, tau_w=0.3, tau_a=0.1)
    if n <= 50:
        return dict(lam=0.2, tau_w=0.3, tau_a=0.2)
    t = (np.log(n) - np.log(50)) / (np.log(500) - np.log(50))
    lam = float(np.exp(np.log(0.2) + t * (np.log(0.05) - np.log(0.2))))
    return dict(lam=lam, tau_w=0.3, tau_a=float(0.2 + t * (0.1 - 0.2)))


# ----------------------------------------------------------------------------
# Data corruption for the robustness experiments (p = 1, so Y is X shifted by one row)
# ----------------------------------------------------------------------------
def corrupt(X, Y, miss, meas, rng):
    if miss == 0 and meas == 0:
        return X, Y
    s = np.vstack([Y[:1], X])                       # rebuild the full series (n+1 rows)
    if meas > 0:
        s = s + rng.normal(size=s.shape) * meas * s.std(axis=0)
    if miss > 0:
        mask = rng.random(s.shape) < miss
        col_mean = np.nanmean(np.where(mask, np.nan, s), axis=0)
        s = np.where(mask, col_mean, s)             # mean imputation
    return s[1:], s[:-1]


# ----------------------------------------------------------------------------
# Methods and ablation variants. Each returns W, A, seconds.
# ----------------------------------------------------------------------------
def run_dynotears(X, Y, hp):
    W, A, info = fit_dynotears(X, Y, hp["lam"], hp["lam"], hp["tau_w"], hp["tau_a"])
    return W, A, info["seconds"]


def run_variant(name, X, Y, hp):
    n, d = X.shape
    lam, tw, ta = hp["lam"], hp["tau_w"], hp["tau_a"]
    if name == "full":
        return run_dynotears(X, Y, hp)
    if name == "no_lag":                            # static NOTEARS: no A at all
        W, A, info = fit_dynotears(X, np.zeros((n, 0)), lam, lam, tw, 0.0)
        return W, A, info["seconds"]
    if name == "no_sparsity":
        W, A, info = fit_dynotears(X, Y, 0.0, 0.0, tw, ta)
    elif name == "no_threshold":
        W, A, info = fit_dynotears(X, Y, lam, lam, 0.0, 0.0)
    elif name == "lam_x4":
        W, A, info = fit_dynotears(X, Y, 4 * lam, 4 * lam, tw, ta)
    elif name == "lam_x0.25":
        W, A, info = fit_dynotears(X, Y, 0.25 * lam, 0.25 * lam, tw, ta)
    else:
        raise ValueError(name)
    return W, A, info["seconds"]


METHODS = {"DYNOTEARS": run_dynotears, "NOTEARS+Lasso": run_notears_lasso}


# ----------------------------------------------------------------------------
# Metrics: F1 / SHD plus an error breakdown and recall on weak vs strong true edges
# ----------------------------------------------------------------------------
def breakdown(T, E, weak_below):
    Tb, Eb = T != 0, E != 0
    fp = int((~Tb & Eb).sum())
    fn = int((Tb & ~Eb).sum())
    rev = int((Tb & ~Eb & Eb.T & ~Tb.T).sum()) if T.shape[0] == T.shape[1] else 0
    weak, strong = Tb & (np.abs(T) < weak_below), Tb & (np.abs(T) >= weak_below)

    def rec(mask):
        return float((mask & Eb).sum() / mask.sum()) if mask.sum() else np.nan
    return dict(missing=fn - rev, spurious=fp - rev, reversed=rev, rec_weak=rec(weak), rec_strong=rec(strong))


def score(Wt, At, W, A):
    row = {}
    mw = edge_metrics(Wt, W)
    bw = breakdown(Wt, W, 1.0)
    for k, v in mw.items():
        row[f"{k}_W"] = v
    for k, v in bw.items():
        row[f"{k}_W"] = v
    if A.shape[0] == 0:                             # variant without a lag term
        for k in ("precision", "recall", "f1", "shd"):
            row[f"{k}_A"] = np.nan
        for k in ("missing", "spurious", "reversed", "rec_weak", "rec_strong"):
            row[f"{k}_A"] = np.nan
    else:
        ma = edge_metrics(At, A)
        ba = breakdown(At, A, 0.4)
        for k, v in ma.items():
            row[f"{k}_A"] = v
        for k, v in ba.items():
            row[f"{k}_A"] = v
    return row


def run_job(job):
    cfg, seed = job["cfg"], job["seed"]
    X, Y, Wt, At = simulate(d=cfg["d"], n=cfg["n"], p=1, k_w=cfg["k_w"], k_a=cfg["k_a"],
                            eta=1.0, noise=cfg["noise"], seed=seed)
    X, Y = corrupt(X, Y, cfg["miss"], cfg["meas"], np.random.default_rng(10_000 + seed))
    hp = hp_for(cfg["n"])
    rows = []
    for name in job["methods"]:
        if job["exp"] == "ablation":
            W, A, sec = run_variant(name, X, Y, hp)
        else:
            W, A, sec = METHODS[name](X, Y, hp)
        row = dict(exp=job["exp"], setting=job["setting"], seed=seed, method=name, seconds=sec, job_id=job["id"], **cfg)
        row.update(score(Wt, At, W, A))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Build the job list, run in batches, append to CSV
# ----------------------------------------------------------------------------
def build_jobs(exps, seeds, quick):
    jobs = []
    for exp in exps:
        if exp == "ablation":
            settings = [(f"n={n}", dict(BASE, n=n), ABLATION_VARIANTS) for n in ABLATION_NS]
        else:
            param, values = EXPS[exp]
            if quick:
                values = values[:2]
            settings = [(str(v), dict(BASE, **{param: v}), list(METHODS)) for v in values]
        for label, cfg, methods in settings:
            if quick:
                cfg = dict(cfg, n=min(cfg["n"], 200), d=min(cfg["d"], 5))
            for seed in range(seeds):
                jid = json.dumps([exp, label, seed])
                jobs.append(dict(exp=exp, setting=label, cfg=cfg, seed=seed, methods=methods, id=jid))
    return jobs


def run_all(args):
    os.makedirs(OUT, exist_ok=True)
    exps = args.exps or (list(EXPS) + ["ablation"])
    seeds = 3 if args.quick else args.seeds
    jobs = build_jobs(exps, seeds, args.quick)
    path = RESULTS if not args.quick else os.path.join(OUT, "results_quick.csv")
    done = set(pd.read_csv(path).job_id) if os.path.exists(path) else set()
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs total, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    step = max(args.jobs * 4, 4)
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        res = Parallel(n_jobs=args.jobs)(delayed(run_job)(j) for j in batch)
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"  finished {min(i + step, len(todo))}/{len(todo)}", flush=True)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------
# Analysis: mean +/- SD, 95% CI, paired tests with Holm correction, ANOVA, plots
# ----------------------------------------------------------------------------
METRICS = ["f1_W", "f1_A", "shd_W", "shd_A", "seconds"]


def ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def boot_ci_mean(diff, rng, B=5000):
    diff = np.asarray(diff, float)
    m = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def order_settings(sub, exp):
    vals = EXPS[exp][1] if exp in EXPS else None
    labels = [str(v) for v in vals] if vals else sorted(sub.setting.unique())
    return [l for l in labels if l in set(sub.setting)]


def analyze(path=RESULTS):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    df = pd.read_csv(path)
    main = df[df.exp != "ablation"]
    rng = np.random.default_rng(0)

    # ---- summary tables ----
    g = main.groupby(["exp", "setting", "method"])
    num = pd.DataFrame(index=g.size().index)
    pretty = pd.DataFrame(index=g.size().index)
    num["n_seeds"] = g.size()
    cols = METRICS + ["precision_W", "recall_W", "precision_A", "recall_A", "missing_W", "spurious_W", "reversed_W",
                      "missing_A", "spurious_A", "rec_weak_W", "rec_strong_W", "rec_weak_A", "rec_strong_A"]
    for c in cols:
        mean, sd = g[c].mean(), g[c].std(ddof=1).fillna(0)
        ci = g[c].apply(ci95)
        num[f"{c}_mean"], num[f"{c}_sd"], num[f"{c}_ci95"] = mean, sd, ci
        if c in METRICS:
            pretty[c] = [f"{m:.2f} +/- {s:.2f}" for m, s in zip(mean, sd)]
    num.reset_index().to_csv(os.path.join(OUT, "summary_numeric.csv"), index=False)
    pretty.reset_index().to_csv(os.path.join(OUT, "summary_pretty.csv"), index=False)

    # ---- paired tests DYNOTEARS vs NOTEARS+Lasso, same seeds ----
    rows = []
    for (exp, setting), sub in main.groupby(["exp", "setting"]):
        a = sub[sub.method == "DYNOTEARS"].sort_values("seed")
        b = sub[sub.method == "NOTEARS+Lasso"].sort_values("seed")
        if len(a) != len(b) or len(a) < 2:
            continue
        for metric in ("shd_W", "shd_A", "f1_W", "f1_A"):
            diff = a[metric].values - b[metric].values
            try:
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            lo, hi = boot_ci_mean(diff, rng)
            rows.append(dict(exp=exp, setting=setting, metric=metric, n_pairs=len(diff), mean_diff=float(diff.mean()),
                             boot_ci_low=lo, boot_ci_high=hi, wilcoxon_p=p))
    tests = pd.DataFrame(rows)
    if len(tests):
        tests["p_holm"] = multipletests(tests.wilcoxon_p.values, method="holm")[1]
        tests["significant_0.05_holm"] = tests.p_holm < 0.05
    tests.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)

    # ---- two-way ANOVA on the sample-size experiment ----
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        s = main[main.exp == "size"].copy()
        s["method"] = s.method.str.replace("+", "_", regex=False)
        out = []
        for metric in ("f1_W", "shd_W", "f1_A", "shd_A"):
            fit = smf.ols(f"{metric} ~ C(method) * C(n)", data=s).fit()
            t = anova_lm(fit, typ=2)
            t["metric"] = metric
            out.append(t.reset_index())
        pd.concat(out).to_csv(os.path.join(OUT, "anova_size.csv"), index=False)
    except Exception as e:
        print("ANOVA skipped:", e)

    # ---- ablation table ----
    ab = df[df.exp == "ablation"]
    if len(ab):
        g2 = ab.groupby(["setting", "method"])
        t = pd.DataFrame(index=g2.size().index)
        for c in ["f1_W", "f1_A", "shd_W", "shd_A", "seconds"]:
            t[c] = [f"{m:.2f} +/- {s:.2f}" for m, s in zip(g2[c].mean(), g2[c].std(ddof=1).fillna(0))]
        t.reset_index().to_csv(os.path.join(OUT, "ablation_table.csv"), index=False)

    # ---- plots ----
    colors = {"DYNOTEARS": "tab:blue", "NOTEARS+Lasso": "tab:orange"}
    for exp in EXPS:
        sub = main[main.exp == exp]
        if sub.empty:
            continue
        labels = order_settings(sub, exp)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for ax, (metric, title) in zip(axes, [("shd_W", "SHD of W (same-day graph)"), ("shd_A", "SHD of A (lag graph)"),
                                              ("seconds", "Run time (s)")]):
            for m, col in colors.items():
                st = sub[sub.method == m].groupby("setting")[metric].agg(["mean", "std"]).reindex(labels)
                ax.errorbar(range(len(labels)), st["mean"], yerr=st["std"].fillna(0), color=col, marker="o",
                            capsize=3, label=m)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlabel(EXPS[exp][0])
            ax.set_title(title)
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Experiment '{exp}' (mean +/- SD over seeds, paired data)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"plot_{exp}.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, col in colors.items():
        s = main[main.method == m]
        ax.scatter(s.seconds, s.shd_W + s.shd_A, s=10, alpha=.4, color=col, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Run time (s, log scale)")
    ax.set_ylabel("SHD(W) + SHD(A)")
    ax.set_title("Accuracy vs computation time (every run)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_shd_vs_time.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote tables and plots to {OUT}/")
    if len(tests):
        print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--exps", nargs="+", choices=list(EXPS) + ["ablation"], default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (e.g. 4)")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(os.path.join(OUT, "results_quick.csv") if a.quick else RESULTS)
    else:
        run_all(a)

"""
Task 8 (M3, experiment 3): probability quality of predictive distributions.

PART A  Next-day predictive distributions, rolling origins (same design as task7_rolling_eval.py)
        Each stock's next-day return ~ N(0, sigma_t^2), where the mean is zero (no model beat zero in Task 7)
        and the variance comes from:
          - Constant variance  (sigma = 1 on the training-standardized scale)
          - EWMA volatility    (lambda = 0.94, RiskMetrics style; no parameters to fit)
          - GARCH(1,1)         (fitted per stock by Gaussian quasi-maximum likelihood on the training window)
        Variances for the test block use the ACTUAL previous-day returns (one-step-ahead), parameters are fitted
        on past data only, and scaling uses the training window only.

PART B  SAME-DAY conditional distributions on the original validation block (NOT forecasts: they use same-day
        information of other stocks). Compares
          - Constant variance
          - One-factor model: x_j | leave-one-out market return
          - DYNOTEARS graph:  x_j | same-day parents (X W + Y A), the model's own conditional distribution
        The one-factor model is the honest baseline: it tests whether the learned graph beats the simple common factor.

Scores (Hyndman & Athanasopoulos, 2021, FPP3, https://otexts.com/fpp3/ ; Sections 5.5, 5.9):
  CRPS (lower is better; closed form for a normal), negative log-likelihood, empirical coverage of central 80% and 95%
  prediction intervals (should be near 0.80 and 0.95), PIT histograms (flat = calibrated), skill score vs constant variance.
GARCH: Bollerslev, T. (1986), Journal of Econometrics 31(3), 307-327 (not covered in the FPP3 chapters read).
EWMA volatility: J.P. Morgan/Reuters (1996), RiskMetrics Technical Document, 4th ed.
DYNOTEARS: Pamfil et al. (2020), AISTATS.

Usage:  python task8_distributional.py          (run from the folder that contains data_out/)
Outputs in data_out/task8/.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm, wilcoxon

OUT = os.path.join("data_out", "task8")
rng = np.random.default_rng(0)
Z80, Z95 = norm.ppf(0.90), norm.ppf(0.975)


# ----------------------------------------------------------------------------
# Scoring a normal predictive distribution N(mu, sig^2)
# ----------------------------------------------------------------------------
def score_normal(x, mu, sig):
    z = (x - mu) / sig
    crps = sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    nll = 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2
    return dict(crps=crps, nll=nll, in80=(np.abs(z) <= Z80).astype(float),
                in95=(np.abs(z) <= Z95).astype(float), pit=norm.cdf(z))


# ----------------------------------------------------------------------------
# Volatility models (variance used for row t depends only on rows < t)
# ----------------------------------------------------------------------------
def ewma_var(Z, lam):
    s2 = np.empty_like(Z)
    s2[0] = 1.0
    for t in range(1, len(Z)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * Z[t - 1] ** 2
    return s2


def garch_nll(p, z, s0):
    om, al, be = p
    if al + be >= 0.999:
        return 1e10
    u = om + al * z[:-1] ** 2                         # u[t] for t = 1..n-1
    s2 = np.empty(len(z))
    s2[0] = s0
    s2[1:] = lfilter([1.0], [1.0, -be], u, zi=[be * s0])[0]
    return 0.5 * np.sum(np.log(s2[1:]) + z[1:] ** 2 / s2[1:])


def fit_garch(z):
    s0 = float(np.var(z))
    best = None
    for start in ([0.05, 0.08, 0.86], [0.02, 0.05, 0.93]):
        r = minimize(garch_nll, start, args=(z, s0), method="L-BFGS-B",
                     bounds=[(1e-6, 1.0), (0.0, 0.5), (0.0, 0.998)])
        if best is None or r.fun < best.fun:
            best = r
    return best.x if best.success or best.fun < 1e9 else np.array([0.05, 0.08, 0.86])


def garch_var(Z, params):
    om, al, be = params[:, 0], params[:, 1], params[:, 2]
    s2 = np.empty_like(Z)
    s2[0] = 1.0
    for t in range(1, len(Z)):
        s2[t] = om + al * Z[t - 1] ** 2 + be * s2[t - 1]
    return s2


# ----------------------------------------------------------------------------
# PART A: rolling-origin next-day distributions
# ----------------------------------------------------------------------------
def part_a(R, start, step, lam):
    n, d = R.shape
    origins = list(range(start, n - step + 1, step))
    print(f"[Part A] {len(origins)} origins from row {origins[0]} to {origins[-1]}, d = {d}", flush=True)
    rows, pits, cover = [], {}, {}
    for k, o in enumerate(origins):
        mu, sd = R[:o].mean(0), R[:o].std(0)
        Z = (R - mu) / sd
        t0 = time.time()
        params = np.array([fit_garch(Z[:o, j]) for j in range(d)])
        sig = {"Constant variance": np.ones((step, d)),
               "EWMA (0.94)": np.sqrt(ewma_var(Z[:o + step], lam)[o:o + step]),
               "GARCH(1,1)": np.sqrt(garch_var(Z[:o + step], params)[o:o + step])}
        X = Z[o:o + step]
        for name, s in sig.items():
            sc = score_normal(X, 0.0, s)
            rows.append(dict(origin=o, model=name, crps=sc["crps"].mean(), nll=sc["nll"].mean(),
                             in80=sc["in80"].mean(), in95=sc["in95"].mean()))
            pits.setdefault(name, []).append(sc["pit"].ravel())
        if k % 5 == 0:
            print(f"  origin {o} ({k + 1}/{len(origins)}), GARCH fits took {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), {m: np.concatenate(v) for m, v in pits.items()}


def summarize_blocks(df, base="Constant variance"):
    piv = df.pivot(index="origin", columns="model", values="crps")
    rows = []
    for m in piv.columns:
        sub = df[df.model == m]
        skill = 1 - piv[m] / piv[base]
        diff = piv[base] - piv[m]
        try:
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
        except ValueError:
            p = np.nan
        boot = [skill.values[rng.integers(0, len(skill), len(skill))].mean() for _ in range(2000)]
        rows.append(dict(model=m, CRPS_mean=sub.crps.mean(), CRPS_sd=sub.crps.std(ddof=1), NLL_mean=sub.nll.mean(),
                         coverage80=sub.in80.mean(), coverage95=sub.in95.mean(),
                         CRPS_skill_vs_const=skill.mean(), skill_CI95_low=np.percentile(boot, 2.5),
                         skill_CI95_high=np.percentile(boot, 97.5), wilcoxon_p=p, n_blocks=len(skill)))
    return pd.DataFrame(rows).sort_values("CRPS_mean")


# ----------------------------------------------------------------------------
# PART B: same-day conditional distributions on the validation block
# ----------------------------------------------------------------------------
def part_b(tau):
    fit_p, data_p = os.path.join("data_out", "fit_dynotears.npz"), os.path.join("data_out", "dataset.npz")
    if not (os.path.exists(fit_p) and os.path.exists(data_p)):
        print("[Part B] skipped: need data_out/fit_dynotears.npz and dataset.npz")
        return None
    F, D = np.load(fit_p, allow_pickle=True), np.load(data_p, allow_pickle=True)
    W, A = F["W"].copy(), F["A"].copy()
    W[np.abs(W) < tau] = 0
    A[np.abs(A) < tau] = 0
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    d = Xtr.shape[1]
    print(f"[Part B] validation block {Xva.shape}, DYNOTEARS edges W={int((W != 0).sum())}, A={int((A != 0).sum())}", flush=True)

    out = {}
    s_const = Xtr.std(0)
    out["Constant variance"] = (np.zeros_like(Xva), np.tile(s_const, (len(Xva), 1)))

    loo = lambda X: (X.sum(1, keepdims=True) - X) / (d - 1)         # leave-one-out market return
    Mtr, Mva = loo(Xtr), loo(Xva)
    beta = (Mtr * Xtr).sum(0) / (Mtr ** 2).sum(0)
    s_f = (Xtr - Mtr * beta).std(0)
    out["One-factor (leave-one-out market)"] = (Mva * beta, np.tile(s_f, (len(Xva), 1)))

    s_g = (Xtr - Xtr @ W - Ytr @ A).std(0)
    out["DYNOTEARS graph (same-day parents)"] = (Xva @ W + Yva @ A, np.tile(s_g, (len(Xva), 1)))

    rows, pits = [], {}
    for name, (mu, sig) in out.items():
        sc = score_normal(Xva, mu, sig)
        rows.append(dict(model=name, CRPS=sc["crps"].mean(), NLL=sc["nll"].mean(),
                         coverage80=sc["in80"].mean(), coverage95=sc["in95"].mean(),
                         RMSE=float(np.sqrt(np.mean((Xva - mu) ** 2)))))
        pits[name] = sc["pit"].ravel()
    tab = pd.DataFrame(rows)
    base = tab.loc[tab.model == "Constant variance", "CRPS"].iloc[0]
    tab["CRPS_skill_vs_const"] = 1 - tab.CRPS / base
    # paired day-level test: is the graph better than the one-factor model?
    cg = score_normal(Xva, *out["DYNOTEARS graph (same-day parents)"])["crps"].mean(1)
    cf = score_normal(Xva, *out["One-factor (leave-one-out market)"])["crps"].mean(1)
    p = wilcoxon(cf - cg).pvalue
    print(f"[Part B] paired Wilcoxon on daily CRPS, graph vs one-factor: p = {p:.4g} "
          f"(graph better on {(cg < cf).mean():.0%} of days)")
    return tab, pits, p


# ----------------------------------------------------------------------------
def pit_plot(pits, title, path):
    fig, axes = plt.subplots(1, len(pits), figsize=(4.2 * len(pits), 3.4), squeeze=False)
    for ax, (m, v) in zip(axes[0], pits.items()):
        ax.hist(v, bins=10, range=(0, 1), density=True, color="tab:blue", alpha=.8)
        ax.axhline(1, color="k", ls="--", lw=.8)
        ax.set_title(m, fontsize=9); ax.set_xlabel("PIT value"); ax.set_ylim(0, 2.2)
    axes[0][0].set_ylabel("density (flat = calibrated)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    R = pd.read_csv(os.path.join("data_out", "log_returns.csv"), index_col=0, parse_dates=True).values
    pd.set_option("display.width", 220, "display.max_columns", 30)

    blocks, pits = part_a(R, a.start, a.step, a.lam)
    blocks.to_csv(os.path.join(OUT, "task8_rolling_block_scores.csv"), index=False)
    tab = summarize_blocks(blocks)
    tab.to_csv(os.path.join(OUT, "task8_rolling_table.csv"), index=False)
    pit_plot(pits, "Part A: next-day predictive distributions, PIT histograms (rolling origins)",
             os.path.join(OUT, "task8_pit_rolling.png"))
    print("\n=== PART A: next-day predictive distributions (lower CRPS/NLL better; coverage should be 0.80 / 0.95) ===")
    print(tab.round(4).to_string(index=False))

    piv = blocks.pivot(index="origin", columns="model", values="crps")
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in piv.columns:
        ax.plot(piv.index, piv[m], label=m)
    ax.set_xlabel("Forecast origin (row index)"); ax.set_ylabel("Block mean CRPS"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task8_crps_by_origin.png"), dpi=150); plt.close(fig)

    if not a.skip_sameday:
        res = part_b(a.tau)
        if res:
            tb, pb, p = res
            tb.to_csv(os.path.join(OUT, "task8_sameday_table.csv"), index=False)
            pit_plot(pb, "Part B: same-day conditional distributions, PIT histograms (validation block)",
                     os.path.join(OUT, "task8_pit_sameday.png"))
            print("\n=== PART B: SAME-DAY conditional distributions (not forecasts) ===")
            print(tb.round(4).to_string(index=False))
    print(f"\nSaved tables and figures in {OUT}/")
    print("Notes: normal predictive distributions assume Gaussian errors (returns are heavy-tailed); "
          "training windows overlap across blocks, so block-level tests and CIs are approximate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA decay")
    ap.add_argument("--tau", type=float, default=0.05, help="threshold on saved DYNOTEARS weights (Part B)")
    ap.add_argument("--skip-sameday", action="store_true")
    main(ap.parse_args())
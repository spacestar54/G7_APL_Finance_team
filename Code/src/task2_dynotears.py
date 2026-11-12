"""
Task 2 (+ small Task 3 self-test): DYNOTEARS implemented from scratch.

Method: Pamfil et al. (2020), "DYNOTEARS: Structure Learning from Time-Series Data", AISTATS 2020, Section 2.
  Model          X = X W + Y A + Z        (W acyclic intra-slice, A inter-slice, Y = [Y_1 | ... | Y_p])
  Objective      (1/2n)||X - XW - YA||_F^2 + lam_w ||W||_1 + lam_a ||A||_1   s.t. h(W) = 0
  Acyclicity     h(W) = tr(exp(W o W)) - d   (Zheng et al., 2018, NeurIPS)
  Solver         W = W+ - W-, A = A+ - A- (all >= 0), augmented Lagrangian, L-BFGS-B (Zhu et al., 1997)
  Convention     W[i, j] is an edge i -> j (row = source, column = target), as in Figure 4 of the paper.

Usage
  python task2_dynotears.py --selftest              # gradient check + simulation checks (a minute or two)
  python task2_dynotears.py --real                  # fit on data_out/dataset.npz from Task 1
  python task2_dynotears.py --real --max-d 20       # quick debug on the first 20 stocks
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt


# ----------------------------------------------------------------------------
# STEP 1. Building blocks: acyclicity term, DAG test
# ----------------------------------------------------------------------------
def h_and_grad(W):
    """h(W) = tr(exp(W o W)) - d and its gradient (exp(W o W))^T o 2W."""
    d = W.shape[0]
    E = sla.expm(W * W)
    return np.trace(E) - d, E.T * 2.0 * W


def is_dag(W):
    """Kahn's algorithm on the nonzero pattern of W."""
    B = (W != 0).astype(int)
    indeg = B.sum(axis=0)
    stack = [i for i in range(len(B)) if indeg[i] == 0]
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        for j in np.nonzero(B[i])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    return seen == len(B)


def enforce_dag(W):
    """
    h(W) ~ 0 only holds approximately, so very weak cycles can survive thresholding.
    Repeatedly remove the smallest-magnitude edge that lies on a cycle (inside a strongly
    connected component) until W is a true DAG. Returns (W, number_of_edges_removed).
    """
    from scipy.sparse.csgraph import connected_components
    W = W.copy()
    removed = 0
    while True:
        B = W != 0
        _, lab = connected_components(B, directed=True, connection="strong")
        on_cycle = B & (lab[:, None] == lab[None, :])
        if not on_cycle.any():
            return W, removed
        mags = np.where(on_cycle, np.abs(W), np.inf)
        i, j = np.unravel_index(np.argmin(mags), W.shape)
        W[i, j] = 0.0
        removed += 1


# ----------------------------------------------------------------------------
# STEP 2. Objective and gradient on the split variable v = [W+, A+, W-, A-]
# ----------------------------------------------------------------------------
def _unpack(v, d, q):
    nW, nA = d * d, q * d
    k = nW + nA
    W = (v[:nW] - v[k:k + nW]).reshape(d, d)
    A = (v[nW:k] - v[k + nW:]).reshape(q, d)
    return W, A


def make_objective(X, Y, lam_w, lam_a):
    n, d = X.shape
    q = Y.shape[1]
    nW, nA = d * d, q * d
    k = nW + nA

    def obj(v, rho, alpha):
        W, A = _unpack(v, d, q)
        R = X - X @ W - Y @ A
        h, Gh = h_and_grad(W)
        f = (0.5 / n) * np.sum(R ** 2) + 0.5 * rho * h * h + alpha * h
        f += lam_w * (v[:nW].sum() + v[k:k + nW].sum())
        f += lam_a * (v[nW:k].sum() + v[k + nW:].sum())
        gW = -(X.T @ R) / n + (rho * h + alpha) * Gh
        gA = -(Y.T @ R) / n
        g = np.concatenate([gW.ravel() + lam_w, gA.ravel() + lam_a,
                            -gW.ravel() + lam_w, -gA.ravel() + lam_a])
        return f, g

    return obj


# ----------------------------------------------------------------------------
# STEP 3. Augmented Lagrangian outer loop with L-BFGS-B inner solves
# ----------------------------------------------------------------------------
def fit_dynotears(X, Y, lam_w=0.1, lam_a=0.1, tau_w=0.0, tau_a=0.0,
                  h_tol=1e-8, rho_max=1e16, max_outer=100, verbose=False):
    """
    Returns W (d x d), A (q x d), info dict.
    The paper does not list the (rho, alpha) schedule; this follows NOTEARS (Zheng et al., 2018):
    rho starts at 1, is multiplied by 10 until h shrinks by a factor of 4, alpha += rho * h.
    """
    t0 = time.time()
    n, d = X.shape
    q = Y.shape[1]
    nW, nA = d * d, q * d
    k = nW + nA
    obj = make_objective(X, Y, lam_w, lam_a)

    bnds = [(0, None)] * (2 * k)
    for i in range(d):                      # no self-loops in W
        bnds[i * d + i] = (0, 0)
        bnds[k + i * d + i] = (0, 0)

    v = np.zeros(2 * k)
    rho, alpha, h = 1.0, 0.0, np.inf
    outer = 0
    while rho < rho_max and outer < max_outer:
        outer += 1
        while rho < rho_max:
            sol = sopt.minimize(obj, v, args=(rho, alpha), method="L-BFGS-B",
                                jac=True, bounds=bnds)
            h_new = h_and_grad(_unpack(sol.x, d, q)[0])[0]
            if verbose:
                print(f"    inner solve: rho={rho:.0e} iters={sol.nit} fevals={sol.nfev} "
                      f"h={h_new:.2e} elapsed={time.time() - t0:.0f}s", flush=True)
            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break
        v, h = sol.x, h_new
        alpha += rho * h
        if verbose:
            print(f"  outer {outer}: rho={rho:.0e} alpha={alpha:.3g} h={h:.3e}")
        if h <= h_tol:
            break

    W, A = _unpack(v, d, q)
    W_raw, A_raw = W.copy(), A.copy()
    W[np.abs(W) < tau_w] = 0.0
    A[np.abs(A) < tau_a] = 0.0
    W, n_removed = enforce_dag(W)
    info = {"edges_removed_for_dag": n_removed, "h_before_threshold": float(h), "h_after_threshold": float(h_and_grad(W)[0]),
            "is_dag": bool(is_dag(W)), "outer_iters": outer, "rho_final": rho,
            "seconds": time.time() - t0, "W_raw": W_raw, "A_raw": A_raw}
    return W, A, info


# ----------------------------------------------------------------------------
# STEP 4. Losses and predictions
# ----------------------------------------------------------------------------
def structural_loss(X, Y, W, A):
    """Frobenius norm ||X - XW - YA||_F (the paper's validation loss, Figure 12 caption)."""
    return float(np.linalg.norm(X - X @ W - Y @ A))


def forecast(Y, W, A):
    """Reduced form X_hat = Y A (I - W)^(-1): uses only past information."""
    d = W.shape[0]
    return Y @ A @ np.linalg.inv(np.eye(d) - W)


# ----------------------------------------------------------------------------
# STEP 5. Simulator (Pamfil et al., 2020, Appendix B.4) and metrics
# ----------------------------------------------------------------------------
def _signed_uniform(rng, lo, hi, size):
    return rng.uniform(lo, hi, size) * rng.choice([-1.0, 1.0], size)


def simulate(d=5, n=500, p=1, k_w=4.0, k_a=1.0, eta=1.5, noise="gaussian", seed=0):
    rng = np.random.default_rng(seed)
    # intra-slice ER DAG: lower-triangular Bernoulli(k/(d-1)), random node order
    prob = min(1.0, k_w / (d - 1))
    Wb = np.tril(rng.random((d, d)) < prob, -1).astype(float)
    perm = rng.permutation(d)
    Wb = Wb[np.ix_(perm, perm)]
    W = Wb * _signed_uniform(rng, 0.5, 2.0, (d, d))
    # inter-slice directed ER, weights decay with lag
    As = []
    for i in range(1, p + 1):
        alpha = 1.0 / eta ** (i - 1)
        Ab = (rng.random((d, d)) < k_a / d).astype(float)
        As.append(Ab * _signed_uniform(rng, 0.3 * alpha, 0.5 * alpha, (d, d)))
    # stability of the implied VAR: B_i = A_i (I - W)^-1
    inv = np.linalg.inv(np.eye(d) - W)
    comp = np.zeros((p * d, p * d))
    for i, Ai in enumerate(As):
        comp[:d, i * d:(i + 1) * d] = Ai @ inv
    if p > 1:
        comp[d:, :-d] = np.eye((p - 1) * d)
    if np.max(np.abs(np.linalg.eigvals(comp))) >= 0.95:
        return simulate(d, n, p, k_w, k_a, eta, noise, seed + 1000)   # resample unstable draw
    # generate: x_t = (sum_i x_{t-i} A_i + z_t)(I - W)^-1
    burn, T = 100, n + p
    x = np.zeros((burn + T, d))
    for t in range(p, burn + T):
        z = rng.normal(size=d) if noise == "gaussian" else rng.exponential(size=d) - 1.0
        x[t] = (sum(x[t - i - 1] @ As[i] for i in range(p)) + z) @ inv
    x = x[burn:]
    X = x[p:]
    Y = np.hstack([x[p - i: T - i] for i in range(1, p + 1)])
    return X, Y, W, np.vstack(As)


def edge_metrics(T, E):
    """Precision / recall / F1 on directed edges, plus SHD (a reversal counts once)."""
    Tb, Eb = (T != 0), (E != 0)
    tp = int((Tb & Eb).sum())
    fp = int((~Tb & Eb).sum())
    fn = int((Tb & ~Eb).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    shd = fp + fn
    if T.shape[0] == T.shape[1]:            # square: reversal = 1 error instead of 2
        shd -= int((Tb & ~Eb & Eb.T & ~Tb.T).sum())
    return {"precision": prec, "recall": rec, "f1": f1, "shd": shd}


# ----------------------------------------------------------------------------
# STEP 6. Self-test: gradient check, h checks, simulation
# ----------------------------------------------------------------------------
def selftest():
    print("=== 1. h(W) checks ===")
    Wu = np.triu(np.ones((4, 4)), 1) * 0.7
    Wc = Wu.copy()
    Wc[3, 0] = 0.7                                  # closes a cycle
    h_dag, h_cyc = h_and_grad(Wu)[0], h_and_grad(Wc)[0]
    print(f"h(DAG) = {h_dag:.2e} (should be ~0), h(with cycle) = {h_cyc:.3f} (should be > 0)")
    assert abs(h_dag) < 1e-9 and h_cyc > 1e-3 and is_dag(Wu) and not is_dag(Wc)

    print("\n=== 2. Gradient check (relative error should be ~1e-6 or smaller) ===")
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(60, 4)), rng.normal(size=(60, 8))
    obj = make_objective(X, Y, 0.1, 0.1)
    v0 = rng.uniform(0.0, 0.3, 2 * (16 + 32))
    f = lambda v: obj(v, 2.0, 0.5)[0]
    g = lambda v: obj(v, 2.0, 0.5)[1]
    err = sopt.check_grad(f, g, v0)
    rel = err / np.linalg.norm(g(v0))
    print(f"check_grad error = {err:.2e}, relative = {rel:.2e}")
    assert rel < 1e-5, "Gradient is wrong"

    print("\n=== 3. Simulation, d=5, n=500, p=1 (paper Fig. 3: F1 close to 1) ===")
    rows = []
    for seed in range(5):
        X, Y, Wt, At = simulate(d=5, n=500, p=1, seed=seed)
        W, A, info = fit_dynotears(X, Y, lam_w=0.05, lam_a=0.05, tau_w=0.3, tau_a=0.1)
        mw, ma = edge_metrics(Wt, W), edge_metrics(At, A)
        rows.append((mw["f1"], ma["f1"], mw["shd"], ma["shd"]))
        print(f" seed {seed}: F1(W)={mw['f1']:.2f} F1(A)={ma['f1']:.2f} SHD(W)={mw['shd']} "
              f"SHD(A)={ma['shd']} DAG={info['is_dag']} {info['seconds']:.1f}s")
        assert info["is_dag"]
    r = np.array(rows)
    print(f" mean F1(W)={r[:, 0].mean():.2f} +/- {r[:, 0].std():.2f},  "
          f"F1(A)={r[:, 1].mean():.2f} +/- {r[:, 1].std():.2f}")

    print("\n=== 4. Figure-2-style replica: d=5, n=500, p=3, eta=1.5 ===")
    X, Y, Wt, At = simulate(d=5, n=500, p=3, k_w=4, k_a=1, eta=1.5, seed=1)
    W, A, info = fit_dynotears(X, Y, lam_w=0.05, lam_a=0.05, tau_w=0.01, tau_a=0.01)
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print("True W:\n", Wt, "\nEstimated W:\n", W)
    print("F1(W) =", round(edge_metrics(Wt, W)["f1"], 2), " F1(A) =", round(edge_metrics(At, A)["f1"], 2),
          " Frobenius error W =", round(float(np.linalg.norm(Wt - W)), 3))
    print("\nSelf-test finished. If steps 1-2 pass and F1 is high in step 3, the optimizer is working.")


# ----------------------------------------------------------------------------
# STEP 7. Real data: fit on dataset.npz from Task 1
# ----------------------------------------------------------------------------
def run_real(args):
    D = np.load(os.path.join("data_out", "dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Xva, Yva = D["X_train"], D["Y_train"], D["X_val"], D["Y_val"]
    tickers = list(D["tickers"])
    if args.max_d:
        m = args.max_d
        Xtr, Xva = Xtr[:, :m], Xva[:, :m]
        Ytr, Yva = Ytr[:, :m], Yva[:, :m]            # p = 1, so Y has d columns
        tickers = tickers[:m]
    d = Xtr.shape[1]
    print(f"Fitting on {Xtr.shape[0]} train rows, d = {d}, lam_w={args.lam_w}, lam_a={args.lam_a}, "
          f"tau_w={args.tau_w}, tau_a={args.tau_a}")
    print(f"Zero-model validation loss ||X_val||_F = {np.linalg.norm(Xva):.3f}")

    print("Progress lines appear after each inner solve. A full d~93 fit can take several minutes.", flush=True)
    out = {}
    for name, (Xt, Yt, Xv, Yv) in {
        "dynotears": (Xtr, Ytr, Xva, Yva),
        "static_notears": (Xtr, np.zeros((Xtr.shape[0], 0)), Xva, np.zeros((Xva.shape[0], 0))),
    }.items():
        print(f"\n[{name}] fitting ...", flush=True)
        W, A, info = fit_dynotears(Xt, Yt, args.lam_w, args.lam_a, args.tau_w, args.tau_a, verbose=not args.quiet)
        loss = structural_loss(Xv, Yv, W, A)
        nzW, nzA = int((W != 0).sum()), int((A != 0).sum())
        pos = float((W[W != 0] > 0).mean()) if nzW else float("nan")
        print(f"\n[{name}] val loss = {loss:.3f} | edges W = {nzW}, A = {nzA} | positive W = {pos:.1%} "
              f"| DAG = {info['is_dag']} | h = {info['h_after_threshold']:.1e} | {info['seconds']:.0f}s")
        nz = np.abs(W[W != 0])
        if nz.size:
            print(f"    |W| of kept edges: min={nz.min():.3f} median={np.median(nz):.3f} max={nz.max():.3f}; "
                  f"edges removed to enforce DAG: {info['edges_removed_for_dag']}")
        if nzA:
            print(f"    A edge weights: {np.round(A[A != 0], 3).tolist()}")
        print("    threshold sensitivity (tau_w=tau_a): tau, edges W, edges A, val loss")
        for tau in (0.01, 0.05, 0.1, 0.2):
            Wt, _ = enforce_dag(np.where(np.abs(info['W_raw']) >= tau, info['W_raw'], 0.0))
            At = np.where(np.abs(info['A_raw']) >= tau, info['A_raw'], 0.0)
            print(f"      {tau:<5} {int((Wt != 0).sum()):>5} {int((At != 0).sum()):>5} "
                  f"{structural_loss(Xv, Yv, Wt, At):.3f}")
        out[name] = {"val_loss": loss, "nnz_W": nzW, "nnz_A": nzA, "pos_frac_W": pos, "seconds": info["seconds"]}
        np.savez(os.path.join("data_out", f"fit_{name}.npz"), W=W, A=A, tickers=np.array(tickers))
    print("\nPaper reference (Pamfil et al., 2020, Fig. 12): DYNOTEARS 167.387, static NOTEARS 167.784, "
          "zero model 217.596 (97 stocks; yours has a different d, so compare patterns, not decimals).")
    with open(os.path.join("data_out", "task2_real_fit.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-d", type=int, default=0, help="debug: use only the first k stocks")
    ap.add_argument("--lam-w", type=float, default=0.1)
    ap.add_argument("--lam-a", type=float, default=0.1)
    ap.add_argument("--tau-w", type=float, default=0.01, help="not given in the paper for S&P; choose and report")
    ap.add_argument("--tau-a", type=float, default=0.01)
    ap.add_argument("--quiet", action="store_true", help="hide progress output")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.real:
        run_real(a)
    if not (a.selftest or a.real):
        ap.print_help()
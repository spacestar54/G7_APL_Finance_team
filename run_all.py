"""
run_all.py : regenerate every number, table and figure of the M4 project, in the right order.

Run from the Finance folder (all scripts side by side; outputs go to ./data_out):
    python run_all.py --dry-run            # print the plan, run nothing
    python run_all.py --skip-slow          # quick SMOKE TEST (small simulation, reduced lambda grid, no bootstrap/ARIMA/rolling);
                                           # numbers from this mode are NOT the reported results
    python run_all.py                      # everything (several hours; the bootstrap and the lambda grid are the slow steps)
    python run_all.py --from-step 12       # resume from step 12
    python run_all.py --only 16 17         # run selected steps only
    python run_all.py --keep-going         # do not stop when a step fails

Notes
  * Step 1 downloads prices with yfinance, so it needs internet. Without internet, put prices in a CSV and use
    python task1_data_pipeline.py --prices YOUR.csv
  * The bootstrap (step 13) must run BEFORE the selected graph is adopted (step 17): its reference graph and its
    refits use the earlier setting (lambda 0.1, tau 0.05). It is reported as a stability analysis of that setting.
  * Step 17 makes the graph selected on an inner split of the training block (lambda 0.03, tau 0.1) the final one
    (data_out/fit_sector_residual.npz). Steps 18 onward therefore use the final graph.
  * Every result is same-day conditional or rolling-origin next-day, as labelled in each script's output.
"""
import argparse
import datetime
import importlib
import os
import platform
import shutil
import subprocess
import sys
import time

PY = sys.executable
LOGDIR = os.path.join("data_out", "logs")


def adopt_selected_graph():
    """Back up the earlier residual graph (once) and make the inner-split graph the final one."""
    src = os.path.join("data_out", "fit_sector_residual_inner.npz")
    dst = os.path.join("data_out", "fit_sector_residual.npz")
    bak = os.path.join("data_out", "fit_sector_residual_lam0.1_tau0.05.npz")
    if not os.path.exists(src):
        raise FileNotFoundError(src + " missing: run step 16 (task16_inner_selection.py) first")
    if os.path.exists(dst) and not os.path.exists(bak):
        shutil.copy(dst, bak)
        print("backed up", dst, "->", bak)
    shutil.copy(src, dst)
    print("final residual graph is now", src)


def write_environment():
    os.makedirs("data_out", exist_ok=True)
    lines = [f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"python: {sys.version.split()[0]}", f"platform: {platform.platform()}"]
    for m in ["numpy", "pandas", "scipy", "statsmodels", "yfinance", "matplotlib", "sklearn", "networkx"]:
        try:
            lines.append(f"{m}: {importlib.import_module(m).__version__}")
        except Exception as e:                                   # missing package is reported, not fatal
            lines.append(f"{m}: not importable ({type(e).__name__})")
    with open(os.path.join("data_out", "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# (id, label, command list or python callable, slow?, quick-mode replacement command or None)
def plan(quick):
    s = [
        (1, "Data pipeline (download, clean, ADF, split)", [PY, "task1_data_pipeline.py"], False),
        (2, "DYNOTEARS self-test", [PY, "task2_dynotears.py", "--selftest"], False),
        (3, "DYNOTEARS + static NOTEARS on S&P 100 (real fit)", [PY, "task2_dynotears.py", "--real"], False),
        (4, "Simulation study (24 settings, 10 seeds, Wilcoxon)",
         [PY, "task3_simulation.py"] + (["--quick"] if quick else []), True),
        (5, "First forecast baselines and RMSE", [PY, "task5_6_baselines_rmse.py"], False),
        (6, "Base network figures (raw-return graph)", [PY, "plot_basenet.py"], False),
        (7, "Rolling-origin forecasts, skill scores, Ljung-Box", [PY, "task7_rolling_eval.py"], True),
        (8, "Next-day and same-day predictive distributions (CRPS, NLL, PIT)", [PY, "task8_distributional.py"], False),
        (9, "Factor ablation" + ("" if quick else ", lambda sensitivity, lag order"),
         [PY, "task9_factor_ablation.py"] + (["--run", "factor"] if quick else []), False),
        (10, "Same-day factor baselines vs graphs", [PY, "task10_factor_baselines.py"], False),
        (11, "Residual graph, earlier setting (lambda 0.1, tau 0.05)", [PY, "task11_sector_residual_graph.py"], False),
        (12, "ARIMA / ARIMAX / ARIMA + GARCH", [PY, "task13_arima.py"], True),
        (13, "Block-bootstrap edge stability (earlier setting; ~45 min)",
         [PY, "task12_bootstrap_stability.py", "--B", "30", "--resume"], True),
        (14, "(reserved)", None, False),
        (15, "(reserved)", None, False),
        (16, "Inner-split selection of lambda, tau (training block only)" + (" [QUICK smoke test: reduced grid]" if quick else ""),
         [PY, "task16_inner_selection.py"] + (["--lams", "0.1", "0.3", "--taus", "0.05"] if quick else []), False),
        (17, "Adopt the selected graph as the final residual graph", adopt_selected_graph, False),
        (18, "Event probabilities: Brier, log-loss, ECE, AUC, reliability", [PY, "task14_brier_calibration.py"], False),
        (19, "Error analysis", [PY, "task15_error_analysis.py"], False),
        (20, "Residual-graph figures (final graph)",
         [PY, "plot_basenet.py", "--fit", os.path.join("data_out", "fit_sector_residual.npz"),
          "--tag", "sector_residual", "--tau", "0.1"], False),
    ]
    return [x for x in s if x[2] is not None]


def run_step(sid, label, action, log):
    t0 = time.time()
    print(f"\n===== STEP {sid}: {label} =====", flush=True)
    if callable(action):
        action()
        return 0, time.time() - t0
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(action, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        p.wait()
    return p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-slow", action="store_true", help="skip slow steps; smaller simulation")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="+")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()

    if not os.path.exists("task2_dynotears.py"):
        sys.exit("Run this from the Finance folder (task2_dynotears.py must be in the current directory).")
    os.makedirs(LOGDIR, exist_ok=True)
    steps = plan(quick=a.skip_slow)
    todo = []
    for sid, label, action, slow in steps:
        if a.only and sid not in a.only:
            continue
        if sid < a.from_step:
            continue
        if a.skip_slow and slow and sid != 4:                    # step 4 runs in --quick mode instead of being skipped
            print(f"skip (slow): step {sid} {label}")
            continue
        if isinstance(action, list) and not os.path.exists(action[1]):
            print(f"WARNING: step {sid} needs {action[1]}, which is not in this folder; it will be skipped.")
            continue
        todo.append((sid, label, action))

    if a.dry_run:
        for sid, label, action in todo:
            print(f"step {sid}: {label}\n    -> {' '.join(action) if isinstance(action, list) else action.__name__}")
        return
    write_environment()
    summary, t_all = [], time.time()
    for sid, label, action in todo:
        rc, secs = run_step(sid, label, action, os.path.join(LOGDIR, f"step{sid:02d}.log"))
        summary.append((sid, label, rc, secs))
        if rc != 0 and not a.keep_going:
            print(f"\nStep {sid} failed (exit code {rc}); see {LOGDIR}. Stopping (use --keep-going to continue).")
            break
    print("\n===== SUMMARY =====")
    for sid, label, rc, secs in summary:
        print(f"step {sid:>2}  {'OK  ' if rc == 0 else 'FAIL'}  {secs/60:6.1f} min  {label}")
    print(f"total {(time.time() - t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
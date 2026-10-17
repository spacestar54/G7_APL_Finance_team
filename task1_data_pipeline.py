"""
Task 1: Download and preprocess S&P 100 data for DYNOTEARS (CSE 516, Applied AI, Finance)

Follows Pamfil et al. (2020), "DYNOTEARS: Structure Learning from Time-Series Data",
AISTATS 2020, Section 4.1 and Appendix C.1:
  - daily closing prices, 1 Jul 2014 to 30 Jun 2019 (inclusive)
  - log-returns, ADF stationarity test, standardize to zero mean / unit variance
  - paper result: n = 1257 samples, d = 97 stocks
  - last 400 points held out for validation, 2 points discarded between train and validation
  - autoregressive order p = 1
  - zero-model validation loss (Frobenius norm of validation data) reported as 217.596 (Figure 12)

Usage
  python task1_data_pipeline.py                      # download with yfinance
  python task1_data_pipeline.py --adjusted           # use Adj Close instead of Close
  python task1_data_pipeline.py --prices my.csv      # use your own price table (Date index, ticker columns)
  python task1_data_pipeline.py --scale train        # standardize with training statistics only
  python task1_data_pipeline.py --synthetic          # offline smoke test with fake prices (NOT real results)

Outputs (in ./data_out)
  prices_raw.csv, prices_clean.csv, log_returns.csv, returns_std.csv,
  dataset.npz (X_train, Y_train, X_val, Y_val, tickers), task1_report.json
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# STEP 0. Configuration
# ----------------------------------------------------------------------------
START = "2014-07-01"
END_INCLUSIVE = "2019-06-30"
VAL_LEN = 400          # last 400 points held out (about 32%)
GAP = 2                # points discarded between train and validation
P = 1                  # autoregressive order
PAPER_N = 1257
PAPER_D = 97
PAPER_ZERO_LOSS = 217.596   # Figure 12 caption: Frobenius norm of the validation data

# The 97 tickers are the axis labels of Figure 4 of the paper (indices 0 to 96), in that order.
PAPER_TICKERS = (
    "COST DIS CMCSA NKE SBUX CHTR F AMZN MCD TGT LOW GM HD BKNG CL PG PM MDLZ MO WBA "
    "PEP WMT KO KMI CVX SLB OXY XOM COP BK MET C ALL JPM MS SPG AIG AXP USB WFC BLK COF "
    "BAC GS LLY AMGN MRK AGN ABT ABBV PFE MDT BIIB CVS JNJ DHR BMY UNH CELG GILD DD UTX "
    "RTN EMR UPS CAT GD GE MMM UNP HON BA LMT FDX QCOM TXN V AAPL NVDA ACN ADBE CSCO FB "
    "ORCL GOOGL GOOG MA MSFT NFLX INTC IBM T VZ EXC DUK SO NEE"
).split()

# Some companies were renamed after 2019. Yahoo serves history under the new symbol.
# CELG, AGN and RTN were acquired or merged and are likely to be unavailable.
# DD in 2014-2019 was the old DuPont / DowDuPont; the current DD symbol is a different company.
# These aliases are a convenience; check each one and report what you used.
YAHOO_ALIAS = {"FB": "META", "UTX": "RTX", "BK": "BNY"}

OUT_DIR = "data_out"


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# STEP 1. Get prices (download, load from CSV, or synthetic smoke test)
# ----------------------------------------------------------------------------
def download_prices(tickers, adjusted):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: pip install yfinance")

    yahoo_syms = [YAHOO_ALIAS.get(t, t) for t in tickers]
    end_exclusive = (pd.Timestamp(END_INCLUSIVE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"Downloading {len(yahoo_syms)} symbols from Yahoo ({START} to {END_INCLUSIVE}) ...")
    data = yf.download(
        yahoo_syms, start=START, end=end_exclusive,
        auto_adjust=False, progress=False, group_by="column",
    )
    field = "Adj Close" if adjusted else "Close"
    prices = data[field].copy()
    # map symbols back to the paper's tickers so column names match Figure 4
    back = {YAHOO_ALIAS.get(t, t): t for t in tickers}
    prices.columns = [back.get(c, c) for c in prices.columns]
    return prices


def synthetic_prices(tickers, seed=0):
    """Fake geometric random walks for an offline smoke test. NOT real data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(START, END_INCLUSIVE)
    rets = rng.normal(0.0004, 0.015, size=(len(idx), len(tickers)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def get_prices(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "prices_raw.csv")
    if args.synthetic:
        log("SYNTHETIC MODE: results are fake and only test the code.")
        prices = synthetic_prices(PAPER_TICKERS)
    elif args.prices:
        log(f"Loading prices from {args.prices}")
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    else:
        prices = download_prices(PAPER_TICKERS, args.adjusted)
    prices = prices.sort_index()
    prices = prices.loc[START:END_INCLUSIVE]
    prices.to_csv(raw_path)
    log(f"Raw price table: {prices.shape[0]} dates x {prices.shape[1]} symbols -> {raw_path}")
    return prices


# ----------------------------------------------------------------------------
# STEP 2. Clean: report and drop tickers that failed or have missing values
# ----------------------------------------------------------------------------
def clean_prices(prices, max_ffill=0):
    report = {}
    requested = list(PAPER_TICKERS)
    present = [t for t in requested if t in prices.columns]
    missing_cols = [t for t in requested if t not in prices.columns]
    prices = prices[present]

    all_nan = [t for t in present if prices[t].isna().all()]
    prices = prices.drop(columns=all_nan)

    prices = prices.dropna(how="all")          # dates where nothing traded
    if max_ffill > 0:
        prices = prices.ffill(limit=max_ffill)
    n_nan = prices.isna().sum()
    partial = [t for t in prices.columns if n_nan[t] > 0]
    prices = prices.drop(columns=partial)

    report["requested"] = len(requested)
    report["no_data_returned"] = missing_cols + all_nan
    report["dropped_for_missing_values"] = partial
    report["kept"] = list(prices.columns)
    log(f"Tickers requested: {len(requested)}; kept: {prices.shape[1]}")
    if report["no_data_returned"]:
        log(f"  No data returned: {report['no_data_returned']}")
    if partial:
        log(f"  Dropped for missing values: {partial}")
    if (prices <= 0).any().any():
        bad = list(prices.columns[(prices <= 0).any()])
        sys.exit(f"Non-positive prices found for {bad}; cannot take logs.")
    prices.to_csv(os.path.join(OUT_DIR, "prices_clean.csv"))
    return prices, report


# ----------------------------------------------------------------------------
# STEP 3. Log-returns: r_t = log P_t - log P_{t-1}
# ----------------------------------------------------------------------------
def log_returns(prices):
    rets = np.log(prices).diff().dropna(how="any")
    rets.to_csv(os.path.join(OUT_DIR, "log_returns.csv"))
    log(f"Log-returns: {rets.shape[0]} rows x {rets.shape[1]} stocks")
    return rets


# ----------------------------------------------------------------------------
# STEP 4. Augmented Dickey-Fuller stationarity test
# ----------------------------------------------------------------------------
def adf_summary(df, name):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log("statsmodels not installed; skipping ADF. Run: pip install statsmodels")
        return None
    pvals = {c: adfuller(df[c].values, autolag="AIC")[1] for c in df.columns}
    s = pd.Series(pvals)
    out = {
        "series": name,
        "n_series": int(len(s)),
        "n_p_below_0.01": int((s < 0.01).sum()),
        "max_p": float(s.max()),
        "non_stationary_at_0.01": list(s[s >= 0.01].index),
    }
    log(f"ADF on {name}: {out['n_p_below_0.01']}/{out['n_series']} series have p < 0.01 "
        f"(max p = {out['max_p']:.4g})")
    return out


# ----------------------------------------------------------------------------
# STEP 5. Split into train / gap / validation
# ----------------------------------------------------------------------------
def split_indices(n_rows, val_len=VAL_LEN, gap=GAP):
    val_start = n_rows - val_len
    train_end = val_start - gap          # train rows are [0, train_end)
    if train_end <= 0:
        sys.exit("Not enough rows for the requested split.")
    return train_end, val_start


# ----------------------------------------------------------------------------
# STEP 6. Standardize each stock (zero mean, unit variance)
# ----------------------------------------------------------------------------
def standardize(rets, mode, train_end):
    """
    mode 'full' : statistics from the whole sample (paper-style replication; leaks validation
                  information into the scaling).
    mode 'train': statistics from training rows only (leak-free variant).
    """
    ref = rets if mode == "full" else rets.iloc[:train_end]
    mu, sd = ref.mean(), ref.std(ddof=0)
    if (sd == 0).any():
        sys.exit(f"Zero variance in: {list(sd[sd == 0].index)}")
    z = (rets - mu) / sd
    z.to_csv(os.path.join(OUT_DIR, "returns_std.csv"))
    return z


# ----------------------------------------------------------------------------
# STEP 7. Build X (current) and Y = [Y_1 | ... | Y_p] (lags) inside one split
# ----------------------------------------------------------------------------
def make_xy(block, p=P):
    """block: (m, d) array. Lags are built inside the block so no split leaks into another."""
    m, d = block.shape
    if m <= p:
        sys.exit("Block shorter than the lag order.")
    X = block[p:]
    Y = np.hstack([block[p - i: m - i] for i in range(1, p + 1)])
    assert X.shape == (m - p, d) and Y.shape == (m - p, p * d)
    return X, Y


# ----------------------------------------------------------------------------
# STEP 8. Checks against the paper
# ----------------------------------------------------------------------------
def checks(rets_std, train_end, val_start, X_tr, Y_tr, X_va, Y_va):
    n_rows, d = rets_std.shape
    val_block = rets_std.values[val_start:]
    zero_loss_all = float(np.linalg.norm(val_block))        # all 400 validation rows
    zero_loss_lag = float(np.linalg.norm(X_va))             # rows left after building lags
    res = {
        "n_return_rows": int(n_rows),
        "paper_n": PAPER_N,
        "d": int(d),
        "paper_d": PAPER_D,
        "train_rows_before_lag": int(train_end),
        "val_rows_before_lag": int(len(val_block)),
        "X_train_shape": list(X_tr.shape),
        "Y_train_shape": list(Y_tr.shape),
        "X_val_shape": list(X_va.shape),
        "Y_val_shape": list(Y_va.shape),
        "zero_model_loss_400_rows": zero_loss_all,
        "zero_model_loss_after_lag": zero_loss_lag,
        "paper_zero_model_loss": PAPER_ZERO_LOSS,
    }
    log("\n=== Checks against Pamfil et al. (2020) ===")
    log(f"Return rows : {n_rows}   (paper n = {PAPER_N})")
    log(f"Stocks d    : {d}   (paper d = {PAPER_D})")
    log(f"Train rows  : {train_end} before lag, Y_train {Y_tr.shape}")
    log(f"Valid rows  : {len(val_block)} before lag, X_val {X_va.shape}")
    log(f"||X_val||_F : {zero_loss_all:.3f} (all 400 rows), {zero_loss_lag:.3f} (after lag)")
    log(f"              paper's zero-model loss = {PAPER_ZERO_LOSS}")
    log("A close match suggests your data, scaling and split resemble the paper's. "
        "A different value means different data, tickers or scaling; report it honestly.")
    return res


# ----------------------------------------------------------------------------
# MAIN: run steps in order
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjusted", action="store_true", help="use Adj Close instead of Close")
    ap.add_argument("--prices", type=str, default=None, help="CSV with Date index and ticker columns")
    ap.add_argument("--scale", choices=["full", "train"], default="full",
                    help="standardize with full-sample (paper) or training-only statistics")
    ap.add_argument("--ffill", type=int, default=0, help="forward-fill gaps up to this many days")
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test with fake data")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"args": vars(args)}

    prices = get_prices(args)                                   # step 1
    prices, clean_rep = clean_prices(prices, args.ffill)        # step 2
    report["cleaning"] = clean_rep
    rets = log_returns(prices)                                  # step 3

    report["adf_log_prices"] = adf_summary(np.log(prices), "log-prices")   # step 4
    report["adf_log_returns"] = adf_summary(rets, "log-returns")

    train_end, val_start = split_indices(len(rets))             # step 5
    rets_std = standardize(rets, args.scale, train_end)         # step 6

    arr = rets_std.values
    X_tr, Y_tr = make_xy(arr[:train_end])                       # step 7
    X_va, Y_va = make_xy(arr[val_start:])

    report["checks"] = checks(rets_std, train_end, val_start,   # step 8
                              X_tr, Y_tr, X_va, Y_va)
    report["train_dates"] = [str(rets_std.index[0].date()), str(rets_std.index[train_end - 1].date())]
    report["val_dates"] = [str(rets_std.index[val_start].date()), str(rets_std.index[-1].date())]
    report["synthetic"] = bool(args.synthetic)

    np.savez(os.path.join(OUT_DIR, "dataset.npz"),
             X_train=X_tr, Y_train=Y_tr, X_val=X_va, Y_val=Y_va,
             tickers=np.array(rets_std.columns))
    with open(os.path.join(OUT_DIR, "task1_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nSaved dataset.npz and task1_report.json in ./{OUT_DIR}")


if __name__ == "__main__":
    main()
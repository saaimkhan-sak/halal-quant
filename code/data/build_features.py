import os
from pathlib import Path

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
DERIVED_DIR = DATA_DIR / "derived"

FUNDAMENTALS_FILE = DERIVED_DIR / "fundamentals_sp.csv"
OUT_FILE = DERIVED_DIR / "features_daily.csv"


def load_fundamentals():
    if not FUNDAMENTALS_FILE.exists():
        raise SystemExit(f"Missing fundamentals file: {FUNDAMENTALS_FILE}")

    f = pd.read_csv(FUNDAMENTALS_FILE)

    required = ["ticker", "total_debt", "shares_outstanding", "total_revenue"]
    for col in required:
        if col not in f.columns:
            raise SystemExit(f"Column '{col}' missing in fundamentals")

    f = f.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    f["ticker"] = f["ticker"].astype(str).str.upper().str.strip()

    # Basic ratios
    f["debt_to_revenue"] = f["total_debt"] / f["total_revenue"]
    f["revenue_per_share"] = f["total_revenue"] / f["shares_outstanding"]

    return f


def load_price_history(ticker: str) -> pd.DataFrame:
    path = MARKET_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price file for {ticker}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing 'Date' column in {path}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("Date").reset_index(drop=True)

    # normalize names
    df.columns = [c.lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"{ticker}: missing close column")

    # Ensure all price columns exist
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Base returns
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)

    # Rolling vol
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # Rolling z-score
    roll_mean = df["close"].rolling(20).mean()
    roll_std = df["close"].rolling(20).std()
    df["zscore_close_20d"] = (df["close"] - roll_mean) / roll_std

    # RSI
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    # MACD
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["atr_14"] = atr.average_true_range()

    # Volume-based features
    # Avoid division by zero by replacing 0 std with NaN, then compute z-score and fill NaN with 0
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std().replace(0, np.nan)
    df["vol_zscore_20d"] = (df["volume"] - vol_mean) / vol_std
    df["vol_zscore_20d"] = df["vol_zscore_20d"].fillna(0.0)

    # Only drop rows where *core* indicators are missing (warm-up period)
    core_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "zscore_close_20d", "rsi_14"]
    df = df.dropna(subset=core_cols).reset_index(drop=True)

    return df

def build_features():
    fundamentals = load_fundamentals()
    all_rows = []

    tickers = list(fundamentals["ticker"].unique())
    print(f"Building feature set for {len(tickers)} tickers…")

    for ticker in sorted(tickers):
        try:
            px = load_price_history(ticker)
        except Exception as e:
            print(f"[SKIP] {ticker}: {e}")
            continue

        try:
            px = add_technical_features(px)
        except Exception as e:
            print(f"[SKIP] {ticker}: failed tech indicators ({e})")
            continue

        f_row = fundamentals[fundamentals["ticker"] == ticker]
        if f_row.empty:
            continue

        f_dict = f_row.iloc[0].to_dict()

        # add fundamentals
        px["ticker"] = ticker
        for k, v in f_dict.items():
            if k not in px.columns:
                px[k] = v

        all_rows.append(px)

    if not all_rows:
        raise SystemExit("No features generated.")

    features = pd.concat(all_rows, ignore_index=True)

    # reorder a bit
    id_cols = ["ticker", "date", "close"]
    other_cols = [c for c in features.columns if c not in id_cols]
    features = features[id_cols + other_cols]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(features)} rows → {OUT_FILE}")


if __name__ == "__main__":
    build_features()
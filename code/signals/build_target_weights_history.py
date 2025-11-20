# file: code/signals/build_target_weights_history.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATE_COL = "date"
TICKER_COL = "ticker"

# Files
FEATURES_TMINUS1_FILENAME = "features_with_sentiment_tminus1.csv"
PRIMARY_SIGNALS_FILENAME = "signals_history_smooth.csv"
FALLBACK_SIGNALS_FILENAME = "signals_history.csv"
OUTPUT_WEIGHTS_FILENAME = "target_weights_history.csv"

# -------------------- PORTFOLIO SETTINGS --------------------

# Cross-sectional selection size
TOP_N = 150                  # number of names to hold

# Basic risk caps
MAX_WEIGHT_PER_NAME = 0.01   # 1% per stock
MAX_WEIGHT_PER_SECTOR = 0.25 # 25% per sector

TOTAL_WEIGHT = 1.0           # fully invested (before risk overlay)

# Rebalance frequency:
# 1  = every trading day
# 5  = roughly weekly
# 20 = roughly monthly  <-- you asked for monthly
REBALANCE_EVERY_N_DAYS = 20

SECTOR_COL = "sector"
SCORE_COL = "score"
SMOOTH_COL = "score_smooth"

# ------------------------------------------------------------


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_signals_file(derived_dir: Path) -> Path:
    primary = derived_dir / PRIMARY_SIGNALS_FILENAME
    fallback = derived_dir / FALLBACK_SIGNALS_FILENAME

    if primary.exists():
        print(f"Using primary signals file: {primary}")
        return primary
    if fallback.exists():
        print(
            f"WARNING: {primary.name} not found. "
            f"Falling back to {fallback.name}."
        )
        return fallback

    print(
        f"ERROR: Neither {PRIMARY_SIGNALS_FILENAME} nor "
        f"{FALLBACK_SIGNALS_FILENAME} found in {derived_dir}"
    )
    sys.exit(1)


def load_features_tminus1(derived_dir: Path) -> pd.DataFrame:
    path = derived_dir / FEATURES_TMINUS1_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run lag_features_tminus1.py first."
        )

    print(f"Loading sector info from {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    missing = [c for c in (DATE_COL, TICKER_COL, SECTOR_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")

    # Only need unique (date, ticker, sector) for join
    return df[[DATE_COL, TICKER_COL, SECTOR_COL]].drop_duplicates()


def greedy_sector_capped_weights(df_day: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single-day DataFrame with columns:
      ticker, sector, score_for_alloc
    build long-only weights with per-name and per-sector caps.
    """

    df_day = df_day.sort_values("score_for_alloc", ascending=False).reset_index(drop=True)

    remaining_total = TOTAL_WEIGHT
    sector_remaining = {}
    weights = []

    for _, row in df_day.iterrows():
        ticker = row[TICKER_COL]
        sector = row[SECTOR_COL]

        if sector not in sector_remaining:
            sector_remaining[sector] = MAX_WEIGHT_PER_SECTOR

        if remaining_total <= 0:
            break

        w_name = MAX_WEIGHT_PER_NAME
        w_sector = sector_remaining[sector]
        w_alloc = min(remaining_total, w_name, w_sector)
        if w_alloc <= 0:
            continue

        weights.append((ticker, sector, w_alloc))
        remaining_total -= w_alloc
        sector_remaining[sector] -= w_alloc

    df_w = pd.DataFrame(weights, columns=[TICKER_COL, SECTOR_COL, "weight"])

    # Normalize to exactly sum to TOTAL_WEIGHT (usually 1)
    s = df_w["weight"].sum()
    if s > 0:
        df_w["weight"] *= TOTAL_WEIGHT / s

    return df_w


def build_weights_history(signals: pd.DataFrame, sector_info: pd.DataFrame) -> pd.DataFrame:
    use_col = SMOOTH_COL if SMOOTH_COL in signals.columns else SCORE_COL
    print(f"Using '{use_col}' for ranking.")

    # Merge sector info as-of each date
    df = signals.merge(
        sector_info,
        on=[DATE_COL, TICKER_COL],
        how="left",
        validate="m:1",
    )

    before = len(df)
    df = df.dropna(subset=[SECTOR_COL])
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows missing sector info.")

    df["score_for_alloc"] = df[use_col]

    # Full list of dates, then subsample for monthly rebal
    unique_dates = sorted(df[DATE_COL].unique())
    if REBALANCE_EVERY_N_DAYS > 1:
        unique_dates = unique_dates[::REBALANCE_EVERY_N_DAYS]

    print(f"Rebalance dates: {len(unique_dates)}")

    records = []

    for dt in unique_dates:
        df_day = df[df[DATE_COL] == dt].copy()
        if df_day.empty:
            continue

        df_day = df_day.sort_values("score_for_alloc", ascending=False).head(TOP_N)
        if df_day.empty:
            continue

        df_w = greedy_sector_capped_weights(df_day)

        for _, row in df_w.iterrows():
            records.append({
                DATE_COL: dt,
                TICKER_COL: row[TICKER_COL],
                "weight": float(row["weight"]),
            })

    weights_history = pd.DataFrame(records)
    weights_history = weights_history.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return weights_history


def main():
    root = get_project_root()
    derived_dir = root / "data" / "derived"

    signals_path = resolve_signals_file(derived_dir)
    print(f"Loading signals from {signals_path}")
    signals = pd.read_csv(signals_path, parse_dates=[DATE_COL])

    missing = [c for c in (DATE_COL, TICKER_COL, SCORE_COL) if c not in signals.columns]
    if missing:
        raise ValueError(f"Signals missing required columns: {missing}")

    sector_info = load_features_tminus1(derived_dir)

    print("Building target weights…")
    weights_history = build_weights_history(signals, sector_info)

    out_path = derived_dir / OUTPUT_WEIGHTS_FILENAME
    weights_history.to_csv(out_path, index=False)

    print(f"Saved target weights history to {out_path}")
    print(
        f"Rows: {len(weights_history)}, "
        f"dates: {weights_history[DATE_COL].nunique()}, "
        f"tickers: {weights_history[TICKER_COL].nunique()}"
    )


if __name__ == "__main__":
    main()
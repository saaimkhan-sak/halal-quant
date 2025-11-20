import math
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
SIGNALS_FILE = DERIVED_DIR / "signals_latest.csv"
OUTPUT_FILE = DERIVED_DIR / "target_portfolio.csv"

# ---- CONFIG ----
TOTAL_EQUITY = 100_000.0        # total portfolio value in dollars
TOP_K = 20                      # number of names to consider
MAX_WEIGHT_PER_NAME = 0.07      # max 7% per single position
MAX_SECTOR_WEIGHT = 0.30        # max 30% per sector
MIN_POSITION_DOLLARS = 1000.0   # drop positions smaller than this


# -------------------------------------------------------------------
# Load and clean signals
# -------------------------------------------------------------------
def load_signals():
    if not SIGNALS_FILE.exists():
        raise SystemExit(f"Signals file not found: {SIGNALS_FILE}")

    df = pd.read_csv(SIGNALS_FILE)

    required_cols = ["ticker", "score", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {SIGNALS_FILE}")

    # Fall back to UNKNOWN sector if missing
    if "sector" not in df.columns:
        print("WARNING: 'sector' missing in signals; using 'UNKNOWN' for all.")
        df["sector"] = "UNKNOWN"

    # Clean missing values
    df = df.dropna(subset=["score", "close"])

    # Sort by model score (descending)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Select top K names for long-only portfolio
# -------------------------------------------------------------------
def select_top_k(df, k):
    return df.head(min(k, len(df))).copy()


# -------------------------------------------------------------------
# Apply risk constraints (per-name cap & sector caps)
# -------------------------------------------------------------------
def apply_weight_constraints(df):
    n = len(df)
    if n == 0:
        return df

    # Start equal-weight
    df["weight"] = 1.0 / n

    # Apply per-name cap
    df["weight"] = df["weight"].clip(upper=MAX_WEIGHT_PER_NAME)

    # Sector caps: compute sector totals, scale down if needed
    sector_sums = df.groupby("sector")["weight"].sum()
    sector_scale = {}

    for sector, total_w in sector_sums.items():
        if total_w > MAX_SECTOR_WEIGHT:
            scale_factor = MAX_SECTOR_WEIGHT / total_w
            sector_scale[sector] = scale_factor
        else:
            sector_scale[sector] = 1.0

    # Apply scaling
    df["weight"] = df.apply(lambda row: row["weight"] * sector_scale[row["sector"]], axis=1)

    # If total weights exceed 100%, renormalize
    total_w = df["weight"].sum()
    if total_w > 1.0:
        df["weight"] = df["weight"] / total_w

    print(f"Total allocated weight after caps: {df['weight'].sum():.2%}")
    return df


# -------------------------------------------------------------------
# Convert weights → dollar exposures → shares
# -------------------------------------------------------------------
def compute_dollar_allocations(df):
    df = df.copy()

    df["target_dollar"] = df["weight"] * TOTAL_EQUITY

    # Drop positions too small to matter
    df = df[df["target_dollar"] >= MIN_POSITION_DOLLARS].copy()
    if df.empty:
        print("All positions fell below MIN_POSITION_DOLLARS; exiting.")
        return df

    # Re-normalize so remaining positions consume full capital
    total_alloc = df["target_dollar"].sum()
    df["weight"] = df["target_dollar"] / total_alloc
    df["target_dollar"] = df["weight"] * TOTAL_EQUITY

    # Convert to integer shares
    df["target_shares"] = (df["target_dollar"] / df["close"]).apply(lambda x: math.floor(x))

    # Recompute dollar exposures based on final integer shares
    df["final_dollar"] = df["target_shares"] * df["close"]
    df["final_weight"] = df["final_dollar"] / TOTAL_EQUITY

    df = df[df["target_shares"] > 0].copy()

    print(f"Total invested capital: ${df['final_dollar'].sum():,.2f} "
          f"({df['final_weight'].sum():.2%} of TOTAL_EQUITY)")
    return df


# -------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------
def main():
    df = load_signals()
    print(f"Loaded {len(df)} signals from {SIGNALS_FILE}")

    top_df = select_top_k(df, TOP_K)
    print(f"Selected top {len(top_df)} names.")

    risk_df = apply_weight_constraints(top_df)
    final_df = compute_dollar_allocations(risk_df)

    if final_df.empty:
        print("No positions to output.")
        return

    # Order columns nicely
    cols = [
        "ticker", "sector", "score", "close",
        "final_weight", "final_dollar", "target_shares"
    ]
    final_df = final_df[cols].sort_values("final_weight", ascending=False)

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved target portfolio to {OUTPUT_FILE}")
    print("\nPreview:\n")
    print(final_df.head(20))


if __name__ == "__main__":
    main()
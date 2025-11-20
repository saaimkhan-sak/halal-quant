import math
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
TARGET_FILE = DERIVED_DIR / "target_portfolio.csv"
CURRENT_FILE = DERIVED_DIR / "current_positions.csv"
FEATURES_FILE = DERIVED_DIR / "features_with_sentiment.csv"
ORDERS_FILE = DERIVED_DIR / "rebalance_orders.csv"

# ---- CONFIG ----
MIN_TRADE_DOLLARS = 500.0   # don't bother with trades smaller than this
TC_BPS = 5.0                # transaction cost in basis points (e.g., 5 bps = 0.05%)


def load_target():
    if not TARGET_FILE.exists():
        raise SystemExit(f"Target portfolio file not found: {TARGET_FILE}")
    df = pd.read_csv(TARGET_FILE)
    # Expect at least: ticker, target_shares or final_shares
    if "target_shares" not in df.columns:
        raise SystemExit("Expected 'target_shares' column in target_portfolio.csv.")
    df["ticker"] = df["ticker"].astype(str)
    df["target_shares"] = df["target_shares"].astype(float)
    return df


def load_current():
    # If no current file, assume all zeros
    if not CURRENT_FILE.exists():
        print(f"No current_positions file found at {CURRENT_FILE}. Assuming zero holdings.")
        return pd.DataFrame(columns=["ticker", "shares"])

    df = pd.read_csv(CURRENT_FILE)
    if "ticker" not in df.columns or "shares" not in df.columns:
        raise SystemExit(f"current_positions.csv must have columns 'ticker' and 'shares'.")
    df["ticker"] = df["ticker"].astype(str)
    df["shares"] = df["shares"].astype(float)
    return df


def load_latest_prices():
    if not FEATURES_FILE.exists():
        raise SystemExit(f"Features file not found: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    if "ticker" not in df.columns or "close" not in df.columns or "date" not in df.columns:
        raise SystemExit("features file must contain 'ticker', 'date', 'close' columns.")
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    # Get last available close per ticker
    df = df.sort_values(["ticker", "date"])
    last = df.groupby("ticker").tail(1)[["ticker", "close"]]
    return last.set_index("ticker")["close"]


def build_orders():
    target = load_target()
    current = load_current()
    prices = load_latest_prices()

    # Universe of tickers involved in either current or target
    all_tickers = sorted(set(target["ticker"].tolist()) | set(current["ticker"].tolist()))
    orders = []

    for t in all_tickers:
        tgt_row = target[target["ticker"] == t]
        cur_row = current[current["ticker"] == t]

        target_shares = float(tgt_row["target_shares"].iloc[0]) if not tgt_row.empty else 0.0
        current_shares = float(cur_row["shares"].iloc[0]) if not cur_row.empty else 0.0

        delta_shares = target_shares - current_shares
        if delta_shares == 0:
            continue

        # Get price; if missing, skip
        if t not in prices.index:
            print(f"Warning: No price for {t}. Skipping order.")
            continue

        price = float(prices.loc[t])
        notional = abs(delta_shares) * price

        # Skip tiny trades
        if notional < MIN_TRADE_DOLLARS:
            continue

        side = "BUY" if delta_shares > 0 else "SELL"
        shares = int(abs(delta_shares))

        est_cost = notional * (TC_MANGLED := (TC_BPS / 10_000.0))

        orders.append(
            {
                "ticker": t,
                "side": side,
                "shares": shares,
                "price": price,
                "notional": notional,
                "est_txn_cost": est_cost,
            }
        )

    return pd.DataFrame(orders)


def main():
    orders = build_orders()
    if orders.empty:
        print("No rebalance trades required.")
        return

    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(ORDERS_FILE, index=False)

    print(f"Generated {len(orders)} orders.")
    print(f"Saved orders to {ORDERS_FILE}")
    print("\nPreview:\n")
    print(orders.head(20))


if __name__ == "__main__":
    main()
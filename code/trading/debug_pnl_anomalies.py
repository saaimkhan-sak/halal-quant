# file: code/trading/debug_pnl_anomalies.py

import pandas as pd
from pathlib import Path

DATE_COL = "date"

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = get_project_root()
    derived = root / "data" / "derived"

    pnl_path = derived / "pnl_daily.csv"
    trades_path = derived / "trade_log_backtest.csv"
    positions_path = derived / "positions_daily.csv"

    print(f"Loading PnL from {pnl_path}")
    pnl = pd.read_csv(pnl_path, parse_dates=[DATE_COL]).sort_values(DATE_COL)

    # flag anomalies
    bad = pnl[(pnl["daily_return"] <= -1.0) | (pnl["portfolio_value"] <= 0)]
    if bad.empty:
        print("No days with daily_return <= -100% or portfolio_value <= 0.")
        return

    print("\n=== Anomaly Days ===")
    print(bad[[DATE_COL, "portfolio_value", "cash", "gross_exposure", "daily_return"]])

    # Try to load trades/positions if they exist
    trades = None
    if trades_path.exists():
        trades = pd.read_csv(trades_path, parse_dates=[DATE_COL])

    positions = None
    if positions_path.exists():
        positions = pd.read_csv(positions_path, index_col=0, parse_dates=True)

    for _, row in bad.iterrows():
        dt = row[DATE_COL]
        print(f"\n--- Detail for {dt.date()} ---")
        print(row)

        if trades is not None:
            day_trades = trades[trades[DATE_COL] == dt]
            print(f"\nTrades on {dt.date()} ({len(day_trades)} rows):")
            print(day_trades.head(20))

        if positions is not None:
            if dt in positions.index:
                pos_row = positions.loc[dt]
                # show 10 largest positions by abs value
                # We need prices to compute notional; use pnl_daily PV as context only here.
                largest = pos_row.abs().sort_values(ascending=False).head(10)
                print("\nLargest positions by shares (abs):")
                print(largest)
            else:
                print(f"No positions row found for {dt.date()}.")


if __name__ == "__main__":
    main()
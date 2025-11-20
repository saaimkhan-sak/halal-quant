# file: code/data/check_price_panel_integrity.py

import pandas as pd
from pathlib import Path

DATE_COL = "date"
TICKER_COL = "ticker"
PRICE_COL = "close"

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    root = get_project_root()
    derived = root / "data" / "derived"

    path = derived / "price_panel.csv"
    print(f"Loading price panel from {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    total_rows = len(df)
    n_nan = df[PRICE_COL].isna().sum()
    n_zero = (df[PRICE_COL] == 0).sum()
    n_neg = (df[PRICE_COL] < 0).sum()

    print(f"Total rows: {total_rows}")
    print(f"NaN close:  {n_nan}")
    print(f"Zero close: {n_zero}")
    print(f"Neg close:  {n_neg}")

    bad = df[df[PRICE_COL].isna() | (df[PRICE_COL] <= 0)].copy()
    if bad.empty:
        print("No NaN/zero/negative prices found.")
        return

    print("\nSample of bad rows:")
    print(bad.head(20))

    # Also show which tickers are affected most
    by_ticker = bad.groupby(TICKER_COL).size().sort_values(ascending=False).head(20)
    print("\nTickers with most bad prices:")
    print(by_ticker)


if __name__ == "__main__":
    main()
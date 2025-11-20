import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
FUND_FILE = DERIVED_DIR / "fundamentals_sp.csv"


def main():
    if not FUND_FILE.exists():
        raise SystemExit(f"fundamentals_sp.csv not found at {FUND_FILE}")

    df = pd.read_csv(FUND_FILE)
    if "ticker" not in df.columns:
        raise SystemExit("fundamentals_sp.csv must have a 'ticker' column")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    sectors = {}
    for i, t in enumerate(df["ticker"].unique(), start=1):
        if "sector" in df.columns and not pd.isna(
            df.loc[df["ticker"] == t, "sector"].iloc[0]
        ):
            # already have sector value; skip
            continue

        print(f"[{i}] Fetching sector for {t}...")
        try:
            info = yf.Ticker(t).info
        except Exception as e:
            print(f"  {t}: error {e}")
            sectors[t] = None
            continue

        sector = info.get("sector")
        if sector is None:
            print(f"  {t}: sector not found in Yahoo info")
        else:
            print(f"  {t}: sector = {sector}")
        sectors[t] = sector
        time.sleep(0.5)  # be nice to Yahoo

    # If 'sector' column doesn't exist yet, create it
    if "sector" not in df.columns:
        df["sector"] = None

    # Fill sectors
    for t, s in sectors.items():
        if s is not None:
            df.loc[df["ticker"] == t, "sector"] = s

    # Some may still be NaN; label as 'Unknown'
    df["sector"] = df["sector"].fillna("Unknown")

    df.to_csv(FUND_FILE, index=False)
    print(f"Updated {FUND_FILE} with sector information.")


if __name__ == "__main__":
    main()
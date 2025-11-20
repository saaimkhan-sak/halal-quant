import time
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
DERIVED_DIR = DATA_DIR / "derived"

ETF_UNION_FILE = DERIVED_DIR / "etf_union_us.csv"
FUNDAMENTALS_FILE = DERIVED_DIR / "fundamentals_sp.csv"


def load_universe():
    """Get the list of tickers we care about."""
    if ETF_UNION_FILE.exists():
        u = pd.read_csv(ETF_UNION_FILE)
        tickers = u["ticker"].dropna().astype(str).str.upper().unique().tolist()
        print(f"Loaded {len(tickers)} tickers from etf_union_us.csv")
        return tickers

    if FUNDAMENTALS_FILE.exists():
        f = pd.read_csv(FUNDAMENTALS_FILE)
        tickers = f["ticker"].dropna().astype(str).str.upper().unique().tolist()
        print(f"Loaded {len(tickers)} tickers from fundamentals_sp.csv")
        return tickers

    raise SystemExit("No ETF union or fundamentals file found to build universe.")


def download_prices_for_ticker(ticker: str, period="10y", interval="1d"):
    """Download price history for one ticker and save to data/market/{TICKER}.csv"""
    try:
        print(f"[{ticker}] downloading {period} {interval} history…")
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as e:
        print(f"[{ticker}] ERROR during download: {e}")
        return

    if df.empty:
        print(f"[{ticker}] no data returned, skipping.")
        return

    # Move index to column for Date
    df = df.reset_index()

    # Normalize column names
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Ensure we have a Date column
    if "Date" not in df.columns:
        print(f"[{ticker}] missing Date column after download, skipping.")
        return

    out_path = MARKET_DIR / f"{ticker}.csv"
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[{ticker}] saved {len(df)} rows → {out_path}")


def main():
    tickers = load_universe()

    # If you want to test on a smaller subset first, uncomment this:
    # tickers = tickers[:50]

    for i, ticker in enumerate(tickers, start=1):
        out_path = MARKET_DIR / f"{ticker}.csv"
        if out_path.exists():
            print(f"[{ticker}] price file already exists, skipping download.")
            continue

        download_prices_for_ticker(ticker, period="10y", interval="1d")

        # polite pause so we don't hammer Yahoo too hard
        time.sleep(1)


if __name__ == "__main__":
    main()
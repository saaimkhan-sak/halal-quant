import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"

FUND_CSV = DERIVED / "fundamentals_sp.csv"   # or fundamentals_ml.csv if you prefer
OUT = DERIVED / "price_panel.csv"

# How far back?
PERIOD = "10y"       # "max", "5y", "1y", etc.
INTERVAL = "1d"      # daily bars

if __name__ == "__main__":
    # Use all tickers we have fundamentals for
    f = pd.read_csv(FUND_CSV)
    tickers = sorted(f["ticker"].dropna().astype(str).unique().tolist())
    print(f"Building panel for {len(tickers)} tickers")

    all_frames = []

    for i, t in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {t} downloading history…")
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period=PERIOD, interval=INTERVAL)
        except Exception as e:
            print(f"  {t}: error: {e}, skipping")
            continue

        if hist.empty:
            print(f"  {t}: no data, skipping")
            continue

        # Standardize columns
        hist = hist.reset_index()  # Date becomes a column
        # Some versions of yfinance use 'Adj Close', some 'Adj Close'
        cols_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        # Only keep what exists
        use_cols = {src: dst for src, dst in cols_map.items() if src in hist.columns}
        hist = hist[list(use_cols.keys()) + ["Date"]].rename(columns=use_cols)
        hist["date"] = pd.to_datetime(hist["Date"]).dt.date
        hist.drop(columns=["Date"], inplace=True)

        # Add ticker identifier
        hist["ticker"] = t

        # Compute daily return (based on adj_close if available, else close)
        if "adj_close" in hist.columns:
            price_col = "adj_close"
        else:
            price_col = "close"

        hist = hist.sort_values("date")
        hist["daily_return"] = hist[price_col].pct_change()

        all_frames.append(hist)

    if not all_frames:
        raise SystemExit("No price data downloaded; check tickers or yfinance.")

    panel = pd.concat(all_frames, ignore_index=True)
    # Optional: drop first row per ticker where daily_return is NaN
    panel = panel.sort_values(["ticker", "date"])
    panel.to_csv(OUT, index=False)

    print(f"\nWrote {OUT} with {len(panel)} rows")
    print(panel.head())
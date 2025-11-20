import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def monthly_market_cap(ticker: str, shares_outstanding: float) -> pd.Series:
    """Build a month-end market cap series from your daily close CSV."""
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv")
    # Ensure Date is true datetime and tz-naive for resampling
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.tz_convert(None)  # make tz-naive
    df = df.set_index("Date").sort_index()

    # Month-end resample (use 'ME' instead of deprecated 'M')
    monthly_close = df["close"].resample("ME").last()
    mcap = monthly_close * float(shares_outstanding)
    return mcap

def avg_mve_36m(ticker: str, shares_outstanding: float) -> float:
    """Return the 36-month average MVE; if <36 months, use available mean for now."""
    mc = monthly_market_cap(ticker, shares_outstanding)
    if len(mc) < 36:
        return float(mc.mean())
    return float(mc.tail(36).mean())
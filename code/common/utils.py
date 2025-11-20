import os
import math
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import DATA_DIR

def ensure_csv(path: str, columns):
    p = Path(path)
    if not p.exists():
        pd.DataFrame(columns=columns).to_csv(p, index=False)

def load_close_series(ticker: str) -> pd.Series:
    df = pd.read_csv(Path(DATA_DIR) / f"{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    return df["Date"], df["close"]

def latest_price_and_date(ticker: str):
    dates, close = load_close_series(ticker)
    return float(close.iloc[-1]), pd.to_datetime(dates.iloc[-1])

def today_stamp() -> str:
    # Use wall-clock timestamp for the log
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def shares_for_budget(price: float, dollars: float) -> int:
    if price <= 0: return 0
    return max(0, math.floor(dollars / price))
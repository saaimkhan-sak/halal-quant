import pandas as pd

def zscore(series: pd.Series, lookback: int = 20) -> pd.Series:
    s = series.dropna()
    mean = s.rolling(lookback).mean()
    std = s.rolling(lookback).std()
    return (s - mean) / std

def mean_reversion_rank(close: pd.Series, lookback: int = 20) -> float:
    rets = close.pct_change()
    z = zscore(rets, lookback)
    return float(z.iloc[-1])  # most recent z-score
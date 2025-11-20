import pandas as pd
from pathlib import Path
from signals import mean_reversion_rank

DATA_DIR = Path("data")

def load_close(ticker: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    return df["close"]

if __name__ == "__main__":
    uni = pd.read_csv("universe.csv")["ticker"].tolist()

    rows = []
    for t in uni:
        try:
            close = load_close(t)
            z = mean_reversion_rank(close, lookback=20)
            rows.append({"ticker": t, "z": z})
        except Exception as e:
            print(f"{t}: {e}")

    out = pd.DataFrame(rows).dropna().sort_values("z")
    # Simple interpretation: z < -1.5 = oversold (BUY candidate); z > +1.5 = overbought (AVOID/SHORT in theory)
    def label(z):
        if z < -1.5: return "BUY"
        elif z > 1.5: return "AVOID"
        return "HOLD"

    out["signal"] = out["z"].apply(label)
    print(out)
    out.to_csv("candidates.csv", index=False)
    print("\nWrote candidates.csv")
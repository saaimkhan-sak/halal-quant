import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"

SRC = DERIVED / "fundamentals_sp.csv"
OUT = DERIVED / "fundamentals_ml.csv"

if __name__ == "__main__":
    df = pd.read_csv(SRC)

    # Columns we keep for generic modeling
    keep_cols = [
        "ticker",
        "total_debt",
        "shares_outstanding",
        "total_revenue",
    ]

    # Make sure these exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Drop all Shariah-specific / flag columns
    ml_df = df[keep_cols].copy()

    # Optional: add some basic ratios
    ml_df["debt_to_revenue"] = ml_df["total_debt"] / ml_df["total_revenue"]
    ml_df["revenue_per_share"] = ml_df["total_revenue"] / ml_df["shares_outstanding"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(ml_df)} tickers")
    print(ml_df.head())
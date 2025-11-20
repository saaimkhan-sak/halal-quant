# file: code/data/lag_features_tminus1.py

import pandas as pd
from pathlib import Path

DATE_COL = "date"
TICKER_COL = "ticker"

INPUT_FILENAME = "features_with_sentiment.csv"
OUTPUT_FILENAME = "features_with_sentiment_tminus1.csv"


def get_project_root() -> Path:
    # /.../code/data/lag_features_tminus1.py -> project root
    return Path(__file__).resolve().parents[2]


def main():
    root = get_project_root()
    derived_dir = root / "data" / "derived"

    src = derived_dir / INPUT_FILENAME
    dst = derived_dir / OUTPUT_FILENAME

    print(f"Loading {src}")
    df = pd.read_csv(src, parse_dates=[DATE_COL])

    # sanity
    for col in (DATE_COL, TICKER_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {INPUT_FILENAME}")

    df = df.sort_values([TICKER_COL, DATE_COL])

    # Treat returns as labels (keep them unshifted)
    label_cols = [c for c in df.columns if c.startswith("ret_")]
    # In case you later add fwd returns / labels, also protect these:
    label_cols += [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["fwd", "target", "label"])
        and c not in label_cols
    ]

    non_feature_cols = {DATE_COL, TICKER_COL}
    feature_cols = [
        c
        for c in df.columns
        if c not in non_feature_cols and c not in label_cols
    ]

    print(f"Label columns (NOT shifted): {label_cols}")
    print(f"Number of feature cols to T-1 shift: {len(feature_cols)}")

    df_lagged = df.copy()
    df_lagged[feature_cols] = (
        df_lagged
        .groupby(TICKER_COL, group_keys=False)[feature_cols]
        .shift(1)
    )

    before = len(df_lagged)
    df_lagged = df_lagged.dropna(subset=feature_cols)
    after = len(df_lagged)

    print(f"Dropped {before - after} rows with incomplete T-1 features.")
    df_lagged.to_csv(dst, index=False)
    print(f"Saved T-1 features to {dst}")


if __name__ == "__main__":
    main()

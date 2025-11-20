import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = "date"
TICKER_COL = "ticker"

FEATURES_FILENAME = "features_with_sentiment_tminus1.csv"
OUTPUT_FILENAME = "signals_history.csv"

# Preferred model (ranker); will fall back to the other if needed
PRIMARY_MODEL_FILENAME = "xgb_rank_fwd5d_cs.pkl"
FALLBACK_MODEL_FILENAME = "xgb_fwd5d.pkl"

# ---- HARD-CODED FEATURE LIST (36 COLS TO MATCH MODEL) ----
# These must all exist in features_with_sentiment_tminus1.csv
EXPLICIT_FEATURE_COLS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "dividends",
    "stock_splits",
    "vol_20d",
    "zscore_close_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_high",
    "bb_low",
    "bb_mid",
    "atr_14",
    "vol_zscore_20d",
    "total_debt",
    "shares_outstanding",
    "is_fully_islamic",
    "non_permissible_revenue",
    "total_revenue",
    "alcohol",
    "pork",
    "gambling",
    "pornography",
    "tobacco_vaping",
    "recreational_cannabis",
    "impermissible_advertising",
    "impermissible_media",
    "deferred_gold_silver",
    "conventional_finance",
    "debt_to_revenue",
    "revenue_per_share",
    "sent_net_pos",
]


def get_project_root() -> Path:
    # /.../code/signals/build_signals_history.py -> project root
    return Path(__file__).resolve().parents[2]


def load_model(models_dir: Path):
    primary = models_dir / PRIMARY_MODEL_FILENAME
    fallback = models_dir / FALLBACK_MODEL_FILENAME

    if primary.exists():
        model_path = primary
    elif fallback.exists():
        print(
            f"WARNING: {PRIMARY_MODEL_FILENAME} not found. "
            f"Falling back to {FALLBACK_MODEL_FILENAME}"
        )
        model_path = fallback
    else:
        raise FileNotFoundError(
            f"Neither {PRIMARY_MODEL_FILENAME} nor {FALLBACK_MODEL_FILENAME} "
            f"found in {models_dir}"
        )

    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def get_feature_columns(df: pd.DataFrame, model) -> list[str]:
    """
    Prefer model.feature_names_in_ if present; otherwise use EXPLICIT_FEATURE_COLS.
    """

    # Best case: model carries feature names from training
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            print(
                "WARNING: The following model feature names are missing in the "
                f"features DataFrame and will be dropped: {missing}"
            )
        feature_cols = [c for c in feature_names if c in df.columns]
        if not feature_cols:
            raise ValueError(
                "model.feature_names_in_ exists, but none are present in the "
                "features DataFrame. Check that you're using the same features "
                "file used for training."
            )
        print(f"Using {len(feature_cols)} feature columns from model.feature_names_in_")
        return feature_cols

    # Fallback: use our explicit list
    print("model.feature_names_in_ not found; using EXPLICIT_FEATURE_COLS.")
    missing = [c for c in EXPLICIT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "The following expected feature columns are missing from "
            f"{FEATURES_FILENAME}:\n  {missing}\n"
            "Check your features_with_sentiment_tminus1.csv header or update "
            "EXPLICIT_FEATURE_COLS accordingly."
        )

    feature_cols = EXPLICIT_FEATURE_COLS.copy()

    # Sanity check vs model's expected number of features
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and n_expected != len(feature_cols):
        raise ValueError(
            f"Model expects {n_expected} features, but EXPLICIT_FEATURE_COLS "
            f"has {len(feature_cols)}. Adjust EXPLICIT_FEATURE_COLS."
        )

    print(f"Using {len(feature_cols)} explicit feature columns:")
    print("  " + ", ".join(feature_cols))
    return feature_cols


def main():
    root = get_project_root()
    derived_dir = root / "data" / "derived"
    models_dir = root / "models"

    features_path = derived_dir / FEATURES_FILENAME
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found.\n"
            "Run code/data/lag_features_tminus1.py first to generate "
            "features_with_sentiment_tminus1.csv."
        )

    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path, parse_dates=[DATE_COL])

    # Basic sanity
    missing = [c for c in (DATE_COL, TICKER_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {FEATURES_FILENAME}: {missing}"
        )

    df = df.sort_values([DATE_COL, TICKER_COL])

    model = load_model(models_dir)
    feature_cols = get_feature_columns(df, model)

    # Keep only rows where all features are available
    before = len(df)
    df_valid = df.dropna(subset=feature_cols).copy()
    after = len(df_valid)

    if after < before:
        print(f"Dropped {before - after} rows with NaNs in feature columns.")

    X = df_valid[feature_cols].values

    print(f"Scoring {len(df_valid)} (date, ticker) rows with {X.shape[1]} features...")
    scores = model.predict(X)
    df_valid["score"] = scores

    signals = (
        df_valid[[DATE_COL, TICKER_COL, "score"]]
        .sort_values([DATE_COL, TICKER_COL])
        .reset_index(drop=True)
    )

    out_path = derived_dir / OUTPUT_FILENAME
    signals.to_csv(out_path, index=False)

    print(f"Saved signals history to {out_path}")
    print(
        f"Rows: {len(signals)}, "
        f"dates: {signals[DATE_COL].nunique()}, "
        f"tickers: {signals[TICKER_COL].nunique()}"
    )


if __name__ == "__main__":
    main()

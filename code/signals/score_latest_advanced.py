import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
FEATURES_FILE = DERIVED_DIR / "features_with_sentiment.csv"
MODEL_PATH = ROOT / "models" / "xgb_rank_fwd5d_cs.pkl"
OUT_PATH = DERIVED_DIR / "signals_latest.csv"


def add_return_features(df, price_col="close"):
    df = df.copy()
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()

    for h in [5, 21, 63]:
        df[f"ret_{h}d"] = df.groupby("ticker")[price_col].pct_change(h)

    for w in [5, 20, 60]:
        df[f"vol_{w}d"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


def sector_neutral_zscore(df, feature_cols):
    df = df.copy()
    if "sector" not in df.columns:
        raise SystemExit("Sector column 'sector' missing.")
    df["sector"] = df["sector"].fillna("Unknown")

    for col in feature_cols:
        sn_col = col + "_snz"
        df[sn_col] = df.groupby(["date", "sector"])[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
    return df


def load_and_prepare_features():
    print(f"Loading features from {FEATURES_FILE} ...")
    df = pd.read_csv(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print("Raw shape:", df.shape)

    df = add_return_features(df, price_col="close")

    nonfeat = {
        "ticker", "date", "sector",
        "fwd_ret_5d", "fwd_ret_5d_w",
        "fwd_ret_5d_cs", "fwd_sharpe_5d",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_features = [c for c in numeric_cols if c not in nonfeat]

    print("Base feature count:", len(base_features))

    df = sector_neutral_zscore(df, base_features)
    feature_cols_snz = [c + "_snz" for c in base_features]

    df[feature_cols_snz] = df[feature_cols_snz].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, feature_cols_snz


def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, xgb.XGBRanker):
        print("WARNING: model is not an XGBRanker.")
    return model


def score_latest(df, model, feature_cols_snz):
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    print(f"Latest date in data: {latest_date} with {len(snap)} tickers")

    if snap.empty:
        raise SystemExit("No rows for latest date; cannot score.")

    X = snap[feature_cols_snz].to_numpy()
    scores = model.predict(X)
    snap["score"] = scores

    snap = snap.sort_values("score", ascending=False).reset_index(drop=True)
    snap["rank"] = snap.index + 1

    out_cols = ["ticker", "date", "sector", "close", "score", "rank"]
    out_cols = [c for c in out_cols if c in snap.columns]

    snap[out_cols].to_csv(OUT_PATH, index=False)
    print(f"Saved {len(snap)} signals to {OUT_PATH}")

    print("\nTop 20 signals:")
    print(snap[out_cols].head(20))


def main():
    df, feature_cols_snz = load_and_prepare_features()
    model = load_model()
    score_latest(df, model, feature_cols_snz)


if __name__ == "__main__":
    main()
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
FEATURES_FILE = DERIVED_DIR / "features_with_sentiment.csv"
MODEL_PATH = ROOT / "models" / "xgb_rank_fwd5d_cs.pkl"

TOP_K = 20
HOLDING_DAYS = 5

# Transaction cost settings
TC_BPS = 5.0  # 5 bps per side
COST_ROUND_TRIP = 2 * TC_BPS / 10_000.0  # e.g. 0.001 = 0.10%


def load_features():
    print(f"Loading features from {FEATURES_FILE} ...")
    df = pd.read_csv(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_quant_features(df):
    """Replicate key pieces of train_model_advanced: returns & forward returns."""
    df = df.copy()

    # Daily returns
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    # Multi-horizon returns
    for h in [5, 21, 63]:
        df[f"ret_{h}d"] = df.groupby("ticker")["close"].pct_change(h)

    # Rolling vols
    for w in [5, 20, 60]:
        df[f"vol_{w}d"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    # 5-day forward return
    df["fwd_ret_5d"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1.0

    # Winsorize fwd_ret_5d globally, center per date
    def winsorize_series(s, lower=1.0, upper=99.0):
        values = s.dropna().values
        if len(values) == 0:
            return s
        lo, hi = np.percentile(values, [lower, upper])
        return s.clip(lower=lo, upper=hi)

    df["fwd_ret_5d_w"] = winsorize_series(df["fwd_ret_5d"])
    df["fwd_ret_5d_cs"] = (
        df["fwd_ret_5d_w"]
        - df.groupby("date")["fwd_ret_5d_w"].transform("mean")
    )

    return df


def sector_neutral_zscore(df, feature_cols):
    df = df.copy()
    if "sector" not in df.columns:
        raise SystemExit("Sector column 'sector' missing. Rebuild features with sector info.")
    df["sector"] = df["sector"].fillna("Unknown")

    for col in feature_cols:
        sn_col = col + "_snz"
        df[sn_col] = df.groupby(["date", "sector"])[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
    return df


def backtest_long_only(df, model, base_features, test_start_date):
    df = df[df["date"] >= test_start_date].copy()
    df = df.sort_values("date")

    df = sector_neutral_zscore(df, base_features)
    feature_cols_snz = [c + "_snz" for c in base_features]

    df[feature_cols_snz] = df[feature_cols_snz].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    results = []

    max_date = df["date"].max()
    last_entry = max_date - pd.Timedelta(days=HOLDING_DAYS)
    entry_dates = sorted(d for d in df["date"].unique() if d <= last_entry)

    for dt in entry_dates:
        g = df[df["date"] == dt].copy()
        g = g[~g["fwd_ret_5d"].isna()]
        if len(g) < TOP_K:
            continue

        X = g[feature_cols_snz].to_numpy()
        scores = model.predict(X)
        g["score"] = scores

        g = g.sort_values("score", ascending=False)
        longs = g.head(TOP_K)

        gross_ret_5d = longs["fwd_ret_5d"].mean()
        # Subtract round-trip trading cost for entering & exiting the basket
        net_ret_5d = gross_ret_5d - COST_ROUND_TRIP

        results.append(
            {
                "entry_date": dt,
                "n_long": len(longs),
                "gross_fwd5d_ret": gross_ret_5d,
                "net_fwd5d_ret": net_ret_5d,
            }
        )

    bt = pd.DataFrame(results).sort_values("entry_date").reset_index(drop=True)
    return bt


def main():
    df = load_features()
    df = add_quant_features(df)

    # Build base_features exactly like in training
    nonfeat = {
        "ticker", "date", "sector",
        "fwd_ret_5d", "fwd_ret_5d_w",
        "fwd_ret_5d_cs", "fwd_sharpe_5d",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_features = [c for c in numeric_cols if c not in nonfeat]

    print("Base features used for scoring:", len(base_features))

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, xgb.XGBRanker)
    print(f"Loaded model from {MODEL_PATH}")

    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=180)
    print(f"Backtesting non-overlapping from {test_start.date()} to {max_date.date()}")

    bt = backtest_long_only(df, model, base_features, test_start_date=test_start)
    print("Backtest result shape:", bt.shape)
    print(bt.head())

    if not bt.empty:
        avg_5d_gross = bt["gross_fwd5d_ret"].mean()
        avg_5d_net = bt["net_fwd5d_ret"].mean()
        vol_5d_net = bt["net_fwd5d_ret"].std()

        ann_ret_net = (1 + avg_5d_net) ** (252 / HOLDING_DAYS) - 1
        ann_vol_net = vol_5d_net * np.sqrt(252 / HOLDING_DAYS)
        sharpe_net = ann_ret_net / (ann_vol_net + 1e-8)

        print(f"\nGross average 5d return: {avg_5d_gross:.4%}")
        print(f"Net average 5d return:   {avg_5d_net:.4%}")
        print(f"Net annualized return:   {ann_ret_net:.2%}")
        print(f"Net annualized vol:      {ann_vol_net:.2%}")
        print(f"Net Sharpe:              {sharpe_net:.2f}")
        print(f"(Cost per 5d cycle:      {COST_ROUND_TRIP:.3%})")

        out_path = DERIVED_DIR / "bt_rank_longonly_5d_with_sentiment_costs.csv"
        bt.to_csv(out_path, index=False)
        print(f"Backtest path: {out_path}")
    else:
        print("No valid trades generated.")


if __name__ == "__main__":
    main()

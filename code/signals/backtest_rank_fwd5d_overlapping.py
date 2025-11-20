import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
FEATURES_FILE = DERIVED_DIR / "features_with_sentiment.csv"
MODEL_PATH = ROOT / "models" / "xgb_rank_fwd5d_cs.pkl"

HOLDING_DAYS = 5
TOP_K = 20

# Transaction cost settings
TC_BPS = 5.0  # per side
COST_ROUND_TRIP = 2 * TC_BPS / 10_000.0  # e.g. 0.10%


def load_features():
    print(f"Loading features from {FEATURES_FILE} ...")
    df = pd.read_csv(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_quant_features(df):
    df = df.copy()

    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    for h in [5, 21, 63]:
        df[f"ret_{h}d"] = df.groupby("ticker")["close"].pct_change(h)

    for w in [5, 20, 60]:
        df[f"vol_{w}d"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    df["fwd_ret_5d"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1.0

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
        raise SystemExit("Sector column 'sector' missing.")
    df["sector"] = df["sector"].fillna("Unknown")

    for col in feature_cols:
        sn_col = col + "_snz"
        df[sn_col] = df.groupby(["date", "sector"])[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
    return df


def build_ret_matrix(df):
    df_ret = df[["date", "ticker", "ret_1d"]].copy()
    df_ret = df_ret.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    return df_ret


def compute_leg_daily_returns(entry_date, tickers, ret_matrix):
    start = entry_date + pd.Timedelta(days=1)
    end = entry_date + pd.Timedelta(days=HOLDING_DAYS)
    mask = (ret_matrix.index >= start) & (ret_matrix.index <= end)
    leg_dates = ret_matrix.index[mask]
    if len(leg_dates) == 0:
        return pd.Series(dtype=float)

    sub = ret_matrix.loc[leg_dates, tickers]
    leg_ret = sub.mean(axis=1)

    if not leg_ret.empty:
        first = leg_ret.index[0]
        leg_ret.loc[first] = leg_ret.loc[first] - COST_ROUND_TRIP

    return leg_ret


def overlapping_backtest(df, model, base_features, test_start_date):
    df = df[df["date"] >= test_start_date].copy()
    df = df.sort_values("date")

    df = sector_neutral_zscore(df, base_features)
    feature_cols_snz = [c + "_snz" for c in base_features]

    df[feature_cols_snz] = df[feature_cols_snz].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret_matrix = build_ret_matrix(df)

    all_dates = sorted(df["date"].unique())
    max_date = df["date"].max()
    last_entry = max_date - pd.Timedelta(days=HOLDING_DAYS)
    entry_dates = [d for d in all_dates if d <= last_entry]

    print(f"Number of entry dates: {len(entry_dates)}")

    legs = []
    for dt in entry_dates:
        g = df[df["date"] == dt].copy()
        g = g.dropna(subset=["ret_1d"])
        if len(g) < TOP_K:
            continue

        X = g[feature_cols_snz].to_numpy()
        scores = model.predict(X)
        g["score"] = scores

        g = g.sort_values("score", ascending=False)
        longs = g.head(TOP_K)
        tickers = longs["ticker"].tolist()

        leg_ret = compute_leg_daily_returns(dt, tickers, ret_matrix)
        if leg_ret.empty:
            continue

        legs.append((dt, leg_ret))

    day_to_legrets = {}
    for entry_dt, leg_ret in legs:
        for day, r in leg_ret.items():
            day_to_legrets.setdefault(day, []).append(r)

    if not day_to_legrets:
        print("No legs generated.")
        return pd.DataFrame()

    records = []
    for day in sorted(day_to_legrets.keys()):
        rets = day_to_legrets[day]
        records.append(
            {"date": day, "port_ret": float(np.mean(rets)), "n_legs": len(rets)}
        )

    bt = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return bt


def main():
    df = load_features()
    df = add_quant_features(df)

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
    print(f"Overlapping backtest from {test_start.date()} to {max_date.date()}")

    bt = overlapping_backtest(df, model, base_features, test_start)
    print("Overlapping backtest result shape:", bt.shape)
    print(bt.head())

    if not bt.empty:
        avg_daily = bt["port_ret"].mean()
        vol_daily = bt["port_ret"].std()
        ann_ret = (1 + avg_daily) ** 252 - 1
        ann_vol = vol_daily * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-8)

        print(f"\nNet average daily return: {avg_daily:.4%}")
        print(f"Net annualized return:    {ann_ret:.2%}")
        print(f"Net annualized vol:       {ann_vol:.2%}")
        print(f"Net Sharpe:               {sharpe:.2f}")
        print(f"Average active legs:      {bt['n_legs'].mean():.2f}")
        print(f"(Cost per leg:            {COST_ROUND_TRIP:.3%})")

        out_path = DERIVED_DIR / "bt_rank_longonly_5d_overlap_with_sentiment_costs.csv"
        bt.to_csv(out_path, index=False)
        print(f"Overlapping backtest path: {out_path}")
    else:
        print("No overlapping trades generated.")


if __name__ == "__main__":
    main()

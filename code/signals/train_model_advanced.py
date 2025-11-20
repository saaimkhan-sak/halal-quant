import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"
FEATURES_FILE = DERIVED_DIR / "features_with_sentiment.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# Feature Engineering
# =========================================================

def add_return_features(df, price_col="close"):
    """Add daily, multi-horizon returns and vol features."""
    df = df.copy()

    # Daily returns
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()

    # Multi-horizon returns
    for h in [5, 21, 63]:
        df[f"ret_{h}d"] = df.groupby("ticker")[price_col].pct_change(h)

    # Rolling vols
    for w in [5, 20, 60]:
        df[f"vol_{w}d"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


def add_forward_returns(df, price_col="close"):
    """Add forward 5-day return target and risk-adjusted version."""
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    # 5-day forward return
    df["fwd_ret_5d"] = (
        df.groupby("ticker")[price_col].shift(-5) / df[price_col] - 1.0
    )

    # Risk-adjustment vol
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()

    df["vol_20d"] = (
        df.groupby("ticker")["ret_1d"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["fwd_sharpe_5d"] = df["fwd_ret_5d"] / (df["vol_20d"] + 1e-6)

    # Winsorize forward returns globally
    def winsorize_series(s, lower=1.0, upper=99.0):
        values = s.dropna().values
        if len(values) == 0:
            return s
        lo, hi = np.percentile(values, [lower, upper])
        return s.clip(lower=lo, upper=hi)

    df["fwd_ret_5d_w"] = winsorize_series(df["fwd_ret_5d"])

    # Center the winsorized fwd ret per date (cross-sectional alpha target)
    df["fwd_ret_5d_cs"] = (
        df["fwd_ret_5d_w"]
        - df.groupby("date")["fwd_ret_5d_w"].transform("mean")
    )

    return df


def sector_neutral_zscore(df, feature_cols, date_col="date", sector_col="sector"):
    """Sector-neutral z-scoring: z-score features within each (date, sector)."""
    df = df.copy()

    if sector_col not in df.columns:
        raise SystemExit(
            "sector column missing in dataframe. "
            "Did you rebuild features after adding sectors?"
        )

    df[sector_col] = df[sector_col].fillna("Unknown")

    for col in feature_cols:
        sn_col = col + "_snz"
        df[sn_col] = df.groupby([date_col, sector_col])[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
    return df


# =========================================================
# Load + Engineer Dataset
# =========================================================

def load_and_engineer():
    print(f"Loading features from {FEATURES_FILE} ...")
    df = pd.read_csv(FEATURES_FILE)

    # Basic checks
    if "ticker" not in df.columns or "date" not in df.columns:
        raise SystemExit("features_with_sentiment.csv must contain ticker and date.")
    if "sector" not in df.columns:
        raise SystemExit("features_with_sentiment.csv missing sector column.")

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print("Raw shape:", df.shape)

    # Engineering
    df = add_return_features(df)
    df = add_forward_returns(df)

    # Drop rows missing target
    df = df.dropna(subset=["fwd_ret_5d_cs"]).reset_index(drop=True)
    print("After adding targets:", df.shape)

    # Feature selection
    nonfeat = {
        "ticker", "date", "sector",
        "fwd_ret_5d", "fwd_ret_5d_w",
        "fwd_ret_5d_cs", "fwd_sharpe_5d"
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_features = [c for c in numeric_cols if c not in nonfeat]

    print("Base feature count:", len(base_features))

    # Sector-neutral zscores
    df = sector_neutral_zscore(df, base_features)

    # Model features are *_snz
    feature_cols = [c + "_snz" for c in base_features]

    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["fwd_ret_5d_cs"]).reset_index(drop=True)
    print("After sector-neutral cleaning:", df.shape)

    return df, feature_cols


# =========================================================
# Build Ranker Dataset
# =========================================================

def make_rank_data(df, feature_cols, target_col="fwd_ret_5d_cs", test_days=180):
    """Time-split train/test with grouping by date for XGBRanker."""
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    print(f"Max date = {max_date}, test cutoff = {cutoff}")

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Sort
    train_df = train_df.sort_values(["date", "ticker"])
    test_df = test_df.sort_values(["date", "ticker"])

    # Arrays
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # Groups = number of samples per date
    group_train = train_df.groupby("date").size().values.tolist()
    group_test = test_df.groupby("date").size().values.tolist()

    print("Train groups:", len(group_train))
    print("Test groups:", len(group_test))

    return X_train, y_train, group_train, X_test, y_test, test_df


# =========================================================
# Training + Evaluation
# =========================================================

def train_ranker(X_train, y_train, group_train):
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "rank:pairwise",
        "random_state": 42,
        "tree_method": "hist",
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
    }

    model = xgb.XGBRanker(**params)

    print("Training XGBRanker...")
    model.fit(X_train, y_train, group=group_train, verbose=True)
    return model


def spearman_corr(a, b):
    """Compute Spearman correlation without SciPy."""
    a_rank = pd.Series(a).rank()
    b_rank = pd.Series(b).rank()
    return a_rank.corr(b_rank)


def evaluate_ranker(model, test_df, feature_cols, target_col="fwd_ret_5d_cs"):
    """Rank-based evaluation."""

    cs_corrs = []
    top_rets = []
    bottom_rets = []

    for dt, g in test_df.groupby("date"):
        if len(g) < 20:
            continue

        X = g[feature_cols].to_numpy()
        y = g[target_col].to_numpy()
        preds = model.predict(X)

        # Spearman correlation
        corr = spearman_corr(y, preds)
        if np.isfinite(corr):
            cs_corrs.append(corr)

        # Top/bottom decile
        g = g.copy()
        g["pred"] = preds
        g = g.sort_values("pred", ascending=False)

        n = len(g)
        k = max(1, n // 10)

        top = g.head(k)["fwd_ret_5d"].mean()
        bottom = g.tail(k)["fwd_ret_5d"].mean()

        top_rets.append(top)
        bottom_rets.append(bottom)

    # Summary metrics
    avg_cs = float(np.mean(cs_corrs)) if cs_corrs else np.nan
    avg_top = float(np.mean(top_rets)) if top_rets else np.nan
    avg_bottom = float(np.mean(bottom_rets)) if bottom_rets else np.nan
    spread = avg_top - avg_bottom

    # Overall directional accuracy
    X_all = test_df[feature_cols].to_numpy()
    y_all = test_df[target_col].to_numpy()
    preds_all = model.predict(X_all)
    sign_acc = np.mean(np.sign(preds_all) == np.sign(y_all))

    print(f"Average daily Spearman corr: {avg_cs:.4f}")
    print(f"Top decile 5d return:        {avg_top*100:.4f}%")
    print(f"Bottom decile 5d return:     {avg_bottom*100:.4f}%")
    print(f"Top-bottom spread:           {spread*100:.4f}%")
    print(f"Directional accuracy:        {sign_acc:.4f}")

    return {
        "spearman": avg_cs,
        "top": avg_top,
        "bottom": avg_bottom,
        "spread": spread,
        "sign_acc": float(sign_acc),
    }


# =========================================================
# Main
# =========================================================

def main():
    df, feature_cols = load_and_engineer()

    X_train, y_train, group_train, X_test, y_test, test_df = make_rank_data(
        df, feature_cols, target_col="fwd_ret_5d_cs", test_days=180
    )

    model = train_ranker(X_train, y_train, group_train)

    metrics = evaluate_ranker(model, test_df, feature_cols, target_col="fwd_ret_5d_cs")

    out_path = MODELS_DIR / "xgb_rank_fwd5d_cs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {out_path}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
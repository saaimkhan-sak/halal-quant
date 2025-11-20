import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "derived"
FEATURES_FILE = DATA_DIR / "features_daily.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    print(f"Loading features from {FEATURES_FILE} ...")
    df = pd.read_csv(FEATURES_FILE)

    # Parse date and sort by time + ticker
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print("Data shape:", df.shape)

    # Compute 5-day ahead return per ticker as target
    df["fwd_ret_5d"] = (
        df.groupby("ticker")["close"].shift(-5) / df["close"] - 1.0
    )

    # Drop rows where target is NaN (last 5 days of each ticker)
    df = df.dropna(subset=["fwd_ret_5d"]).reset_index(drop=True)
    print("After adding target & dropping NaNs on target:", df.shape)

    # Replace inf/-inf with NaN in all numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Drop any remaining rows with NaNs in numeric fields
    before = df.shape[0]
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped {before - after} rows with NaN/inf in numeric columns. Final shape: {df.shape}")

    return df


def make_train_test(df, test_days=180):
    # We'll use a time-based split: last `test_days` calendar days as test set
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    print(f"Max date in data: {max_date}, test cutoff: {cutoff}")

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Features: drop non-feature columns
    drop_cols = ["ticker", "date", "fwd_ret_5d"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["fwd_ret_5d"].values

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["fwd_ret_5d"].values

    # Ensure all columns are numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]  # align columns

    print("Final feature count:", X_train.shape[1])

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_xgb_regressor(X_train, y_train, X_valid, y_valid):
    # Basic XGBoost regressor with reasonable defaults
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist",
    }

    model = xgb.XGBRegressor(**params)

    # Set evaluation metric on the model (works across XGBoost versions)
    model.set_params(eval_metric="rmse")

    print("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=True,
    )
    return model

def evaluate(model, X_test, y_test):
    print("Evaluating on test set...")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Directional accuracy: how often sign(pred) matches sign(actual)
    sign_pred = np.sign(preds)
    sign_true = np.sign(y_test)
    directional_acc = (sign_pred == sign_true).mean()

    print(f"Test MSE: {mse:.6f}")
    print(f"Test R^2: {r2:.4f}")
    print(f"Directional accuracy (sign match): {directional_acc:.4f}")

    return preds


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, train_df, test_df = make_train_test(df, test_days=180)

    # Optionally, you can carve out a validation set from the training period:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    model = train_xgb_regressor(X_tr, y_tr, X_val, y_val)

    # Evaluate on held-out *future* period
    _ = evaluate(model, X_test, y_test)

    # Save model
    model_path = MODELS_DIR / "xgb_fwd5d.pkl"
    save_model(model, model_path)


if __name__ == "__main__":
    main()
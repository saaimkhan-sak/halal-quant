from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
FEATURES_IN = DERIVED / "features_daily.csv"
NEWS_SENT = DERIVED / "news_sentiment_headlines.csv"
FEATURES_OUT = DERIVED / "features_with_sentiment.csv"


def load_data():
    if not FEATURES_IN.exists():
        raise SystemExit(f"Features file not found: {FEATURES_IN}")
    if not NEWS_SENT.exists():
        raise SystemExit(f"News sentiment file not found: {NEWS_SENT}")

    feats = pd.read_csv(FEATURES_IN)
    feats["date"] = pd.to_datetime(feats["date"]).dt.date

    news = pd.read_csv(NEWS_SENT)
    news["date"] = pd.to_datetime(news["date"]).dt.date

    return feats, news


def aggregate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate daily sentiment per ticker
    agg = (
        news_df.groupby(["ticker", "date"])
        .agg(
            sent_pos_mean=("sent_pos", "mean"),
            sent_neg_mean=("sent_neg", "mean"),
            sent_neu_mean=("sent_neu", "mean"),
            sent_headline_count=("title", "count"),
        )
        .reset_index()
    )
    return agg


def merge_features(feats: pd.DataFrame, news_agg: pd.DataFrame) -> pd.DataFrame:
    merged = feats.merge(news_agg, how="left", on=["ticker", "date"])

    # Fill missing sentiment as "no news" -> neutral-ish
    merged["sent_pos_mean"] = merged["sent_pos_mean"].fillna(0.0)
    merged["sent_neg_mean"] = merged["sent_neg_mean"].fillna(0.0)
    merged["sent_neu_mean"] = merged["sent_neu_mean"].fillna(0.0)
    merged["sent_headline_count"] = merged["sent_headline_count"].fillna(0).astype(int)

    # Simple net sentiment feature
    merged["sent_net_pos"] = merged["sent_pos_mean"] - merged["sent_neg_mean"]

    DERIVED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FEATURES_OUT, index=False)
    print(f"Saved merged features with sentiment to {FEATURES_OUT}")
    return merged


def main():
    feats, news = load_data()
    print(f"Feature rows: {len(feats)}, News sentiment rows: {len(news)}")

    news_agg = aggregate_news(news)
    print(f"Aggregated sentiment rows: {len(news_agg)}")

    merge_features(feats, news_agg)


if __name__ == "__main__":
    main()
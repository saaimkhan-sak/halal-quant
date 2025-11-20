from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
NEWS_PATH = DERIVED / "news_raw_yf.csv"
OUT_PATH = DERIVED / "news_sentiment_headlines.csv"

MODEL_NAME = "ProsusAI/finbert"  # Financial sentiment model


def load_news():
    if not NEWS_PATH.exists():
        raise SystemExit(f"News file not found: {NEWS_PATH}")
    df = pd.read_csv(NEWS_PATH)
    # Make sure date and title are in usable formats
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["title"] = df["title"].fillna("").astype(str)
    return df


def setup_finbert():
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
    )
    return clf


def infer_sentiment(clf, texts, batch_size=8):
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        outputs = clf(batch, truncation=True, max_length=128)
        for out in outputs:
            # out is list of dicts: [{'label': 'positive', 'score': ...}, ...]
            scores = {d["label"].lower(): d["score"] for d in out}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)
            results.append((pos, neg, neu))
    return results


def main():
    df = load_news()
    print(f"Loaded {len(df)} headlines")

    clf = setup_finbert()

    sentiments = infer_sentiment(clf, df["title"].tolist(), batch_size=8)
    df["sent_pos"], df["sent_neg"], df["sent_neu"] = zip(*sentiments)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved sentiment-scored headlines to {OUT_PATH}")


if __name__ == "__main__":
    main()
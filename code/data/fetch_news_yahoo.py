import time
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"

ETF_UNION_FILE = DERIVED / "etf_union_us.csv"
OUT_PATH = DERIVED / "news_raw_yahoo_rss.csv"
DAYS_BACK = 365  # 1 year


# Use a real browser UA (e.g. from your actual browser)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


SAFE_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # start small


def fetch_rss(ticker, cutoff_date):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except Exception as e:
        print(f"[{ticker}] connection error: {e}")
        return []

    if resp.status_code == 429:
        print(f"[{ticker}] 429 (rate-limited). Sleeping 60s then retrying once...")
        time.sleep(60)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
        except Exception as e:
            print(f"[{ticker}] retry connection error: {e}")
            return []
        if resp.status_code == 429:
            print(f"[{ticker}] still 429 after retry, skipping for now.")
            return []

    if resp.status_code != 200:
        print(f"[{ticker}] HTTP {resp.status_code}, skipping")
        return []

    feed = feedparser.parse(resp.text)
    rows = []
    for entry in feed.entries:
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6])
        if not published:
            continue
        if published.date() < cutoff_date:
            continue

        rows.append(
            {
                "ticker": ticker,
                "published": published,
                "date": published.date(),
                "title": getattr(entry, "title", ""),
                "summary": getattr(entry, "summary", ""),
                "link": getattr(entry, "link", ""),
            }
        )
    return rows


def main():
    cutoff = datetime.utcnow().date() - timedelta(days=DAYS_BACK)
    all_rows = []

    print(f"Fetching Yahoo RSS for tickers: {SAFE_TICKERS}")

    for i, t in enumerate(SAFE_TICKERS, 1):
        print(f"[{i}/{len(SAFE_TICKERS)}] {t}")
        rows = fetch_rss(t, cutoff)
        all_rows.extend(rows)
        print(f"  -> got {len(rows)} headlines for {t}")
        time.sleep(3.0)  # be extra polite

    if not all_rows:
        print("No headlines fetched (still blocked or no data).")
        return

    df = pd.DataFrame(all_rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} RSS headlines to {OUT_PATH}")


if __name__ == "__main__":
    main()
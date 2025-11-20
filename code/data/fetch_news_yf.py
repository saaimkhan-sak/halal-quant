import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"

ETF_UNION_FILE = DERIVED / "etf_union_us.csv"
OUT_PATH = DERIVED / "news_raw_yf.csv"


def parse_pub_date_str(iso_str):
    """
    Parse ISO timestamp like '2025-11-18T11:00:42Z' into a date.
    """
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        return dt.date()
    except Exception:
        return None


def load_all_tickers():
    if not ETF_UNION_FILE.exists():
        raise SystemExit(f"Ticker file not found: {ETF_UNION_FILE}")
    df = pd.read_csv(ETF_UNION_FILE)
    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
    print(f"Loaded {len(tickers)} tickers from etf_union_us.csv")
    return tickers


def main():
    all_rows = []

    tickers = load_all_tickers()

    print(f"Fetching yfinance .news for {len(tickers)} tickers")

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}")
        try:
            tk = yf.Ticker(t)
            news_list = tk.news  # list of {'id':..., 'content':{...}}
        except Exception as e:
            print(f"  {t}: error getting news: {e}")
            news_list = []

        kept = 0
        for item in news_list or []:
            content = item.get("content", {})
            title = content.get("title", "")
            pubdate_str = content.get("pubDate")
            url_info = content.get("canonicalUrl", {}) or content.get("clickThroughUrl", {})
            link = url_info.get("url", "")

            if not title:
                continue

            pub_date = parse_pub_date_str(pubdate_str) if pubdate_str else None
            if pub_date is None:
                # fallback: treat as today's date so we don't drop everything
                pub_date = datetime.utcnow().date()

            all_rows.append(
                {
                    "ticker": t,
                    "date": pub_date,
                    "title": title,
                    "link": link,
                }
            )
            kept += 1

        print(f"  -> kept {kept} items for {t}")
        # Small sleep to be polite; yfinance.news is usually tolerant, but no need to hammer
        time.sleep(1.5)

    if not all_rows:
        print("No news items fetched via yfinance for any ticker.")
        return

    df = pd.DataFrame(all_rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} news items to {OUT_PATH}")


if __name__ == "__main__":
    main()
import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA"]

for t in tickers:
    print(f"=== {t} ===")
    tk = yf.Ticker(t)
    news_list = tk.news
    print(f"Number of items: {len(news_list) if news_list else 0}")
    if news_list:
        print("First item keys:", list(news_list[0].keys()))
        print("First item example:", news_list[0])
    print()
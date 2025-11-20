# Halal-Alpha: A Shariah-Compliant Quant Equity Research Platform  
**Author:** Saaim Khan  
**Affiliation:** Harvard Medical School  
**Repository:** https://github.com/saaimkhan-sak/halal-quant  

---

## 📌 Overview

**Halal-Alpha** is a fully self-taught, end-to-end quantitative equity strategy designed to operate under **strict Shariah-compliant constraints**.  
I built this project from scratch—without formal quant training—to learn how institutional quant platforms are engineered from data ingestion to portfolio construction to risk overlays.

This repository contains:

- A complete **data pipeline** for prices, fundamentals, and news sentiment  
- A **lookahead-proof ML feature set**  
- A cross-sectional **XGBoost ranking model**  
- **Portfolio construction** with name/sector caps  
- **Backtesting** with realistic turnover, transaction costs, and trading constraints  
- A **historical P&L simulation engine** with a volatility-targeting risk overlay  
- Comprehensive **diagnostics & analytics** (Sharpe, DD, rank stability, exposures, turnover, holdings)

The goal of this project is to demonstrate the problem-solving depth, technical ability, and research discipline required to build a robust quant framework suitable for real capital.

---

## 📂 Project Structure
halal-quant/
│
├── code/
│   ├── data/                     # Data ingestion, ETL, feature engineering
│   ├── signals/                  # ML model training, scoring, smoothing
│   ├── screening/                # Shariah-compliant universe filters
│   ├── trading/                  # Backtesting, PnL simulation, risk overlays
│   └── common/                   # Shared utilities
│
├── data/
│   ├── market/                   # Raw price data (gitignored)
│   ├── derived/                  # Engineered features, signals, PnL (gitignored)
│   └── compliance/               # Shariah screening inputs
│
├── models/                       # XGBoost models (gitignored)
│
├── figures/                      # Diagnostic plots
│
├── reports/                      # Notebooks/reports if added later
│
└── README.md

> **Note**: Large data files and models are intentionally excluded from GitHub via `.gitignore` to stay within GitHub limits and keep the repo clean.

---

## ⚙️ Technical Capabilities

### **1. Shariah-Compliant Universe Construction**
- ETFs: SPUS, HLAL, SPSK, UMMA, etc.
- Fundamental screens:
  - Debt-to-revenue ratio
  - Impermissible revenue categories (alcohol, pork, gambling, pornography, tobacco/vaping, cannabis, conventional finance)
- Merged into a historical universe of ~350–400 compliant names.

### **2. Data Pipeline**
Scripts under `code/data/` handle:

- Price panel construction  
- Fundamentals + debt metrics  
- News scraping & FinBERT sentiment  
- Feature engineering (returns, volatility, RSI, MACD, Bollinger Bands, ATR, volume z-scores)  
- Sector tagging  
- Deduplicated, cleaned, merged dataset  

All features are shifted one day using:

`features_with_sentiment_tminus1.csv`

to eliminate **lookahead bias**.

---

## 📈 **3. Machine Learning Model**
- Algorithm: **XGBoost Ranker / Regressor**  
- Objective: predict **relative forward 5-day returns** (cross-sectional)  
- Input features: 36 carefully chosen numeric features  
- Trained on rolling windows to avoid regime overfitting  
- Output stored as: signals_history.csv, signals_history_smooth.csv  (10-day rolling mean)

Rank stability analysis shows **daily rank correlation ~0.98**, indicating a slow-moving, stable signal—ideal for long-horizon portfolios.

---

## 🧮 **4. Portfolio Construction**

Monthly (20-day) rebalancing:

- Select **Top 150 names** by smoothed score  
- Apply:
  - **1% per-name cap**
  - **25% per-sector cap**
- Fully invested long-only book  
- Output: target_weights_history.csv

Sector exposure plots show broad diversification and smooth transitions across time.

---

## 📊 **5. Backtesting & PnL Simulation**

Located in `code/trading/simulate_pnl_from_weights.py`:

- Daily PnL simulation  
- Price-aware trade execution  
- **Transaction costs:** 10 bps per side  
- **Turnover cap:** 50% of portfolio value per rebalance  
- **Minimum trade notional:** $100  
- **Volatility overlay:** targets 18% annualized vol  
  - If realized vol exceeds target → risk scaled down automatically
- Outputs:
  - `pnl_daily.csv`
  - `positions_daily.csv`
  - `trade_log_backtest.csv`

---

## 📉 **6. Strategy Performance (2016–2025)**

Using the “production-style” variant:
- TOP150  
- Monthly rebalancing  
- 10-day smoothing  
- 18% vol targeting  

| Metric | Value |
|-------|-------|
| **Total Return** | +262.7% |
| **CAGR** | 13.98% |
| **Annualized Vol** | 16.99% |
| **Sharpe** | 0.856 |
| **Max Drawdown** | –37.4% |
| **Turnover** | ~25% annually |

Earlier variants (e.g., no overlay, weekly rebalancing) show higher return (~16–17%) but with higher risk (~–39% DD). Together, the variants form a realistic risk–return frontier.

---

## 📊 Diagnostics Included

- Equity curve + drawdown  
- Rolling Sharpe + rolling vol  
- Daily return distribution  
- Sector exposures over time  
- Rank stability  
- Holding period distribution (median ≈ 960 trading days)  
- Risk overlay scale usage  
- Trade logs and turnover statistics  

These tools helped debug problems (removing a hidden lookahead issue, fixing NaNs in price data, catching a negative PV bug) and build intuition about the strategy.

---

## 🧠 Philosophy and Problem-Solving Approach

While building Halal-Alpha, I treated each step as if I were the only quant and the only engineer on the team:

- **Everything must be reproducible**  
- **Every component must make sense economically and statistically**  
- **No step can rely on “magic” or black-box assumptions**  
- **If a metric looks too good, assume it’s a bug until proven real**  
- **If a model works, understand *why* it works**  

This project was built entirely self-taught, outside any formal quant internship, and reflects the mindset I would bring to an institutional systematic team.

---

## 🔗 **Code Availability**

All reproducible source code for Halal-Alpha is hosted here:

👉 **https://github.com/saaimkhan-sak/halal-quant**

Large datasets and models are excluded from GitHub due to size limits but can be regenerated using the included pipeline scripts.

---

## 📬 Contact

If you're interested in discussing this project, collaborating, or exploring halal-compliant alpha research, feel free to reach out.

**Email:** skhan@hms.harvard.edu  
**GitHub:** https://github.com/saaimkhan-sak  
**LinkedIn:** https://www.linkedin.com/in/saaim-khan-153953164/

---

*Halal-Alpha represents an ongoing exploration into high-integrity, Shariah-compliant systematic strategies. I welcome feedback from practitioners, portfolio managers, researchers, and MDs who have experience operating such strategies at scale.*

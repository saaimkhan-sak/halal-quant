import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
CONFIGS = ROOT / "configs"

ETF_UNION = DERIVED / "etf_union_us.csv"
OUT = DERIVED / "fundamentals_sp.csv"

# ---------- helpers to pull from yfinance ----------

def safe_first(series_or_none):
    if series_or_none is None or len(series_or_none) == 0:
        return np.nan
    return float(series_or_none.iloc[0])

def extract_total_revenue(ticker_obj) -> float:
    """
    Try to find total revenue from Yahoo income statement.
    """
    try:
        fin = ticker_obj.financials  # columns = most recent years/quarters
        if fin is None or fin.empty:
            return np.nan

        # Try exact label
        if "Total Revenue" in fin.index:
            return safe_first(fin.loc["Total Revenue"])

        # Fallback: any row containing 'revenue'
        rev_rows = [idx for idx in fin.index if "revenue" in idx.lower()]
        if rev_rows:
            return safe_first(fin.loc[rev_rows[0]])

    except Exception:
        pass
    return np.nan

def extract_total_debt(ticker_obj) -> float:
    """
    Estimate total debt as sum of any balance sheet rows that contain 'debt'.
    """
    try:
        bs = ticker_obj.balance_sheet
        if bs is None or bs.empty:
            return 0.0

        debt_rows = [idx for idx in bs.index if "debt" in idx.lower()]
        if not debt_rows:
            return 0.0

        total = 0.0
        for idx in debt_rows:
            try:
                total += float(bs.loc[idx].iloc[0])
            except Exception:
                continue
        return total
    except Exception:
        return 0.0

def extract_shares_outstanding(ticker_obj) -> float:
    """
    Try to get shares outstanding from .info.
    """
    try:
        info = ticker_obj.info
    except Exception:
        info = {}
    for key in ("sharesOutstanding", "floatShares"):
        v = info.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return np.nan

def get_sector_flags(ticker_obj) -> dict:
    """
    Roughly flag conventional finance using sector/industry.
    Other banned sectors are left False for now.
    """
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    is_fin = any(s in sector for s in ["financial", "finance", "bank"]) or \
             any(i in industry for i in ["bank", "insurance", "capital markets", "financ"])

    return {
        "alcohol": False,
        "pork": False,
        "gambling": False,
        "pornography": False,
        "tobacco_vaping": False,
        "recreational_cannabis": False,
        "impermissible_advertising": False,
        "impermissible_media": False,
        "deferred_gold_silver": False,
        "conventional_finance": is_fin,
    }

# ---------- main ----------

if __name__ == "__main__":
    # load ETF union tickers
    if not ETF_UNION.exists():
        raise SystemExit("Run code/data/etl_holdings.py first to build etf_union_us.csv")

    union = pd.read_csv(ETF_UNION)
    tickers = union["ticker"].dropna().astype(str).tolist()

    # load overrides if present
    try:
        overrides = pd.read_csv(CONFIGS / "involvement_overrides.csv")
    except Exception:
        overrides = pd.DataFrame(columns=[
            "ticker","estimated_impure_pct","alcohol","pork","gambling","pornography",
            "tobacco_vaping","recreational_cannabis","impermissible_advertising",
            "impermissible_media","deferred_gold_silver","conventional_finance","is_fully_islamic"
        ])

    rows = []
    for t in tickers:
        print(f"[{t}] fetching from Yahoo…")
        try:
            tk = yf.Ticker(t)
        except Exception:
            print(f"[{t}] error constructing yfinance Ticker, skipping")
            continue

        rev = extract_total_revenue(tk)
        debt = extract_total_debt(tk)
        shares = extract_shares_outstanding(tk)
        flags = get_sector_flags(tk)

        if np.isnan(rev) or np.isnan(shares) or rev == 0:
            print(f"[{t}] skipping (missing revenue or shares)")
            continue

        # merge overrides
        o = overrides[overrides["ticker"] == t]
        o = o.iloc[0] if len(o) else None

        def pick(flag: str) -> bool:
            if o is not None and flag in o and pd.notna(o[flag]):
                return str(o[flag]).strip().lower() == "true"
            return bool(flags.get(flag, False))

        merged_flags = {
            "alcohol": pick("alcohol"),
            "pork": pick("pork"),
            "gambling": pick("gambling"),
            "pornography": pick("pornography"),
            "tobacco_vaping": pick("tobacco_vaping"),
            "recreational_cannabis": pick("recreational_cannabis"),
            "impermissible_advertising": pick("impermissible_advertising"),
            "impermissible_media": pick("impermissible_media"),
            "deferred_gold_silver": pick("deferred_gold_silver"),
            "conventional_finance": pick("conventional_finance"),
        }

        is_islamic = False
        if o is not None and "is_fully_islamic" in o and pd.notna(o["is_fully_islamic"]):
            is_islamic = str(o["is_fully_islamic"]).strip().lower() == "true"

        # for now, non_permissible_revenue = 0; you can adjust with overrides later
        non_perm = 0.0
        if o is not None and "estimated_impure_pct" in o and pd.notna(o["estimated_impure_pct"]):
            try:
                pct = float(o["estimated_impure_pct"])
                non_perm = pct * rev
            except Exception:
                non_perm = 0.0

        rows.append({
            "ticker": t,
            "total_debt": debt,
            "shares_outstanding": shares,
            "is_fully_islamic": is_islamic,
            "non_permissible_revenue": non_perm,
            "total_revenue": rev,
            **merged_flags
        })

    out_df = pd.DataFrame(rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} with {len(out_df)} rows")
    print(out_df.head())
import os
import time
import requests
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Config / paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
CONFIGS = ROOT / "configs"

API_KEY = 'EVZIDOP2ORICJE9B'

BASE = "https://www.alphavantage.co/query"

# Free Alpha Vantage: ~5 calls/min → space calls ~12–13 seconds apart
SLEEP = 13

# For testing, you can cap how many tickers you process at once
# Set to None to try all tickers in etf_union_us.csv
MAX_TICKERS = 20  # e.g. 40 for faster test runs


# -------------------------------------------------------------------
# Low-level Alpha Vantage helper
# -------------------------------------------------------------------

def av_get(params: dict) -> dict | None:
    """Call Alpha Vantage with basic throttling/error handling.
       Returns parsed JSON dict, or None if throttled / error / no data."""
    params = {**params, "apikey": API_KEY}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # If AV is throttling or returns an error/info message, treat as no data
    if any(k in data for k in ("Note", "Information", "Error Message")):
        msg = data.get("Note") or data.get("Information") or data.get("Error Message")
        print(f"[AV WARNING] {msg}")
        return None

    return data


def overview(ticker: str) -> dict | None:
    return av_get({"function": "OVERVIEW", "symbol": ticker})


def income_stmt(ticker: str) -> dict | None:
    return av_get({"function": "INCOME_STATEMENT", "symbol": ticker})


def balance_sheet(ticker: str) -> dict | None:
    return av_get({"function": "BALANCE_SHEET", "symbol": ticker})


# -------------------------------------------------------------------
# Extractors from Alpha Vantage payloads
# -------------------------------------------------------------------

def get_shares(ov: dict) -> float | float:
    """Get shares outstanding from overview payload, or NaN if not available."""
    if ov is None:
        return float("nan")
    for k in ("SharesOutstanding", "SharesOutstandingTTM", "SharesFloat"):
        v = ov.get(k)
        if v and v != "None":
            try:
                return float(v)
            except Exception:
                continue
    return float("nan")


def latest_total_revenue(inc: dict) -> float:
    """Grab latest totalRevenue from income statement (annual or quarterly)."""
    if inc is None:
        return float("nan")

    for node in ("annualReports", "quarterlyReports"):
        if node in inc and inc[node]:
            try:
                return float(inc[node][0].get("totalRevenue") or 0.0)
            except Exception:
                continue
    return float("nan")


def interest_income_floor(inc: dict) -> float:
    """Use interestIncome as a conservative floor for impure income."""
    if inc is None:
        return 0.0

    for node in ("annualReports", "quarterlyReports"):
        if node in inc and inc[node]:
            val = inc[node][0].get("interestIncome")
            if val and val != "None":
                try:
                    return max(0.0, float(val))
                except Exception:
                    pass
    return 0.0


def total_debt_from_bs(bs: dict) -> float:
    """Approximate total debt as longTermDebt + shortTermDebt from balance sheet."""
    if bs is None:
        return 0.0

    lt, st = 0.0, 0.0
    for node in ("annualReports", "quarterlyReports"):
        if node in bs and bs[node]:
            r = bs[node][0]
            try:
                lt = float(r.get("longTermDebt") or 0.0)
                st = float(r.get("shortTermDebt") or 0.0)
            except Exception:
                pass
            break
    return lt + st


def get_sector_flags(ov: dict) -> dict:
    """Very rough classification: mark conventional finance based on sector/industry."""
    if ov is None:
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
            "conventional_finance": False,
        }

    sector = (ov.get("Sector") or "").lower()
    industry = (ov.get("Industry") or "").lower()

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


# -------------------------------------------------------------------
# Main: build fundamentals_sp.csv for ETF tickers
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 1) which tickers? → union of US tickers from your halal ETF holdings
    try:
        etf_union = pd.read_csv(DERIVED / "etf_union_us.csv")
        uni = etf_union["ticker"].tolist()
    except Exception:
        raise SystemExit("Run code/data/etl_holdings.py first to build etf_union_us.csv")

    # Optional cap for testing
    if MAX_TICKERS is not None:
        uni = uni[:MAX_TICKERS]
        print(f"[INFO] Limiting to first {len(uni)} tickers for this run")

    # 2) load overrides (manual involvement flags / estimated impure %)
    try:
        overrides = pd.read_csv(CONFIGS / "involvement_overrides.csv")
    except Exception:
        overrides = pd.DataFrame(columns=[
            "ticker", "estimated_impure_pct", "alcohol", "pork", "gambling", "pornography",
            "tobacco_vaping", "recreational_cannabis", "impermissible_advertising",
            "impermissible_media", "deferred_gold_silver", "conventional_finance", "is_fully_islamic"
        ])

    rows: list[dict] = []

    for t in uni:
        print(f"[{t}] OVERVIEW…")
        ov = overview(t)
        if ov is None:
            print(f"[{t}] skipping (no overview / throttled)")
            continue
        time.sleep(SLEEP)

        print(f"[{t}] INCOME_STATEMENT…")
        inc = income_stmt(t)
        if inc is None or not any(k in inc for k in ("annualReports", "quarterlyReports")):
            print(f"[{t}] skipping (no income statement)")
            continue
        time.sleep(SLEEP)

        print(f"[{t}] BALANCE_SHEET…")
        bs = balance_sheet(t)
        if bs is None or not any(k in bs for k in ("annualReports", "quarterlyReports")):
            print(f"[{t}] skipping (no balance sheet)")
            continue
        time.sleep(SLEEP)

        shares = get_shares(ov)
        rev = latest_total_revenue(inc)
        interest_floor = interest_income_floor(inc)
        debt = total_debt_from_bs(bs)
        flags = get_sector_flags(ov)

        # basic sanity check: we need shares and revenue to do anything useful
        if pd.isna(shares) or pd.isna(rev) or rev == 0:
            print(f"[{t}] skipping (missing shares or revenue)")
            continue

        # merge overrides for this ticker (if present)
        o = overrides[overrides["ticker"] == t]
        o = o.iloc[0] if len(o) else None
        est_pct = 0.0
        if o is not None and "estimated_impure_pct" in o and pd.notna(o["estimated_impure_pct"]):
            try:
                est_pct = float(o["estimated_impure_pct"])
            except Exception:
                est_pct = 0.0

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

        # impure revenue = interest floor + estimated_impure_pct * total_revenue
        impure = float(interest_floor) + float(est_pct) * float(rev or 0.0)

        rows.append({
            "ticker": t,
            "total_debt": debt,
            "shares_outstanding": shares,
            "is_fully_islamic": is_islamic,
            "non_permissible_revenue": impure,
            "total_revenue": rev,
            **merged_flags,
        })

    out_df = pd.DataFrame(rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out_path = DERIVED / "fundamentals_sp.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {out_path} with {len(out_df)} rows")
    print(out_df.head())
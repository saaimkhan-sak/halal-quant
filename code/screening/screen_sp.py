import pandas as pd
from pathlib import Path
from rules_sp import R
from mve import avg_mve_36m

FUND_CSV = "fundamentals_sp.csv"
OUT_UNIVERSE = "universe.csv"
OUT_REPORT = "screen_report.csv"

HIST_DIR = Path("compliance_history")
HIST_DIR.mkdir(exist_ok=True)
HIST_FILE = HIST_DIR / "leverage_history.csv"   # stores one row per monthly review per ticker

def sector_excluded(row) -> bool:
    # conventional finance excluded unless fully Islamic
    if str(row.get("conventional_finance","false")).lower() == "true" and \
       str(row.get("is_fully_islamic","false")).lower() != "true":
        return True
    for col in R.banned_categories:
        if str(row.get(col, "false")).lower() == "true":
            return True
    return False

def impure_ok(row) -> bool:
    rev = float(row["total_revenue"])
    imp = float(row["non_permissible_revenue"])
    return (rev > 0) and ((imp / rev) < R.max_impure_rev_ratio)

def raw_leverage_side(ratio: float) -> str:
    """
    Returns:
      'pass'  -> strictly inside the hard limit (< 0.33)
      'fail'  -> strictly beyond buffer (> 0.33 + 0.02 = 0.35)
      'buffer'-> in the gray zone [0.33, 0.35]
    """
    if ratio < R.max_leverage:
        return "pass"
    if ratio > (R.max_leverage + R.buffer):
        return "fail"
    return "buffer"

def decide_with_history(ticker: str, ratio: float, today: str) -> tuple[str, str]:
    """
    Apply S&P-style buffer + 3 consecutive rule using a simple history file.
    - If current side is 'buffer' => keep previous final status (or 'pass' if none).
    - If side is 'pass' or 'fail' (outside buffer):
        Flip only if this side occurs for 3 consecutive reviews (including today),
        otherwise keep previous final status.
    Returns (final_status, reason_string)
    """
    side = raw_leverage_side(ratio)

    # Load or init history
    if HIST_FILE.exists():
        hist = pd.read_csv(HIST_FILE)
    else:
        hist = pd.DataFrame(columns=["ticker","date","ratio","side","final_status"])

    # previous final status (default to 'pass' if none)
    prev_rows = hist[hist["ticker"] == ticker]
    prev_final = prev_rows.iloc[-1]["final_status"] if not prev_rows.empty else "pass"

    if side == "buffer":
        final_status = prev_final
        reason = f"buffer keep prev={prev_final} (ratio={ratio:.4f})"
    else:
        # Count consecutive occurrences of this side outside buffer
        last_two = prev_rows.tail(2)
        # compute sides for the last two ratios if missing (backward compat)
        if not last_two.empty and "side" not in last_two.columns:
            last_two = last_two.assign(side=last_two["ratio"].apply(raw_leverage_side))

        consec = 1  # today
        if len(last_two) >= 1 and last_two.iloc[-1]["side"] == side:
            consec += 1
        if len(last_two) >= 2 and last_two.iloc[-2]["side"] == side:
            consec += 1

        if consec >= R.consecutive_required:
            final_status = side
            reason = f"{side} 3/3 (ratio={ratio:.4f})"
        else:
            final_status = prev_final
            reason = f"{side} {consec}/3; keep prev={prev_final} (ratio={ratio:.4f})"

    # append today’s record
    new_row = pd.DataFrame([{
        "ticker": ticker, "date": today, "ratio": ratio,
        "side": side, "final_status": final_status
    }])
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist.to_csv(HIST_FILE, index=False)

    return final_status, reason

def leverage_ratio(ticker: str, total_debt: float, shares_outstanding: float) -> float:
    mve36 = avg_mve_36m(ticker, shares_outstanding)
    return float("inf") if mve36 <= 0 else float(total_debt) / float(mve36)

if __name__ == "__main__":
    f = pd.read_csv(FUND_CSV)
    num_cols = ["total_debt","shares_outstanding","non_permissible_revenue","total_revenue"]
    f[num_cols] = f[num_cols].apply(pd.to_numeric, errors="coerce")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    passed = []

    for _, row in f.iterrows():
        t = row["ticker"]

        # 1) sector/activity
        if sector_excluded(row):
            rows.append({"ticker": t, "status": "fail", "reason": "sector"})
            continue

        # 2) impure revenue < 5%
        if not impure_ok(row):
            rows.append({"ticker": t, "status": "fail", "reason": "impure>5%"})
            continue

        # 3) leverage with buffer + 3-consecutive
        ratio = leverage_ratio(t, row["total_debt"], row["shares_outstanding"])
        side_final, reason = decide_with_history(t, ratio, today)

        rows.append({"ticker": t, "status": side_final, "reason": reason})
        if side_final == "pass":
            passed.append(t)

    pd.DataFrame({"ticker": passed}).to_csv(OUT_UNIVERSE, index=False)
    pd.DataFrame(rows).to_csv(OUT_REPORT, index=False)

    print("Passed universe:", passed)
    print(f"Wrote {OUT_UNIVERSE} and {OUT_REPORT}")
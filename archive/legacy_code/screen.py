import pandas as pd
from rules import DEFAULT_RULES as R

def shariah_pass(row) -> bool:
    # Protect against divide-by-zero or missing data
    if row.total_assets <= 0 or row.total_revenue <= 0:
        return False

    debt_ratio = row.total_debt / row.total_assets
    cash_interest_ratio = row.cash / row.total_assets
    ar_cash_ratio = (row.cash + row.accounts_receivable) / row.total_assets
    impure_income_ratio = row.impure_income / row.total_revenue

    if debt_ratio >= R.max_debt_to_assets: return False
    if cash_interest_ratio >= R.max_cash_interest_to_assets: return False
    if ar_cash_ratio >= R.max_ar_plus_cash_to_assets: return False
    if impure_income_ratio > R.max_impure_income_ratio: return False
    return True

if __name__ == "__main__":
    f = pd.read_csv("fundamentals.csv")
    # enforce numeric types
    numeric_cols = ["total_assets","total_debt","cash","accounts_receivable","total_revenue","impure_income"]
    f[numeric_cols] = f[numeric_cols].apply(pd.to_numeric, errors="coerce")

    f["is_halal"] = f.apply(shariah_pass, axis=1)
    passed = f[f["is_halal"]].copy()

    passed[["ticker"]].to_csv("universe.csv", index=False)
    print("Screened universe:")
    print(passed[["ticker","is_halal"]])
    print("\nWrote universe.csv")
# file: code/trading/analyze_pnl.py

import pandas as pd
import numpy as np
from pathlib import Path

DATE_COL = "date"

PNL_FILENAME = "pnl_daily.csv"
TRADES_FILENAME = "trade_log_backtest.csv"
POSITIONS_FILENAME = "positions_daily.csv"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------- Metrics ----------------------


def compute_drawdown(cum_returns):
    """
    cum_returns is cumulative return series (1 + cumulative_pct, i.e. equity multiple).
    Returns: max_drawdown, drawdown_series
    """
    running_max = cum_returns.cummax()
    dd = (cum_returns - running_max) / running_max
    max_dd = dd.min()
    return max_dd, dd


def annualized_return(total_return, n_days):
    return (1 + total_return) ** (252 / n_days) - 1


def annualized_vol(daily_returns):
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns, rfr=0.0):
    """
    rfr = risk-free rate (daily). Set to 0 for simplicity.
    """
    er = daily_returns.mean()
    vol = daily_returns.std()
    if vol == 0:
        return np.nan
    return (er - rfr) / vol * np.sqrt(252)


def compute_turnover(trades_df, pnl_df):
    """
    daily turnover = total traded notional / portfolio_value
    Requires trades_df with columns: date, notional
    """
    if trades_df.empty:
        return 0.0, pd.Series(index=pnl_df[DATE_COL], data=0.0)

    # Aggregate trade notional per day
    trade_notional_by_day = (
        trades_df.groupby("date")["notional"]
        .sum()
        .astype(float)
    )

    merged = pnl_df.set_index("date").copy()
    merged["trade_notional"] = trade_notional_by_day
    merged["trade_notional"] = merged["trade_notional"].fillna(0.0)

    merged["turnover"] = merged["trade_notional"] / merged["portfolio_value"]

    # Turnover cannot be negative; clamp small numerical noise
    merged["turnover"] = merged["turnover"].clip(lower=0.0)

    avg_turnover = merged["turnover"].mean()
    return avg_turnover, merged["turnover"]


# ---------------------- Main Report ----------------------


def main():
    root = get_project_root()
    derived = root / "data" / "derived"

    pnl_path = derived / PNL_FILENAME
    trades_path = derived / TRADES_FILENAME

    print(f"Loading PnL: {pnl_path}")
    pnl = pd.read_csv(pnl_path, parse_dates=[DATE_COL])
    pnl = pnl.sort_values(DATE_COL)

    # Basic cumulative returns
    pnl["cum_return"] = (1 + pnl["daily_return"]).cumprod() - 1

    # Metrics
    total_return = pnl["cum_return"].iloc[-1]
    n_days = len(pnl)
    ann_ret = annualized_return(total_return, n_days)
    ann_vol = annualized_vol(pnl["daily_return"])
    sharpe = sharpe_ratio(pnl["daily_return"])

    max_dd, dd_series = compute_drawdown(1 + pnl["cum_return"])
    pnl["drawdown"] = dd_series.values

    # Turnover
    if trades_path.exists():
        trades = pd.read_csv(trades_path, parse_dates=["date"])
        avg_turnover, turnover_series = compute_turnover(trades, pnl)
    else:
        trades = None
        avg_turnover = None
        turnover_series = None

    # Exposure stats
    gross_exposure_mean = pnl["gross_exposure"].mean()
    gross_exposure_min = pnl["gross_exposure"].min()
    gross_exposure_max = pnl["gross_exposure"].max()

    # Monthly returns table
    pnl["month"] = pnl[DATE_COL].dt.to_period("M")
    monthly = pnl.groupby("month")["daily_return"].apply(
        lambda r: (1 + r).prod() - 1
    )

    # ------------ Print Report ------------
    print("\n=== PERFORMANCE REPORT ===\n")
    print(f"Start date: {pnl[DATE_COL].iloc[0].date()}")
    print(f"End date:   {pnl[DATE_COL].iloc[-1].date()}")
    print(f"Number of days: {n_days}\n")

    print(f"Total return:          {total_return: .4%}")
    print(f"Annualized return:     {ann_ret: .4%}")
    print(f"Annualized volatility: {ann_vol: .4%}")
    print(f"Sharpe ratio:          {sharpe: .3f}")
    print(f"Max drawdown:          {max_dd: .4%}\n")

    if avg_turnover is not None:
        print(f"Average daily turnover: {avg_turnover: .2%}")
    else:
        print("No trade file found; skipping turnover stats.")

    print("\nGross exposure:")
    print(f"   mean: {gross_exposure_mean:,.2f}")
    print(f"   min:  {gross_exposure_min:,.2f}")
    print(f"   max:  {gross_exposure_max:,.2f}")

    print("\n=== Monthly Returns ===")
    print(monthly.to_string())

    # Save extended report: drawdown, monthly stats, etc.
    out_report = derived / "pnl_report.csv"
    report_df = pnl[[DATE_COL, "daily_return", "cum_return", "drawdown"]].copy()
    report_df.to_csv(out_report, index=False)
    print(f"\nSaved detailed report to {out_report}")


if __name__ == "__main__":
    main()
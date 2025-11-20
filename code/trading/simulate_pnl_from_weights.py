# file: code/trading/simulate_pnl_from_weights.py

import numpy as np
import pandas as pd
from pathlib import Path

DATE_COL = "date"
TICKER_COL = "ticker"
PRICE_COL = "close"
WEIGHT_COL = "weight"

INITIAL_CAPITAL = 100_000.0

TRANSACTION_COST_BPS = 10.0    # 10 bps per side
MIN_TRADE_NOTIONAL = 100.0     # suppress tiny trades
MAX_DAILY_TURNOVER = 0.5       # cap 50% of portfolio per day

PRICE_FILENAME = "price_panel.csv"
WEIGHTS_FILENAME = "target_weights_history.csv"

OUT_PNL_FILENAME = "pnl_daily.csv"
OUT_POSITIONS_FILENAME = "positions_daily.csv"
OUT_TRADES_FILENAME = "trade_log_backtest.csv"

# ------- RISK OVERLAY CONFIG (VOL TARGETING) -------

RISK_OVERLAY_ENABLED = True
TARGET_ANNUAL_VOL = 0.18          # target annualized vol, e.g. 18%
RISK_WINDOW_DAYS = 63             # lookback window for realized vol (~3 months)
MIN_DAYS_FOR_RISK = 30            # minimum history before overlay activates


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_price_panel(path: Path) -> pd.DataFrame:
    print(f"Loading price panel from {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    missing = [c for c in [DATE_COL, TICKER_COL, PRICE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in price panel: {missing}")

    prices = (
        df
        .pivot(index=DATE_COL, columns=TICKER_COL, values=PRICE_COL)
        .sort_index()
    )

    # Treat zeros/NaNs as missing and fill
    prices = prices.replace(0, np.nan)
    n_bad = prices.isna().sum().sum()
    if n_bad > 0:
        print(f"Found {n_bad} NaN/zero price entries before fill; forward/backward filling.")
    prices = prices.ffill().bfill()

    all_nan_cols = prices.columns[prices.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Dropping {len(all_nan_cols)} tickers with no valid prices at all.")
        prices = prices.drop(columns=all_nan_cols)

    print(f"Price panel shape after cleaning: {prices.shape} (dates x tickers)")
    return prices


def load_weights_history(path: Path) -> pd.DataFrame:
    print(f"Loading target weights from {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    missing = [c for c in [DATE_COL, TICKER_COL, WEIGHT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in weights history: {missing}")

    df = df.sort_values([DATE_COL, TICKER_COL])

    weights = (
        df
        .pivot(index=DATE_COL, columns=TICKER_COL, values=WEIGHT_COL)
        .sort_index()
    )

    print(f"Weights history shape: {weights.shape} (rebalance dates x tickers)")
    return weights


def simulate_pnl(prices: pd.DataFrame, weights_history: pd.DataFrame):
    # align universe
    all_tickers = sorted(list(set(prices.columns) | set(weights_history.columns)))
    prices = prices.reindex(columns=all_tickers)
    weights_history = weights_history.reindex(columns=all_tickers).fillna(0.0)

    rebalance_dates = set(weights_history.index)
    dates = prices.index
    n_dates = len(dates)

    print(f"Simulation {dates[0].date()} -> {dates[-1].date()} ({n_dates} days)")
    print(f"Universe size: {len(all_tickers)}")
    print(f"Rebalance days: {len(rebalance_dates)}")

    positions = pd.Series(0.0, index=all_tickers)
    cash = INITIAL_CAPITAL
    prev_portfolio_value = INITIAL_CAPITAL

    pnl_records = []
    trades_records = []
    positions_daily_records = []

    # keep track of realized returns for risk overlay
    realized_returns = []
    last_risk_scale = 1.0

    for t, dt in enumerate(dates):
        price_today = prices.loc[dt]
        valid_mask = price_today > 0

        if not valid_mask.any():
            print(f"WARNING: No valid prices on {dt.date()}, skipping day.")
            daily_ret = 0.0
            portfolio_value = prev_portfolio_value
            gross_exposure = 0.0

            positions_daily_records.append(positions.copy().rename(dt))
            pnl_records.append(
                {
                    "date": dt,
                    "portfolio_value": float(portfolio_value),
                    "cash": float(cash),
                    "gross_exposure": float(gross_exposure),
                    "daily_return": float(daily_ret),
                    "risk_scale": float(last_risk_scale),
                }
            )
            continue

        price_today = price_today[valid_mask]
        positions = positions.reindex(price_today.index).fillna(0.0)

        position_values = positions * price_today
        gross_exposure = position_values.abs().sum()
        portfolio_value = cash + position_values.sum()

        if portfolio_value <= 0:
            raise RuntimeError(
                f"Non-positive portfolio value ({portfolio_value}) on {dt.date()} "
                f"before daily return computation."
            )

        if t == 0:
            daily_ret = 0.0
        else:
            daily_ret = portfolio_value / prev_portfolio_value - 1.0

        realized_returns.append(daily_ret)

        snapshot = positions.copy()
        snapshot.name = dt
        positions_daily_records.append(snapshot)

        # ----- RISK OVERLAY: compute risk_scale for this day -----
        risk_scale = 1.0
        if RISK_OVERLAY_ENABLED:
            hist = realized_returns[-RISK_WINDOW_DAYS:]
            if len(hist) >= MIN_DAYS_FOR_RISK:
                hist = np.array(hist)
                realized_vol = hist.std(ddof=1) * np.sqrt(252)
                if realized_vol > 0 and realized_vol > TARGET_ANNUAL_VOL:
                    risk_scale = TARGET_ANNUAL_VOL / realized_vol
                    # clamp to [0,1]
                    risk_scale = max(min(risk_scale, 1.0), 0.0)
        last_risk_scale = risk_scale

        # Rebalance
        if dt in rebalance_dates:
            target_weights_full = weights_history.loc[dt].fillna(0.0)

            # restrict to tickers we can price today
            target_weights = target_weights_full.reindex(price_today.index).fillna(0.0)

            total_abs = target_weights.abs().sum()
            if total_abs > 1.0 + 1e-6:
                print(
                    f"Warning {dt.date()}: |weights| sum={total_abs:.3f} >1; rescaling."
                )
                target_weights = target_weights / total_abs

            # apply risk overlay (downscale risk, leave excess in cash)
            if RISK_OVERLAY_ENABLED and risk_scale < 1.0:
                print(
                    f"{dt.date()} risk overlay: realized vol above target, "
                    f"risk_scale={risk_scale:.3f}"
                )
            target_weights = target_weights * risk_scale

            # recompute PV with latest prices
            position_values = positions * price_today
            portfolio_value = cash + position_values.sum()
            if portfolio_value <= 0:
                raise RuntimeError(
                    f"Non-positive portfolio value ({portfolio_value}) on {dt.date()} "
                    f"before rebalancing."
                )

            target_values = portfolio_value * target_weights
            target_shares = target_values / price_today

            raw_trades_shares = (target_shares - positions).fillna(0.0)
            raw_notional = (raw_trades_shares * price_today).abs()
            gross_trade_notional = raw_notional.sum()

            turnover = gross_trade_notional / portfolio_value if portfolio_value > 0 else 0.0

            scale = 1.0
            if MAX_DAILY_TURNOVER > 0 and turnover > MAX_DAILY_TURNOVER:
                scale = MAX_DAILY_TURNOVER / turnover
                print(
                    f"{dt.date()} turnover {turnover:.2%} > cap {MAX_DAILY_TURNOVER:.2%}; "
                    f"scaling trades by {scale:.3f}"
                )

            scaled_trades_shares = raw_trades_shares * scale
            scaled_notional = (scaled_trades_shares * price_today).abs()

            execute_mask = scaled_notional >= MIN_TRADE_NOTIONAL
            exec_trades_shares = scaled_trades_shares.where(execute_mask, 0.0)
            exec_notional = (exec_trades_shares * price_today).abs()

            txn_cost = (exec_notional.sum() * TRANSACTION_COST_BPS) / 10_000.0

            cash_flow = -(exec_trades_shares * price_today).sum()
            cash = cash + cash_flow - txn_cost
            positions = positions + exec_trades_shares

            position_values = positions * price_today
            portfolio_value = cash + position_values.sum()

            if portfolio_value <= 0:
                raise RuntimeError(
                    f"Non-positive portfolio value ({portfolio_value}) on {dt.date()} "
                    f"AFTER rebalancing."
                )

            if t == 0:
                daily_ret = 0.0
            else:
                daily_ret = portfolio_value / prev_portfolio_value - 1.0
                realized_returns[-1] = daily_ret  # update with post-trade PV

            # log trades
            for ticker in price_today.index:
                shares = exec_trades_shares.get(ticker, 0.0)
                if shares == 0.0:
                    continue
                side = "BUY" if shares > 0 else "SELL"
                notional = shares * price_today[ticker]
                trades_records.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "side": side,
                        "shares": float(shares),
                        "price": float(price_today[ticker]),
                        "notional": float(notional),
                    }
                )

        if not np.isfinite(portfolio_value):
            raise RuntimeError(
                f"Non-finite portfolio value on {dt.date()}: {portfolio_value}"
            )

        pnl_records.append(
            {
                "date": dt,
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "gross_exposure": float(gross_exposure),
                "daily_return": float(daily_ret),
                "risk_scale": float(last_risk_scale),
            }
        )
        prev_portfolio_value = portfolio_value

    pnl_df = pd.DataFrame(pnl_records).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(trades_records).sort_values(["date", "ticker"])
    positions_daily_df = pd.DataFrame(positions_daily_records)
    positions_daily_df.index.name = "date"

    return pnl_df, positions_daily_df, trades_df


def main():
    root = get_project_root()
    derived_dir = root / "data" / "derived"

    price_path = derived_dir / PRICE_FILENAME
    weights_path = derived_dir / WEIGHTS_FILENAME

    prices = load_price_panel(price_path)
    weights_history = load_weights_history(weights_path)

    start = max(prices.index.min(), weights_history.index.min())
    end = prices.index.max()

    prices = prices.loc[start:end]
    weights_history = weights_history.loc[weights_history.index >= start]

    pnl_df, positions_daily_df, trades_df = simulate_pnl(prices, weights_history)

    pnl_out = derived_dir / OUT_PNL_FILENAME
    pos_out = derived_dir / OUT_POSITIONS_FILENAME
    trades_out = derived_dir / OUT_TRADES_FILENAME

    pnl_df.to_csv(pnl_out, index=False)
    positions_daily_df.to_csv(pos_out)
    trades_df.to_csv(trades_out, index=False)

    print(f"Saved daily PnL to {pnl_out}")
    print(f"Saved daily positions to {pos_out}")
    print(f"Saved trade log to {trades_out}")


if __name__ == "__main__":
    main()
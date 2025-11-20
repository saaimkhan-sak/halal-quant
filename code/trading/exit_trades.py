import pandas as pd
from datetime import datetime
from pathlib import Path
from config import POSITIONS_CSV, TRADE_LOG_CSV, HOLD_DAYS
from utils import ensure_csv, latest_price_and_date, today_stamp, load_close_series

def trading_days_held(ticker: str, entry_date_str: str) -> int:
    dates, _ = load_close_series(ticker)
    entry_date = pd.to_datetime(entry_date_str)
    # count bars strictly after entry_date up to the last available date
    held_days = dates[dates > entry_date].shape[0]
    return int(held_days)

if __name__ == "__main__":
    ensure_csv(POSITIONS_CSV, ["ticker","entry_date","entry_price","qty"])
    ensure_csv(TRADE_LOG_CSV, ["timestamp","action","ticker","qty","price","z","note","pnl"])

    positions = pd.read_csv(POSITIONS_CSV)
    if positions.empty:
        print("No open positions to exit.")
        raise SystemExit

    keep_rows = []
    log_rows = []
    exits = 0

    for _, pos in positions.iterrows():
        t = pos["ticker"]
        entry_date = pos["entry_date"]
        entry_price = float(pos["entry_price"])
        qty = int(pos["qty"])

        days = trading_days_held(t, entry_date)
        if days >= HOLD_DAYS:
            # exit at latest close
            px, mkt_date = latest_price_and_date(t)
            pnl = (px - entry_price) * qty
            log_rows.append({"timestamp": today_stamp(), "action":"SELL", "ticker": t,
                             "qty": qty, "price": px, "z": "", "note":f"exit after {days} bars",
                             "pnl": round(pnl, 2)})
            exits += 1
        else:
            keep_rows.append(pos)

    new_positions = pd.DataFrame(keep_rows, columns=positions.columns)
    new_positions.to_csv(POSITIONS_CSV, index=False)

    log = pd.read_csv(TRADE_LOG_CSV)
    if log_rows:
        log = pd.concat([log, pd.DataFrame(log_rows)], ignore_index=True)
        log.to_csv(TRADE_LOG_CSV, index=False)

    if exits:
        print(f"Exited {exits} position(s).")
    else:
        print("No exits today (not yet at hold horizon).")
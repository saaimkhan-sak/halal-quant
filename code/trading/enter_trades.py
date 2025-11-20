import pandas as pd
from pathlib import Path
from config import (CANDIDATES_CSV, POSITIONS_CSV, TRADE_LOG_CSV,
                    BUY_Z_THRESHOLD, MAX_NEW_POSITIONS, DOLLARS_PER_TRADE)
from utils import ensure_csv, latest_price_and_date, shares_for_budget, today_stamp

if __name__ == "__main__":
    ensure_csv(POSITIONS_CSV, ["ticker","entry_date","entry_price","qty"])
    ensure_csv(TRADE_LOG_CSV, ["timestamp","action","ticker","qty","price","z","note","pnl"])

    # load current positions and candidates
    positions = pd.read_csv(POSITIONS_CSV)
    held = set(positions["ticker"].tolist())

    cands = pd.read_csv(CANDIDATES_CSV)
    # keep only BUYs by threshold; sort by most negative z (strongest)
    buy_list = cands[cands["z"] < BUY_Z_THRESHOLD].sort_values("z").head(MAX_NEW_POSITIONS)

    rows_added = 0
    log_rows = []
    pos_rows = []

    for _, row in buy_list.iterrows():
        t = row["ticker"]
        z = float(row["z"])
        if t in held:
            continue  # already holding

        price, mkt_date = latest_price_and_date(t)
        qty = shares_for_budget(price, DOLLARS_PER_TRADE)
        if qty == 0:
            continue

        # append to positions
        pos_rows.append({"ticker": t, "entry_date": mkt_date.strftime("%Y-%m-%d"),
                         "entry_price": price, "qty": qty})
        # log
        log_rows.append({"timestamp": today_stamp(), "action":"BUY", "ticker": t,
                         "qty": qty, "price": price, "z": z, "note":"enter", "pnl": ""})
        rows_added += 1

    if rows_added > 0:
        positions = pd.concat([positions, pd.DataFrame(pos_rows)], ignore_index=True)
        positions.to_csv(POSITIONS_CSV, index=False)
        log = pd.read_csv(TRADE_LOG_CSV)
        log = pd.concat([log, pd.DataFrame(log_rows)], ignore_index=True)
        log.to_csv(TRADE_LOG_CSV, index=False)
        print(f"Entered {rows_added} trade(s).")
    else:
        print("No entries today (no BUY candidates or already holding).")
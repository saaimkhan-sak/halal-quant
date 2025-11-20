# file: code/trading/plot_sector_exposures.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATE_COL = "date"
TICKER_COL = "ticker"
SECTOR_COL = "sector"

POSITIONS_FILENAME = "positions_daily.csv"
PRICE_PANEL_FILENAME = "price_panel.csv"
FEATURES_TMINUS1_FILENAME = "features_with_sentiment_tminus1.csv"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_figures_dir(root: Path) -> Path:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def build_sector_map(features_path: Path) -> dict:
    """
    Build a dict: ticker -> sector
    Use first non-null sector per ticker.
    """
    print(f"Loading sector info from {features_path}")
    df = pd.read_csv(features_path, parse_dates=[DATE_COL])

    if SECTOR_COL not in df.columns:
        raise ValueError(f"{SECTOR_COL} column not found in {features_path}")

    df = df[[TICKER_COL, SECTOR_COL]].dropna().drop_duplicates()
    sector_map = df.set_index(TICKER_COL)[SECTOR_COL].to_dict()
    print(f"Built sector map for {len(sector_map)} tickers.")
    return sector_map


def main():
    root = get_project_root()
    derived = root / "data" / "derived"
    figures_dir = ensure_figures_dir(root)

    positions_path = derived / POSITIONS_FILENAME
    prices_path = derived / PRICE_PANEL_FILENAME
    features_path = derived / FEATURES_TMINUS1_FILENAME

    # Load positions (index = date, columns = tickers, values = shares)
    print(f"Loading positions from {positions_path}")
    positions = pd.read_csv(positions_path, index_col=0, parse_dates=True)
    positions.index.name = DATE_COL

    # Load prices
    print(f"Loading price panel from {prices_path}")
    price_df = pd.read_csv(prices_path, parse_dates=[DATE_COL])
    prices = (
        price_df
        .pivot(index=DATE_COL, columns=TICKER_COL, values="close")
        .sort_index()
    )

    # Align dates
    common_dates = positions.index.intersection(prices.index)
    positions = positions.loc[common_dates]
    prices = prices.loc[common_dates]

    # Align tickers
    common_tickers = positions.columns.intersection(prices.columns)
    positions = positions[common_tickers]
    prices = prices[common_tickers]

    # Load sector map
    sector_map = build_sector_map(features_path)

    # Map tickers -> sectors; any missing get "Unknown"
    ticker_sector = {
        t: sector_map.get(t, "Unknown")
        for t in common_tickers
    }

    # Compute sector weights by date
    sector_weights_list = []

    for dt in common_dates:
        pos_row = positions.loc[dt]
        price_row = prices.loc[dt]

        # Dollar exposure per ticker
        exposure = pos_row * price_row
        total_exposure = exposure.sum()

        if total_exposure <= 0:
            # all cash or no positions
            sector_weights_list.append({DATE_COL: dt})
            continue

        # Group by sector
        df_e = pd.DataFrame({
            "ticker": common_tickers,
            "exposure": exposure.values,
        })
        df_e[SECTOR_COL] = df_e["ticker"].map(ticker_sector)
        sector_exposure = df_e.groupby(SECTOR_COL)["exposure"].sum()

        sector_weight = sector_exposure / total_exposure

        row = {DATE_COL: dt}
        row.update(sector_weight.to_dict())
        sector_weights_list.append(row)

    sector_weights = pd.DataFrame(sector_weights_list).set_index(DATE_COL)
    sector_weights = sector_weights.fillna(0.0)

    # Focus on top sectors by average weight
    avg_weights = sector_weights.mean().sort_values(ascending=False)
    top_sectors = list(avg_weights.head(8).index)

    other_cols = [c for c in sector_weights.columns if c not in top_sectors]
    if other_cols:
        sector_weights["Other"] = sector_weights[other_cols].sum(axis=1)
    sector_weights_top = sector_weights[top_sectors + (["Other"] if other_cols else [])]

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_weights_top.plot.area(ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Sector Exposures Over Time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = figures_dir / "sector_exposures_over_time.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
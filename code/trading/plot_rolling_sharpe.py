# file: code/trading/plot_rolling_sharpe.py

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATE_COL = "date"
PNL_FILENAME = "pnl_daily.csv"

ROLLING_WINDOW_DAYS = 63  # ~3 months of trading days


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_figures_dir(root: Path) -> Path:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def main():
    root = get_project_root()
    derived = root / "data" / "derived"
    figures_dir = ensure_figures_dir(root)

    pnl_path = derived / PNL_FILENAME
    print(f"Loading PnL from {pnl_path}")
    df = pd.read_csv(pnl_path, parse_dates=[DATE_COL]).sort_values(DATE_COL)

    # Rolling Sharpe and vol
    r = df["daily_return"]

    rolling_vol = r.rolling(ROLLING_WINDOW_DAYS).std() * np.sqrt(252)
    rolling_mean = r.rolling(ROLLING_WINDOW_DAYS).mean() * 252  # annualized mean
    rolling_sharpe = rolling_mean / (rolling_vol.replace(0, np.nan))

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df[DATE_COL], rolling_sharpe, label="Rolling Sharpe (63d)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rolling Sharpe")
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df[DATE_COL], rolling_vol, label="Rolling Vol (annualized)", alpha=0.5)
    ax2.set_ylabel("Rolling Volatility")

    plt.title("Rolling 3-Month Sharpe and Volatility")
    fig.tight_layout()

    out_path = figures_dir / "rolling_sharpe_and_vol.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
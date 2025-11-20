# file: code/trading/plot_return_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATE_COL = "date"
PNL_FILENAME = "pnl_daily.csv"


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

    r = df["daily_return"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(r, bins=50)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Daily Returns")
    ax.grid(True, alpha=0.3)

    out_path = figures_dir / "daily_return_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
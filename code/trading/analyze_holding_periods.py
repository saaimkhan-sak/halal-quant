# file: code/trading/analyze_holding_periods.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATE_COL = "date"


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

    positions_path = derived / "positions_daily.csv"
    print(f"Loading positions from {positions_path}")
    pos = pd.read_csv(positions_path, index_col=0, parse_dates=True)
    pos.index.name = DATE_COL

    dates = pos.index
    tickers = pos.columns

    holding_lengths = []

    for ticker in tickers:
        series = pos[ticker].fillna(0.0)
        in_pos = False
        length = 0

        for val in series:
            if not in_pos and val != 0:
                # start new holding
                in_pos = True
                length = 1
            elif in_pos and val != 0:
                length += 1
            elif in_pos and val == 0:
                # end holding
                holding_lengths.append(length)
                in_pos = False
                length = 0
        # if still in a position at the end, count that episode too
        if in_pos and length > 0:
            holding_lengths.append(length)

    holding_lengths = np.array(holding_lengths)
    if holding_lengths.size == 0:
        print("No non-zero positions found; cannot compute holding periods.")
        return

    print(f"Computed {len(holding_lengths)} holding episodes.")

    # Summary stats
    s = pd.Series(holding_lengths)
    print("\nHolding period (days) summary:")
    print(s.describe())
    for p in [10, 25, 50, 75, 90]:
        print(f"{p}th percentile: {np.percentile(holding_lengths, p):.2f} days")

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(holding_lengths, bins=50)
    ax.set_xlabel("Holding period (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Holding Periods")
    ax.grid(True, alpha=0.3)

    out_path = figures_dir / "holding_period_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved holding period distribution to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
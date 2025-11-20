# file: code/trading/plot_equity_curve.py

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATE_COL = "date"
REPORT_FILENAME = "pnl_report.csv"

# -----------------------------

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def ensure_figures_dir(root: Path) -> Path:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir

# -----------------------------


def main():
    root = get_project_root()
    derived = root / "data" / "derived"
    figures_dir = ensure_figures_dir(root)

    report_path = derived / REPORT_FILENAME
    print(f"Loading PnL report from {report_path}")

    df = pd.read_csv(report_path, parse_dates=[DATE_COL]).sort_values(DATE_COL)

    # Equity curve (using initial capital = 100k)
    initial_capital = 100_000.0
    df["equity"] = (1 + df["cum_return"]) * initial_capital

    # ----- Plot -----
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df[DATE_COL], df["equity"], label="Equity Curve", color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value (Equity)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df[DATE_COL], df["drawdown"], label="Drawdown", color="red", alpha=0.6)
    ax2.set_ylabel("Drawdown")

    plt.title("Equity Curve and Drawdown Over Time")
    fig.tight_layout()

    # ----- Save figure -----
    output_path = figures_dir / "equity_curve_and_drawdown.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    # Optional: show plot
    plt.show()


if __name__ == "__main__":
    main()
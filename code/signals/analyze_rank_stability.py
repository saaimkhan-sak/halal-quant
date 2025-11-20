# file: code/signals/analyze_rank_stability.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATE_COL = "date"
TICKER_COL = "ticker"
SCORE_COL = "score"
SMOOTH_COL = "score_smooth"

SIGNALS_FILENAME_PRIMARY = "signals_history_smooth.csv"
SIGNALS_FILENAME_FALLBACK = "signals_history.csv"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_figures_dir(root: Path) -> Path:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def load_signals(derived_dir: Path) -> pd.DataFrame:
    primary = derived_dir / SIGNALS_FILENAME_PRIMARY
    fallback = derived_dir / SIGNALS_FILENAME_FALLBACK

    if primary.exists():
        path = primary
    elif fallback.exists():
        path = fallback
    else:
        raise FileNotFoundError(
            f"Neither {SIGNALS_FILENAME_PRIMARY} nor {SIGNALS_FILENAME_FALLBACK} found in {derived_dir}"
        )

    print(f"Loading signals from {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    required = [DATE_COL, TICKER_COL, SCORE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Signals file missing required columns: {missing}")

    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


def main():
    root = get_project_root()
    derived = root / "data" / "derived"
    figures_dir = ensure_figures_dir(root)

    df = load_signals(derived)

    use_col = SMOOTH_COL if SMOOTH_COL in df.columns else SCORE_COL
    print(f"Using '{use_col}' for rank stability analysis.")

    dates = sorted(df[DATE_COL].unique())

    rank_corr_records = []
    all_rank_changes = []

    prev_ranks = None
    prev_date = None

    for dt in dates:
        df_day = df[df[DATE_COL] == dt].copy()
        # Rank: lowest rank = worst, highest = best
        df_day["rank"] = df_day[use_col].rank(method="dense")

        ranks = df_day.set_index(TICKER_COL)["rank"]

        if prev_ranks is not None:
            common_tickers = ranks.index.intersection(prev_ranks.index)
            if len(common_tickers) > 0:
                r_today = ranks.loc[common_tickers]
                r_prev = prev_ranks.loc[common_tickers]

                # Pearson correlation of ranks
                corr = np.corrcoef(r_today.values, r_prev.values)[0, 1]
                rank_corr_records.append({"date": dt, "rank_corr": corr})

                # Rank changes
                rank_change = r_today - r_prev
                all_rank_changes.extend(rank_change.values.tolist())
        prev_ranks = ranks
        prev_date = dt

    # Convert correlation records
    rank_corr_df = pd.DataFrame(rank_corr_records).sort_values("date")

    # ---- Plot rank correlation over time ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rank_corr_df["date"], rank_corr_df["rank_corr"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank correlation vs previous day")
    ax.set_title("Rank Stability Over Time")
    ax.grid(True, alpha=0.3)

    rank_corr_path = figures_dir / "rank_stability_over_time.png"
    fig.tight_layout()
    fig.savefig(rank_corr_path, dpi=300)
    print(f"Saved rank stability figure to {rank_corr_path}")
    plt.show()

    # ---- Plot rank change distribution ----
    all_rank_changes = np.array(all_rank_changes)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(all_rank_changes, bins=50)
    ax2.set_xlabel("Daily rank change (r_today - r_prev)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Daily Rank Changes")
    ax2.grid(True, alpha=0.3)

    rank_change_path = figures_dir / "rank_change_distribution.png"
    fig2.tight_layout()
    fig2.savefig(rank_change_path, dpi=300)
    print(f"Saved rank change distribution figure to {rank_change_path}")
    plt.show()

    # Print some summary stats
    print("\nRank correlation summary:")
    print(rank_corr_df["rank_corr"].describe())

    print("\nRank change summary:")
    print(pd.Series(all_rank_changes).describe())


if __name__ == "__main__":
    main()
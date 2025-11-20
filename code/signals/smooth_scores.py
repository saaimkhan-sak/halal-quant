# file: code/signals/smooth_scores.py

import pandas as pd
from pathlib import Path
import sys

# ---------- CONFIG ----------

DATE_COL = "date"
TICKER_COL = "ticker"
SCORE_COL = "score"          # change to your actual signal column name

# Default filenames (we'll fall back from history -> latest)
PRIMARY_INPUT = "signals_history.csv"
FALLBACK_INPUT = "signals_latest.csv"

OUTPUT_SUFFIX = "_smooth"    # appended before .csv

# choose smoothing: rolling or EMA
USE_ROLLING_MEAN = True
ROLLING_WINDOW = 10   # in trading days

USE_EMA = False      # set to True to use EMA instead
EMA_ALPHA = 0.3

# ----------------------------


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_input_file(derived_dir: Path) -> Path:
    """
    Try PRIMARY_INPUT first; if missing but FALLBACK_INPUT exists, use that.
    """
    primary = derived_dir / PRIMARY_INPUT
    fallback = derived_dir / FALLBACK_INPUT

    if primary.exists():
        print(f"Using primary signals file: {primary}")
        return primary

    if fallback.exists():
        print(
            f"WARNING: {primary.name} not found. "
            f"Falling back to {fallback.name} (likely only latest day)."
        )
        return fallback

    # Neither file exists -> tell user clearly what to do
    msg_lines = [
        f"ERROR: Neither {PRIMARY_INPUT} nor {FALLBACK_INPUT} exist in {derived_dir}.",
        "",
        "You have a few options:",
        f"  1) Create {PRIMARY_INPUT} as a historical signals file with columns:",
        f"       {DATE_COL},{TICKER_COL},{SCORE_COL}",
        f"  2) Or rename your existing signals file to {PRIMARY_INPUT},",
        "     and ensure it has those columns.",
        f"  3) Or change PRIMARY_INPUT/FALLBACK_INPUT in smooth_scores.py to match your file.",
    ]
    print("\n".join(msg_lines))
    sys.exit(1)


def main():
    root = get_project_root()
    derived_dir = root / "data" / "derived"

    src = resolve_input_file(derived_dir)
    dst = src.with_name(src.stem + OUTPUT_SUFFIX + src.suffix)

    print(f"Loading signals from {src}")
    df = pd.read_csv(src, parse_dates=[DATE_COL])

    missing = [c for c in [DATE_COL, TICKER_COL, SCORE_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {src.name}: {missing}\n"
            f"Expected at least: {DATE_COL}, {TICKER_COL}, {SCORE_COL}"
        )

    df = df.sort_values([TICKER_COL, DATE_COL])

    if USE_ROLLING_MEAN:
        print(f"Applying {ROLLING_WINDOW}-day rolling mean on '{SCORE_COL}'")
        df["score_smooth"] = (
            df
            .groupby(TICKER_COL)[SCORE_COL]
            .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
    elif USE_EMA:
        print(f"Applying EMA (alpha={EMA_ALPHA}) on '{SCORE_COL}'")
        df["score_smooth"] = (
            df
            .groupby(TICKER_COL)[SCORE_COL]
            .transform(lambda s: s.ewm(alpha=EMA_ALPHA, adjust=False).mean())
        )
    else:
        raise ValueError("Either USE_ROLLING_MEAN or USE_EMA must be True.")

    df.to_csv(dst, index=False)
    print(f"Saved smoothed signals to {dst}")


if __name__ == "__main__":
    main()

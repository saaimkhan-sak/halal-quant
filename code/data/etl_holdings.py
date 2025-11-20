import re, glob
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDINGS_DIR = ROOT / "data" / "holdings"
OUT = ROOT / "data" / "derived" / "etf_union_us.csv"

# filenames to skip entirely
SKIP_FILES = {"spsk-holdings.csv"}

BASE_CANDIDATES = ["ticker", "symbol", "code"]

def pick_ticker_column(df: pd.DataFrame, filename: str) -> str:
    cols = list(df.columns)
    lower_map = {c: c.lower().strip() for c in cols}

    # 1) direct matches
    for orig, low in lower_map.items():
        if any(key in low for key in BASE_CANDIDATES):
            return orig

    # 2) heuristic: look for many ticker-looking values
    for c in cols:
        if df[c].dtype == object:
            frac = df[c].astype(str).str.match(r"^[A-Za-z0-9\.\-]{1,8}$").mean()
            if frac > 0.2:
                print(f"[WARN] Guessing column '{c}' is ticker in {filename}")
                return c

    # 3) fallback: first object column
    print(f"[ERROR] Could not confidently find ticker column in {filename}")
    print(" → Columns:", cols)
    obj_cols = [c for c in cols if df[c].dtype == object]
    if obj_cols:
        print(f"[FALLBACK] Using '{obj_cols[0]}' as ticker")
        return obj_cols[0]

    raise ValueError(f"No usable ticker column found in {filename}")

def normalize_us_ticker(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().upper()
    t = t.replace("/", ".")               # BRK/B → BRK.B
    t = re.sub(r"\s*\([A-Z]+\)$", "", t)  # remove "(USD)" "(STOCK)" etc
    return t

def looks_us(t: str) -> bool:
    # match simple US-style tickers: ABC or ABCD or ABCD.X
    return bool(re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", t))

if __name__ == "__main__":
    files = glob.glob(str(HOLDINGS_DIR / "*-holdings.csv"))
    if not files:
        raise SystemExit(f"No holdings CSVs found in {HOLDINGS_DIR}")

    all_tickers = []

    for f in files:
        filename = Path(f).name
        if filename in SKIP_FILES:
            print(f"[INFO] Skipping {filename} (no tickers)")
            continue

        df = pd.read_csv(f)
        col = pick_ticker_column(df, filename)
        series = df[col].astype(str).map(normalize_us_ticker)
        tickers = [t for t in series if t and looks_us(t)]
        all_tickers.extend(tickers)

    union = sorted(set(all_tickers))
    out_df = pd.DataFrame({"ticker": union, "label_shariah_etf": 1})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)

    print(f"\nWrote {OUT} with {len(union)} US tickers")

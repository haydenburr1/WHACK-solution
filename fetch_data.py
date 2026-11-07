"""Download and cache daily prices for the Market Worlds stock universe.

Run:  python fetch_data.py
Outputs (in ./data):
  universe.csv    - ticker, name, sector, sub_industry, avg_dollar_volume  (commit this)
  prices.parquet  - adjusted close, one row per trading day, one column per ticker (gitignore this)
Delete prices.parquet to force a re-download; delete universe.csv as well to rebuild the stock list.

run Tobias:
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/fetch_data.py

"""
import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
PRICES_FILE = DATA_DIR / "prices.parquet"

START = "2017-06-01"   # ~6 months of warm-up so the first 2018 frame has correlation history
END = "2026-01-01"     # yfinance treats end as exclusive

# One sector per cube face. GICS names exactly as Wikipedia spells them.
SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Industrials",
    "Energy",
]
MAX_PER_SECTOR = 50    # keep the most heavily traded N stocks per sector
BATCH_SIZE = 50
MAX_GAP_DAYS = 2       # forward-fill gaps up to this long; drop tickers with longer gaps

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_universe() -> pd.DataFrame:
    """Use the committed universe.csv if present, otherwise scrape the S&P 500 list."""
    if UNIVERSE_CSV.exists():
        print(f"Using existing {UNIVERSE_CSV.name}")
        return pd.read_csv(UNIVERSE_CSV)

    print("Fetching S&P 500 constituents from Wikipedia...")
    resp = requests.get(
        WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (Market Worlds hackathon project)"},
        timeout=30,
    )
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]

    uni = table.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })[["ticker", "name", "sector", "sub_industry"]]
    uni["ticker"] = uni["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    return uni[uni["sector"].isin(SECTORS)].reset_index(drop=True)


def download(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close and volume in batches, with retries."""
    closes, volumes = [], []
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for b, i in enumerate(range(0, len(tickers), BATCH_SIZE), start=1):
        batch = tickers[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                df = yf.download(batch, start=START, end=END,
                                 auto_adjust=True, progress=False, threads=True)
                if df.empty:
                    raise RuntimeError("empty response")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  batch {b} failed ({e}), retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"batch {b} failed 3 times: {batch}")

        closes.append(df["Close"])
        volumes.append(df["Volume"])
        print(f"batch {b}/{n_batches} done ({len(batch)} tickers)")
        time.sleep(1)  # be polite to Yahoo

    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    return close, volume


def clean(close: pd.DataFrame, volume: pd.DataFrame, uni: pd.DataFrame):
    """Fill short gaps, drop incomplete tickers, keep the top N per sector by liquidity."""
    close = close.dropna(how="all")              # rows where nothing traded
    close = close.ffill(limit=MAX_GAP_DAYS)
    complete = close.columns[close.notna().all()]
    dropped = sorted(set(uni["ticker"]) - set(complete))
    if dropped:
        print(f"Dropping {len(dropped)} tickers with missing data: {', '.join(dropped)}")
    close = close[complete]

    volume = volume.reindex(index=close.index, columns=close.columns)
    dollar_vol = (close * volume).mean()

    uni = uni[uni["ticker"].isin(close.columns)].copy()
    uni["avg_dollar_volume"] = uni["ticker"].map(dollar_vol)
    uni = (uni.sort_values("avg_dollar_volume", ascending=False)
              .groupby("sector").head(MAX_PER_SECTOR)
              .sort_values(["sector", "sub_industry", "ticker"])
              .reset_index(drop=True))

    close = close[uni["ticker"]]
    close.index.name = "date"
    return close, uni


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if PRICES_FILE.exists() and UNIVERSE_CSV.exists():
        close = pd.read_parquet(PRICES_FILE)
        print(f"Cached: {close.shape[1]} tickers, {close.shape[0]} days "
              f"({close.index[0].date()} to {close.index[-1].date()})")
        return

    uni = load_universe()
    print(f"{len(uni)} candidate tickers across {uni['sector'].nunique()} sectors")

    close, volume = download(uni["ticker"].tolist())
    close, uni = clean(close, volume, uni)

    close.to_parquet(PRICES_FILE)
    uni.to_csv(UNIVERSE_CSV, index=False)

    print(f"\nSaved {close.shape[1]} tickers x {close.shape[0]} days "
          f"({close.index[0].date()} to {close.index[-1].date()})")
    print(uni.groupby("sector").size().to_string())


if __name__ == "__main__":
    main()
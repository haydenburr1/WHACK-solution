"""Rolling 30-day annualized volatility from data/prices.parquet (written by fetch_data.py).

Run (from anywhere):  python pipeline/volatility.py
Outputs (in the repo's data/ folder):
  volatility.csv / volatility.parquet  - annualized volatility, one column per ticker
  normalized_volatility.parquet        - same, scaled to [0, 1] on a shared 1st-99th percentile scale
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # repo-root data/

PRICES_FILE = DATA_DIR / "prices.parquet"
OUTPUT_FILE = DATA_DIR / "volatility.parquet"
CSV_FILE = DATA_DIR / "volatility.csv"
NORMALIZED_FILE = DATA_DIR / "normalized_volatility.parquet"

# Settings
WINDOW = 30
TRADING_DAYS = 252


def main():
    print("Loading price data...")

    # Load adjusted closing prices
    prices = pd.read_parquet(PRICES_FILE)

    print(f"Loaded {prices.shape[1]} tickers x {prices.shape[0]} days")

    # Calculate daily percentage returns
    returns = prices.pct_change(fill_method=None)

    # Calculate 30-day rolling volatility
    volatility = returns.rolling(
        window=WINDOW,
        min_periods=1
    ).std()

    # Annualize volatility
    annualized_volatility = volatility * np.sqrt(TRADING_DAYS)

    # Normalize across all stocks using a shared scale
    # Use the 1st and 99th percentiles to reduce outlier effects
    lower = annualized_volatility.stack().quantile(0.01)
    upper = annualized_volatility.stack().quantile(0.99)

    if pd.isna(lower) or pd.isna(upper) or upper <= lower:
        raise ValueError("Not enough valid volatility data to normalize.")

    normalized = (
        (annualized_volatility - lower) / (upper - lower)
    ).clip(0, 1)

    # Save results
    annualized_volatility.to_parquet(OUTPUT_FILE)
    annualized_volatility.to_csv(CSV_FILE)   # the visualisation scripts read the CSV
    normalized.to_parquet(NORMALIZED_FILE)

    print("\nVolatility calculation complete!")
    for f in (OUTPUT_FILE, CSV_FILE, NORMALIZED_FILE):
        print(f"Saved: {f}")

    print("\nSample normalized volatility:")
    print(normalized.tail())


if __name__ == "__main__":
    main()
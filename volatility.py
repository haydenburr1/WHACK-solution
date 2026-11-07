
from pathlib import Path

import numpy as np
import pandas as pd

# Project data directory
DATA_DIR = Path(__file__).resolve().parent / "data"

PRICES_FILE = DATA_DIR / "prices.parquet"
OUTPUT_FILE = DATA_DIR / "volatility.parquet"

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
    print(volatility)
    annualized_volatility.to_csv("data/volatility.csv")

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

    normalized.to_parquet(
        DATA_DIR / "normalized_volatility.parquet"
    )

    print("\nVolatility calculation complete!")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {DATA_DIR / 'normalized_volatility.parquet'}")

    print("\nSample normalized volatility:")
    print(normalized.tail())


if __name__ == "__main__":
    main()
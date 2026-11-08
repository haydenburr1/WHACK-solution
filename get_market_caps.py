#!/usr/bin/env python3
"""
Fetch market capitalization for every company in the Wikipedia S&P 500 list.

Install:
    pip install pandas requests lxml yfinance

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/get_market_caps.py --out caps.csv

Run:
    python sp500_market_caps.py
    python sp500_market_caps.py --out caps.csv --workers 16
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (sp500-market-cap-script; educational use)"}


def get_sp500_table() -> pd.DataFrame:
    """Scrape the constituents table from Wikipedia."""
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # The first table on the page is the current constituents list.
    df = pd.read_html(StringIO(resp.text))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["wiki_symbol", "company", "sector", "sub_industry"]
    # Yahoo Finance uses dashes for share classes (BRK.B -> BRK-B).
    df["ticker"] = df["wiki_symbol"].str.replace(".", "-", regex=False)
    return df


def get_market_cap(ticker: str):
    """Return (ticker, market_cap_in_usd or None)."""
    try:
        cap = yf.Ticker(ticker).fast_info["market_cap"]
        return ticker, float(cap) if cap else None
    except Exception:
        # Fall back to the slower .info endpoint
        try:
            cap = yf.Ticker(ticker).info.get("marketCap")
            return ticker, float(cap) if cap else None
        except Exception:
            return ticker, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="sp500_market_caps.csv", help="output CSV path")
    parser.add_argument("--workers", type=int, default=8, help="parallel requests")
    args = parser.parse_args()

    df = get_sp500_table()
    print(f"Found {len(df)} companies on Wikipedia. Fetching market caps...")

    caps = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(get_market_cap, t) for t in df["ticker"]]
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, cap = fut.result()
            caps[ticker] = cap
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done")

    df["market_cap_usd"] = df["ticker"].map(caps)
    df["market_cap_bn"] = (df["market_cap_usd"] / 1e9).round(2)
    df = df.sort_values("market_cap_usd", ascending=False, na_position="last")

    df.to_csv(args.out, index=False)
    missing = df["market_cap_usd"].isna().sum()
    print(f"\nSaved {len(df)} rows to {args.out} ({missing} missing).")
    print(df[["ticker", "company", "market_cap_bn"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
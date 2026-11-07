import pandas as pd
from pathlib import Path

file = Path("data") / "prices.parquet"
print(pd.read_parquet(file).head())

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/parquet_to_csv.py
"""
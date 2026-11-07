import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys

base = Path.cwd() / "data"

symb = "GC=F"

days_back = 30
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)

new = yf.download(symb, start=start_date, end=end_date, progress=False, interval="1d")
print('downloaded')
print(new)
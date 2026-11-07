"""Adat: a repo data/volatility.csv + data/universe.csv fájljai (vagy demo adat)."""
from pathlib import Path
import numpy as np
import pandas as pd
from themes import SECTOR_ORDER

LAYOUT_RANGE = 0.75      # a részvények (u,v) helye ezen belül van a lapon


def load_repo(data_dir="data"):
    """-> (vol: T x n DataFrame, uni: n soros DataFrame ticker/sector/sub_industry).
    Előbb a volatility.parquet-t keresi (ezt írja a volatility.py), ha nincs, a volatility.csv-t."""
    d = Path(data_dir)
    pq, csv = d / "volatility.parquet", d / "volatility.csv"
    if pq.exists():
        vol = pd.read_parquet(pq)
    elif csv.exists():
        vol = pd.read_csv(csv, index_col=0)
    else:
        raise FileNotFoundError(f"Nincs volatility.parquet vagy volatility.csv itt: {d.resolve()}. "
                                "Futtasd: python volatility.py (vagy próbáld: --demo)")
    vol.index = pd.to_datetime(vol.index)
    uni = pd.read_csv(d / "universe.csv")
    uni = uni[uni["ticker"].isin(vol.columns)].drop_duplicates("ticker").reset_index(drop=True)
    vol = vol[uni["ticker"]].dropna(how="all").bfill().ffill()
    return vol, uni


def demo_dataset(per_sector=12, T=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rows, cols = [], {}
    for i, s in enumerate(SECTOR_ORDER):
        factor = 0.18 + 0.08 * np.sin(2 * np.pi * (t / 120 + i / 6)) ** 2 \
                 + 0.35 * np.exp(-((t - (40 + 40 * i)) ** 2) / 60)
        for j in range(per_sector):
            tk = f"{s[:3].upper()}{j}"
            cols[tk] = np.clip(factor * (0.7 + 0.6 * rng.random()) + 0.03 * rng.standard_normal(T), 0.05, None)
            rows.append((tk, s, f"sub{j % 3}"))
    uni = pd.DataFrame(rows, columns=["ticker", "sector", "sub_industry"])
    return pd.DataFrame(cols, index=pd.RangeIndex(T, name="t")), uni


def build_layout(uni, tickers, seed=0):
    """Laponként: (oszlopindexek a vol táblában, (n,2) helyek a lapon).
    Részvények rácson, alágazat szerint rendezve, hogy a hasonlók egymás mellé kerüljenek."""
    pos_of = {t: i for i, t in enumerate(tickers)}
    rng = np.random.default_rng(seed)
    out = []
    for s in SECTOR_ORDER:
        rows = uni[uni["sector"] == s].sort_values(["sub_industry", "ticker"])
        idx = np.array([pos_of[t] for t in rows["ticker"]], dtype=int)
        n = len(idx)
        m = max(int(np.ceil(np.sqrt(n))), 1)
        g = np.linspace(-LAYOUT_RANGE, LAYOUT_RANGE, m) if m > 1 else np.array([0.0])
        pos = np.array([(g[i % m], g[i // m]) for i in range(n)]) + rng.normal(scale=0.03, size=(n, 2))
        out.append((idx, pos))
    return out

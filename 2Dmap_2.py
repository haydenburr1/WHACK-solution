import numpy as np

from build_terrain import build_terrain, show_terrain

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/2Dmap_2.py
"""


import time
import pandas as pd
from pathlib import Path
from MDS import mds_rows

vol = pd.read_csv(Path("data") / "volatility.csv", index_col=0).reset_index()
univ = pd.read_csv(Path("data") / "universe.csv", index_col=0).reset_index()
returns = pd.read_parquet(Path("data") / "prices.parquet").pct_change().reset_index()

sectors = sorted(univ["sector"].unique())
cell_size = 10
margin = 2.5        # peaks stay this far from the cell edge, so the slopes can reach floor inside the cell
beach = 1.5         # fade-to-water width at the cell edge
res = 200           # resolution per cell
hmax = 5.0          # height of the most volatile stock in the whole universe
n_cols, n_rows = 3, 2

corners = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# ---- pass 1: positions and volatilities for every sector ----
sector_data = []
for i, s in enumerate(sectors):
    tickers = univ.loc[univ["sector"] == s, "ticker"].values
    last_vola = vol[tickers].values[-1].astype(float)
    subset_returns = returns[tickers].values[1:]

    corr = np.corrcoef(subset_returns, rowvar=False)
    res_mds, _ = mds_rows(corr)                      # (n, 2) in [-1, 1]

    cx, cy = corners[i]
    usable = cell_size - 2 * margin
    xy = np.empty_like(res_mds, dtype=float)
    xy[:, 0] = cx + margin + (res_mds[:, 0] + 1) / 2 * usable
    xy[:, 1] = cy + margin + (res_mds[:, 1] + 1) / 2 * usable
    sector_data.append((cx, cy, xy, last_vola))

# one shared height scale, so sectors are comparable
vmax = max(d[3].max() for d in sector_data)

# ---- pass 2: build each island, place it in the global grid ----
GZ = np.zeros((n_cols * res, n_rows * res))         # ij indexing: [x, y]
gx = np.concatenate([np.linspace(c * cell_size, (c + 1) * cell_size, res) for c in range(n_cols)])
gy = np.concatenate([np.linspace(r * cell_size, (r + 1) * cell_size, res) for r in range(n_rows)])
GX, GY = np.meshgrid(gx, gy, indexing="ij")

for i, (cx, cy, xy, last_vola) in enumerate(sector_data):
    heights = hmax * last_vola / vmax
    heights = np.maximum(heights, 0.3)               # peaks must be above floor

    X, Y, Z = build_terrain(
        xy, heights,
        extent=(cx, cx + cell_size, cy, cy + cell_size),
        resolution=res,
        floor=0.0,
        seed=i,
    )

    # guarantee water at the cell border
    d_edge = np.minimum.reduce([X - cx, cx + cell_size - X, Y - cy, cy + cell_size - Y])
    Z = Z * smoothstep(d_edge / beach)

    ci, ri = int(cx // cell_size), int(cy // cell_size)
    GZ[ci * res:(ci + 1) * res, ri * res:(ri + 1) * res] = Z

# water is exactly GZ <= 0 (cells no peak reached); land is GZ > 0

GZ[GZ > 0] += 1
# PLOT

show_terrain(GX,GY,GZ)
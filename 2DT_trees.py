import numpy as np

from build_terrain import build_terrain, show_terrain_time

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/2DT.py
"""


import time
import pandas as pd
from pathlib import Path
import sys
from MDS import mds_rows

vol = pd.read_csv(Path("data") / "volatility.csv", index_col=0).reset_index()
univ = pd.read_csv(Path("data") / "universe.csv", index_col=0).reset_index()
returns = pd.read_parquet(Path("data") / "prices.parquet").pct_change().reset_index()

vol["date"] = pd.to_datetime(vol["date"])
returns["date"] = pd.to_datetime(returns["date"])

sectors = sorted(univ["sector"].unique())
cell_size = 10
margin = 2.5        # peaks stay this far from the cell edge, so the slopes can reach floor inside the cell
beach = 1.5         # fade-to-water width at the cell edge
res = 200           # resolution per cell
hmax = 5.0          # height of the most volatile stock in the whole universe
n_cols, n_rows = 3, 2

NORMALIZE_VOL = False

corners = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

window = 30
max_len_time_values = 100

time_values = vol["date"].iloc[window:window + max_len_time_values].values
sector_data = np.empty((len(sectors), len(time_values)), dtype=object)
for j, t in enumerate(time_values):
    start_time = t - pd.Timedelta(days=window)
    for i, s in enumerate(sectors):
        tickers = univ.loc[univ["sector"] == s, "ticker"].values

        if NORMALIZE_VOL:
            subset_vola = vol.loc[
                (vol["date"] >= start_time) &
                (vol["date"] <= t),
                tickers
            ].values.mean(axis=0)
            subset_vola = (subset_vola - np.mean(subset_vola)) / np.std(subset_vola)
        else:
            subset_vola = vol.loc[
                vol["date"] == t, tickers
            ].values.astype(float).flatten()

        subset_returns = returns.loc[
            (returns["date"] >= start_time) &
            (returns["date"] <= t),
            tickers
        ].values
        cum_ret = np.prod(1 + subset_returns, axis=0) - 1

        corr = np.corrcoef(subset_returns, rowvar=False)
        res_mds, _ = mds_rows(corr)                      # (n, 2) in [-1, 1]
        
        cx, cy = corners[i]
        usable = cell_size - 2 * margin
        xy = np.empty_like(res_mds, dtype=float)
        xy[:, 0] = cx + margin + (res_mds[:, 0] + 1) / 2 * usable
        xy[:, 1] = cy + margin + (res_mds[:, 1] + 1) / 2 * usable
        sector_data[i,j] = (cx, cy, xy, subset_vola, cum_ret)

GZ = np.zeros((n_cols * res, n_rows * res, len(time_values)))         # ij indexing: [x, y, t]
gx = np.concatenate([np.linspace(c * cell_size, (c + 1) * cell_size, res) for c in range(n_cols)])
gy = np.concatenate([np.linspace(r * cell_size, (r + 1) * cell_size, res) for r in range(n_rows)])
GX, GY = np.meshgrid(gx, gy, indexing="ij")

vmax = [
    max(sector_data[i,j][3].max() for i in range(sector_data.shape[0]))
    for j in range(sector_data.shape[1])
]

c = 0
for j, t in enumerate(time_values):
    for i, s in enumerate(sectors):
        cx, cy, xy, last_vola, _ = sector_data[i,j]
        vmax_this = vmax[j]
        heights = hmax * last_vola / vmax_this
        heights = np.maximum(heights, 0.3)               # peaks must be above floor

        X, Y, Z = build_terrain(
            xy, heights,
            extent=(cx, cx + cell_size, cy, cy + cell_size),
            resolution=res,
            floor=0.0,
            seed=i,
        )
        print(i, j)

        # guarantee water at the cell border
        d_edge = np.minimum.reduce([X - cx, cx + cell_size - X, Y - cy, cy + cell_size - Y])
        Z = Z * smoothstep(d_edge / beach)

        ci, ri = int(cx // cell_size), int(cy // cell_size)
        GZ[ci * res:(ci + 1) * res, ri * res:(ri + 1) * res, j] = Z

GZ[GZ > 0] += 0.5

np.save("struct_Z.npy", GZ)
np.save("struct_X.npy", GX)
np.save("struct_Y.npy", GY)

show_terrain_time(
    GX,
    GY,
    GZ,
    time_values
)

rng = np.random.default_rng(0)

max_trees, spread = 30, 0.7
# fixed offsets per (sector, stock)
offsets = {(i, k): rng.normal(0, spread, (max_trees, 2))
           for i in range(len(sectors))
           for k in range(len(sector_data[i, 0][2]))}

def surface_z(GZ, gx, gy, j, x, y):
    ix = np.clip(np.searchsorted(gx, x), 0, len(gx) - 1)
    iy = np.clip(np.searchsorted(gy, y), 0, len(gy) - 1)
    return GZ[ix, iy, j]

from scipy.stats import rankdata

def veg_scores(sector_data, j):
    all_ret = np.concatenate([sector_data[i, j][4] for i in range(sector_data.shape[0])])
    pct = (rankdata(all_ret) - 1) / (len(all_ret) - 1)       # percentile rank
    score = np.where(all_ret > 0, pct, 0.0)                  # no trees on losers
    # split back per sector
    out, k = [], 0
    for i in range(sector_data.shape[0]):
        n = len(sector_data[i, j][4]); out.append(score[k:k+n]); k += n
    return out

trees = []   # trees[j] -> array of (x, y, z, size)
for j in range(len(time_values)):
    scores = veg_scores(sector_data, j)
    frame = []
    for i in range(len(sectors)):
        xy = sector_data[i, j][2]
        for k, (px, py) in enumerate(xy):
            s = scores[i][k]
            n = int(round(max_trees * s))
            if n == 0:
                continue
            pts = np.array([px, py]) + offsets[(i, k)][:n]
            z = surface_z(GZ, gx, gy, j, pts[:, 0], pts[:, 1])
            ok = z > 0.6                                  # skip water/beach (after the +0.5 shift)
            size = 0.2 + 0.6 * s                          # better performers = bigger trees
            frame.extend((x, y, zz, size) for (x, y), zz in zip(pts[ok], z[ok]))
    trees.append(np.array(frame))
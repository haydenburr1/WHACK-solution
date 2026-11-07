import numpy as np

from build_terrain import build_terrain, show_terrain_time

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/2Dmap_2t.py
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

window = 30
corners = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]

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

vmax = max(d[3].max() for d in sector_data)

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------
# GLOBAL GRID
# --------------------------------------------------

gx = np.concatenate([
    np.linspace(c * cell_size, (c + 1) * cell_size, res)
    for c in range(n_cols)
])

gy = np.concatenate([
    np.linspace(r * cell_size, (r + 1) * cell_size, res)
    for r in range(n_rows)
])

GX, GY = np.meshgrid(gx, gy, indexing="ij")


# --------------------------------------------------
# TIME SERIES
# --------------------------------------------------

# Use the same number of observations for both datasets.
n_times = min(len(vol), len(returns))

dates = vol.iloc[:n_times, 0].values

all_tickers = univ["ticker"].unique()

# One common height scale for all dates and sectors.
vmax = np.nanmax(
    vol.loc[:n_times - 1, all_tickers].to_numpy(dtype=float)
)




def make_frame(t):

    GZ = np.zeros((n_cols * res, n_rows * res))

    # Ensure t is a valid endpoint
    t = int(t)

    # Rolling window: [t - window : t]
    start = t - window

    # ----------------------------------------------
    # Build every sector
    # ----------------------------------------------

    for i, s in enumerate(sectors):

        tickers = univ.loc[
            univ["sector"] == s, "ticker"
        ].values

        # Volatility at current time t
        last_vola = vol.loc[t, tickers].values.astype(float)

        # Returns over the rolling window
        subset_returns = returns.loc[
            start:t, tickers
        ]

        # Calculate correlation matrix
        corr = np.corrcoef(
            subset_returns.to_numpy(),
            rowvar=False
        )

        np.fill_diagonal(corr, 1.0)

        # MDS positions
        res_mds, _ = mds_rows(corr)

        # Sector location
        cx, cy = corners[i]

        usable = cell_size - 2 * margin

        xy = np.empty_like(res_mds, dtype=float)

        xy[:, 0] = (
            cx + margin
            + (res_mds[:, 0] + 1) / 2 * usable
        )

        xy[:, 1] = (
            cy + margin
            + (res_mds[:, 1] + 1) / 2 * usable
        )

        # ------------------------------------------
        # Generate terrain heights
        # ------------------------------------------

        heights = hmax * last_vola / vmax
        heights = np.maximum(heights, 0.3)

        X, Y, Z = build_terrain(
            xy,
            heights,
            extent=(
                cx,
                cx + cell_size,
                cy,
                cy + cell_size
            ),
            resolution=res,
            floor=0.0,
            seed=i,
        )

        # Guarantee water at cell borders
        d_edge = np.minimum.reduce([
            X - cx,
            cx + cell_size - X,
            Y - cy,
            cy + cell_size - Y
        ])

        Z = Z * smoothstep(d_edge / beach)

        # ------------------------------------------
        # Insert into global terrain
        # ------------------------------------------

        ci = int(cx // cell_size)
        ri = int(cy // cell_size)

        GZ[
            ci * res:(ci + 1) * res,
            ri * res:(ri + 1) * res
        ] = Z

    return GZ



show_terrain_time(
    GX,
    GY,
    make_frame,
    dates,
    hmax
)
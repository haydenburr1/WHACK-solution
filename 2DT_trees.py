import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
import pyvista as pv

"""
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/2DT_trees.py
"""



from build_terrain import show_terrain_time_trees
from MDS import mds_rows

# ---------------- load precomputed terrain ----------------
GZ = np.load("struct_Z.npy")   # already includes the +0.5 land shift
GX = np.load("struct_X.npy")
GY = np.load("struct_Y.npy")

# 1D axes recovered from the meshgrid (built with indexing="ij")
gx = GX[:, 0]
gy = GY[0, :]

# ---------------- data ----------------
vol = pd.read_csv(Path("data") / "volatility.csv", index_col=0).reset_index()
univ = pd.read_csv(Path("data") / "universe.csv", index_col=0).reset_index()
returns = pd.read_parquet(Path("data") / "prices.parquet").pct_change().reset_index()



vol["date"] = pd.to_datetime(vol["date"])
returns["date"] = pd.to_datetime(returns["date"])

sectors = sorted(univ["sector"].unique())
cell_size = 10
margin = 2.5
corners = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]

# must be identical to the values used when the terrain was generated
window = 30
max_len_time_values = 100

time_values = vol["date"].iloc[window:window + max_len_time_values].values
assert GZ.shape[2] == len(time_values), "saved terrain has a different number of time steps"

# ---------------- per-stock positions and returns ----------------
sector_data = np.empty((len(sectors), len(time_values)), dtype=object)

for j, t in enumerate(time_values):
    start_time = t - pd.Timedelta(days=window)
    for i, s in enumerate(sectors):
        tickers = univ.loc[univ["sector"] == s, "ticker"].values

        subset_returns = returns.loc[
            (returns["date"] >= start_time) &
            (returns["date"] <= t),
            tickers
        ].values
        mean_returns = np.mean(subset_returns, axis=0)

        corr = np.corrcoef(subset_returns, rowvar=False)
        res_mds, _ = mds_rows(corr)                      # (n, 2) in [-1, 1]

        cx, cy = corners[i]
        usable = cell_size - 2 * margin
        xy = np.empty_like(res_mds, dtype=float)
        xy[:, 0] = cx + margin + (res_mds[:, 0] + 1) / 2 * usable
        xy[:, 1] = cy + margin + (res_mds[:, 1] + 1) / 2 * usable
        sector_data[i, j] = (cx, cy, xy, mean_returns)

# ---------------- trees ----------------
rng = np.random.default_rng(0)

max_side = 8                 # max small trees per stock
main_min, main_max = 0.6, 3.0    # main tree height range (min for weakest, max for best)
side_frac = 0.2              # side tree height as a fraction of the main tree height
MAIN_FOR_LOSERS = True       # False -> stocks with negative returns get no trees at all

# fixed per-stock random (angle, radial fraction) so side trees don't jump between frames
offsets = {(i, k): np.column_stack([rng.uniform(0, 2 * np.pi, max_side),
                                    rng.uniform(0, 1, max_side)])
           for i in range(len(sectors))
           for k in range(len(sector_data[i, 0][2]))}

def surface_z(GZ, gx, gy, j, x, y):
    ix = np.clip(np.searchsorted(gx, x), 0, len(gx) - 1)
    iy = np.clip(np.searchsorted(gy, y), 0, len(gy) - 1)
    return GZ[ix, iy, j]

def veg_scores(sector_data, j):
    all_ret = np.concatenate([sector_data[i, j][3] for i in range(sector_data.shape[0])])
    pct = (rankdata(all_ret) - 1) / (len(all_ret) - 1)
    score = np.where(all_ret > 0, pct, 0.0)
    out, k = [], 0
    for i in range(sector_data.shape[0]):
        n = len(sector_data[i, j][3])
        out.append(score[k:k + n])
        k += n
    return out

trees = []   # trees[j] -> (n, 4) array of x, y, z, size
for j in range(len(time_values)):
    scores = veg_scores(sector_data, j)
    frame = []
    for i in range(len(sectors)):
        xy = sector_data[i, j][2]
        for k, (px, py) in enumerate(xy):
            s = scores[i][k]
            if s == 0 and not MAIN_FOR_LOSERS:
                continue

            # ---- main tree, on the peak ----
            main_size = main_min + (main_max - main_min) * s
            z_peak = surface_z(GZ, gx, gy, j, np.array([px]), np.array([py]))[0]
            if z_peak <= 0.6:
                continue                      # peak is not on land
            frame.append((px, py, z_peak, main_size))

            # ---- small trees around it ----
            n = int(round(max_side * s))
            if n == 0:
                continue
            ang, u = offsets[(i, k)][:n, 0], offsets[(i, k)][:n, 1]
            # start outside the main canopy (radius ~0.22 * height), then spread outwards
            dist = 0.22 * main_size + 0.15 + u * 0.6
            sx = px + dist * np.cos(ang)
            sy = py + dist * np.sin(ang)
            z = surface_z(GZ, gx, gy, j, sx, sy)
            ok = z > 0.6
            side_size = side_frac * main_size
            frame.extend((x, y, zz, side_size) for x, y, zz in zip(sx[ok], sy[ok], z[ok]))
    trees.append(np.array(frame).reshape(-1, 4))

show_terrain_time_trees(
    GX, GY, GZ,
    time_values,
    trees=trees,
    background_image="background_2.jpeg",
    cloud_values=
)
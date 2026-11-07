"""
Six sectors on the six faces of a cube.

Every sector is built exactly as in the flat map (MDS positions -> peaks with
volatility heights -> terrain with a beach fade), but each sector's height map
is wrapped onto one face of a cube: the face plane is sea level, and the
terrain rises out of the face along its outward normal. Because the beach fade
forces the height to 0 at every cell border, neighbouring faces meet at the
cube edges at water level, with no gaps.

Use the slider (release it to update) to move through time.

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/2DT_cube.py
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

from build_terrain import build_terrain
from MDS import mds_rows

# --------------------------------------------------
# DATA
# --------------------------------------------------

vol = pd.read_csv(Path("data") / "volatility.csv", index_col=0).reset_index()
univ = pd.read_csv(Path("data") / "universe.csv", index_col=0).reset_index()
returns = pd.read_parquet(Path("data") / "prices.parquet").pct_change().reset_index()

sectors = sorted(univ["sector"].unique())        # one sector per cube face
assert len(sectors) == 6, f"expected 6 sectors, found {len(sectors)}"
sector_tickers = [univ.loc[univ["sector"] == s, "ticker"].values for s in sectors]

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

cell_size = 10      # side length of the cube = side length of every face
margin = 2.5        # peaks stay this far from the face edge, so slopes can reach the floor inside it
beach = 1.5         # fade-to-water width at the face edge
res = 200           # resolution per face
hmax = 5.0          # height of the most volatile stock in the whole universe
window = 30         # rolling window (days) for the correlations
frame_step = 5      # days between slider positions
height_scale = 1.0  # terrain height as it appears on the cube (1.0 = same as in the flat map)

# --------------------------------------------------
# TIME SERIES
# --------------------------------------------------

# Use the same number of observations for both datasets.
n_times = min(len(vol), len(returns))
dates = vol.iloc[:n_times, 0].values

all_tickers = univ["ticker"].unique()

# One common height scale for all dates and sectors.
vmax = np.nanmax(vol.loc[:n_times - 1, all_tickers].to_numpy(dtype=float))

# Row 0 of pct_change() is NaN, so the first usable window starts at row 1.
frame_times = list(range(window + 1, n_times, frame_step))


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------
# ONE SECTOR AT ONE TIME -> (res, res) HEIGHT MAP
# --------------------------------------------------

def sector_height(i, t):
    tickers = sector_tickers[i]

    # Volatility at time t
    last_vola = np.nan_to_num(vol.loc[t, tickers].values.astype(float))

    # Correlation over the rolling window [t - window, t]
    window_returns = returns.loc[t - window:t, tickers].to_numpy()
    corr = np.nan_to_num(np.corrcoef(window_returns, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # MDS positions, kept inside [-1, 1] so the peaks stay inside the face
    pos, _ = mds_rows(corr)
    pos = pos / max(1.0, np.abs(pos).max())

    usable = cell_size - 2 * margin
    xy = margin + (pos + 1) / 2 * usable            # local face coordinates

    heights = np.maximum(hmax * last_vola / vmax, 0.3)

    X, Y, Z = build_terrain(
        xy,
        heights,
        extent=(0, cell_size, 0, cell_size),
        resolution=res,
        floor=0.0,
        seed=i,
    )

    # Guarantee water at the face borders
    d_edge = np.minimum.reduce([X, cell_size - X, Y, cell_size - Y])
    return Z * smoothstep(d_edge / beach)


@functools.lru_cache(maxsize=32)
def make_frame(t):
    """Height maps of all six sectors at time t (cached, so going back is instant)."""
    return tuple(sector_height(i, t) for i in range(6))


# --------------------------------------------------
# CUBE GEOMETRY
# --------------------------------------------------

# One entry per face: (outward normal, e1, e2) with e1 x e2 = normal.
# Face i shows sectors[i].
FACES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),     # +x
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),    # -x
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),     # +y
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),    # -y
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),     # +z
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),    # -z
]

half = cell_size / 2
u = np.linspace(-half, half, res)
U, V = np.meshgrid(u, u, indexing="ij")          # same layout as the height maps


def face_mesh(face, Z):
    """Wrap a (res, res) height map onto a cube face; the face plane is sea level."""
    n, e1, e2 = (np.array(v, dtype=float) for v in face)
    P = (half + height_scale * Z)[..., None] * n \
        + U[..., None] * e1 + V[..., None] * e2   # (res, res, 3)
    mesh = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])
    mesh["height"] = Z.ravel(order="F")           # StructuredGrid uses Fortran order
    return mesh


# Water at 0, then beach, grass, hills, rock and snow (scale is 0..hmax)
LAND = LinearSegmentedColormap.from_list("water_land", [
    (0.00, "#1d4e89"), (0.03, "#2f80c0"),
    (0.04, "#e8d9a0"), (0.12, "#c2d48a"),
    (0.35, "#4f8a3f"), (0.65, "#8a7a5a"), (1.00, "#ffffff"),
])

# --------------------------------------------------
# VIEWER
# --------------------------------------------------

plotter = pv.Plotter()


def draw(k):
    t = frame_times[k]
    for i, (face, Z) in enumerate(zip(FACES, make_frame(t))):
        plotter.add_mesh(
            face_mesh(face, Z), scalars="height", cmap=LAND, clim=(0, hmax),
            smooth_shading=True, show_scalar_bar=False,
            name=f"face_{i}", reset_camera=False,     # same name -> replaces the old mesh
        )
    plotter.add_text(str(dates[t]), name="date", position="upper_left", font_size=14)


def on_slide(value):
    draw(int(round(value)))


draw(len(frame_times) - 1)                        # start at the latest date

label_pts = np.array([
    np.array(f[0], dtype=float) * (half + height_scale * hmax + 0.8) for f in FACES
])
plotter.add_point_labels(label_pts, sectors, font_size=18,
                         show_points=False, shape=None, always_visible=True)

plotter.add_slider_widget(
    on_slide, rng=[0, len(frame_times) - 1], value=len(frame_times) - 1,
    title="time", fmt="%.0f",
)

plotter.camera_position = "iso"
plotter.show()
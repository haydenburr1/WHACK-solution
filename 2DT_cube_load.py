"""
Cube from the saved terrain arrays: one sector per cube face.

Loads
    struct_X.npy  (n_cols*res, n_rows*res)      global x coordinates
    struct_Y.npy  (n_cols*res, n_rows*res)      global y coordinates
    struct_Z.npy  (n_cols*res, n_rows*res, T)   heights, [x, y, time]
which are the arrays written by the flat-map script (GX, GY, GZ). Sector i sits in
grid cell (column i % n_cols, row i // n_cols), exactly as in that script, so
sector i is cut out of the array and wrapped onto cube face i.

Optional extra files (nothing breaks if they are missing):
    struct_time.npy       np.save("struct_time.npy", time_values)  -> dates in the label
    data/universe.csv     sector names for the face labels

Each face is oriented so the map looks un-mirrored from outside the cube; on the
four side faces the map's "up" (+y) points up (+z). The face plane is sea level
and land rises out of the cube. Nothing is recomputed, so the slider is fast.

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/cube_from_struct.py
"""

from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

n_cols, n_rows = 3, 2       # same grid as the flat map
height_scale = 1.0          # how far the terrain sticks out of the cube (1.0 = as in the flat map)
water_below = 0.25          # heights below this are coloured as water

# --------------------------------------------------
# LOAD
# --------------------------------------------------

GX = np.load("struct_X.npy")
GY = np.load("struct_Y.npy")
GZ = np.load("struct_Z.npy", mmap_mode="r")          # big: read frames on demand

assert GZ.ndim == 3, f"struct_Z should be (x, y, time), got shape {GZ.shape}"
assert GX.shape == GY.shape == GZ.shape[:2], \
    f"shape mismatch: X {GX.shape}, Y {GY.shape}, Z {GZ.shape}"

res = GZ.shape[0] // n_cols
assert GZ.shape[:2] == (n_cols * res, n_rows * res), \
    f"{GZ.shape[:2]} is not {n_cols} x {n_rows} cells of equal size"
n_sectors = n_cols * n_rows
n_times = GZ.shape[2]
zmax = float(np.max(GZ))                             # one colour scale for every frame

try:
    time_values = np.load("struct_time.npy")
    def label(j):
        return str(np.datetime_as_string(time_values[j], unit="D"))
except FileNotFoundError:
    def label(j):
        return f"frame {j}"

try:
    import pandas as pd
    names = sorted(pd.read_csv(Path("data") / "universe.csv", index_col=0)["sector"].unique())
    assert len(names) == n_sectors
except Exception:
    names = [f"sector {i}" for i in range(n_sectors)]

# --------------------------------------------------
# CUBE GEOMETRY
# --------------------------------------------------

# Face of sector i: (outward normal, e1 = direction of the map's x axis,
# e2 = direction of the map's y axis), with e1 x e2 = normal.
FACES = [
    ((1, 0, 0),  (0, 1, 0),  (0, 0, 1)),    # sector 0 -> +x   (map up = +z)
    ((0, 1, 0),  (-1, 0, 0), (0, 0, 1)),    # sector 1 -> +y
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),    # sector 2 -> -x
    ((0, -1, 0), (1, 0, 0),  (0, 0, 1)),    # sector 3 -> -y
    ((0, 0, 1),  (1, 0, 0),  (0, 1, 0)),    # sector 4 -> +z   (top)
    ((0, 0, -1), (1, 0, 0),  (0, -1, 0)),   # sector 5 -> -z   (bottom)
]

faces = []
for i, (n, e1, e2) in enumerate(FACES):
    n, e1, e2 = (np.array(v, dtype=float) for v in (n, e1, e2))
    assert np.allclose(np.cross(e1, e2), n), f"face {i} is not right-handed"

    ci, ri = i % n_cols, i // n_cols                  # grid cell of sector i
    cell = (slice(ci * res, (ci + 1) * res), slice(ri * res, (ri + 1) * res))
    Xc, Yc = GX[cell], GY[cell]
    # local map coordinates, centred on the cell
    U = Xc - 0.5 * (Xc.min() + Xc.max())
    V = Yc - 0.5 * (Yc.min() + Yc.max())
    in_plane = U[..., None] * e1 + V[..., None] * e2   # (res, res, 3)
    faces.append((cell, n, in_plane))

half = 0.5 * (GX[:res, :res].max() - GX[:res, :res].min())     # half the cell size


def face_mesh(i, j):
    """Mesh of face i at time index j; the face plane is sea level."""
    cell, n, in_plane = faces[i]
    Z = np.asarray(GZ[cell[0], cell[1], j])
    P = in_plane + (half + height_scale * Z)[..., None] * n
    mesh = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])
    mesh["height"] = Z.ravel(order="F")               # StructuredGrid uses Fortran order
    return mesh


# Water below `water_below`, then beach, grass, hills, rock and snow
w = min(water_below / zmax, 0.5)
rest = 1.0 - w
LAND = LinearSegmentedColormap.from_list("water_land", [
    (0.0, "#1d4e89"), (w, "#2f80c0"),
    (w + 1e-6, "#e8d9a0"), (w + 0.10 * rest, "#c2d48a"),
    (w + 0.35 * rest, "#4f8a3f"), (w + 0.65 * rest, "#8a7a5a"), (1.0, "#ffffff"),
])

# --------------------------------------------------
# VIEWER
# --------------------------------------------------

plotter = pv.Plotter()


def draw(j):
    for i in range(n_sectors):
        plotter.add_mesh(
            face_mesh(i, j), scalars="height", cmap=LAND, clim=(0, zmax),
            smooth_shading=True, show_scalar_bar=False,
            name=f"face_{i}", reset_camera=False,     # same name -> replaces the old mesh
        )
    plotter.add_text(label(j), name="date", position="upper_left", font_size=14)


def on_slide(value):
    draw(int(round(value)))


draw(0)

label_pts = np.array([
    np.array(f[0], dtype=float) * (half + height_scale * zmax + 0.8) for f in FACES
])
plotter.add_point_labels(label_pts, names, font_size=18,
                         show_points=False, shape=None, always_visible=True)

plotter.add_slider_widget(
    on_slide, rng=[0, n_times - 1], value=0, title="time", fmt="%.0f",
)

plotter.camera_position = "iso"
plotter.show()
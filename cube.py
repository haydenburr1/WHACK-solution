"""
Six sectors on the six faces of a cube, each face textured exactly like 2DT.py.

Every face is built with the same pipeline as the flat map in 2DT.py:
MDS positions of the sector's stocks (from rolling return correlations) ->
peaks whose heights are the stocks' volatility -> build_terrain -> beach fade
to water at the face border -> land lifted by 0.5 -> "terrain" colormap with
the same fixed colour range. The height map is then wrapped onto a cube face:
the face plane is sea level and the terrain rises along the outward normal.
Because the beach fade forces the height to 0 at every face border, faces meet
at the cube edges at water level, with no gaps.

Speed:
  * struct_Z.npy (written by 2DT.py) already holds the finished terrain for
    its days, so those days are read straight from the file: no terrain is grown.
  * Every other day is grown in background worker processes, nearest to the
    slider first; until a day is ready the slider shows the nearest finished one.
  * The six face meshes are created once and only their points change per day.

Trees and colours follow 2DT_trees.py: every stock gets a main tree on its peak,
sized by how its mean return over the window ranks against all stocks, and
winners get up to 8 small trees around it. Colours use the same green-mountain
colormap (hard water edge at sea level).

Controls: slider (updates while dragging), Left / Right arrow keys, T toggles
trees, hover a peak for stock info.

for tobias:
./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/cube.py

"""

import functools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

USE_BACKGROUND = True

# pyvista / vtk / scipy.spatial are imported inside the drawing code, not here:
# on Windows every worker process re-runs this file's top level, and the
# workers only grow terrain, so skipping those imports makes them start faster.
from build_terrain import build_terrain
from MDS import mds_rows

# --------------------------------------------------
# DATA (loaded exactly as in 2DT.py)
# --------------------------------------------------

vol = pd.read_csv(Path("data") / "volatility.csv", index_col=0).reset_index()
univ = pd.read_csv(Path("data") / "universe.csv", index_col=0).reset_index()
returns = pd.read_parquet(Path("data") / "prices.parquet").pct_change().reset_index()

vol["date"] = pd.to_datetime(vol["date"])
returns["date"] = pd.to_datetime(returns["date"])

sectors = sorted(univ["sector"].unique())   # same order as 2DT.py -> same seeds
sector_tickers = [univ.loc[univ["sector"] == s, "ticker"].values for s in sectors]

# --------------------------------------------------
# TERRAIN PARAMETERS (identical to 2DT.py)
# --------------------------------------------------

cell_size = 10      # 2DT cell size = cube side length
margin = 2.5        # peaks stay this far from the face edge
beach = 1.5         # fade-to-water width at the face edge
res = 200           # resolution per face
hmax = 5.0          # height of the most volatile stock on each date
land_lift = 0.5     # 2DT.py does GZ[GZ > 0] += 0.5
window = 30         # rolling window (days) for the correlations
n_cols = 3          # 2DT.py's grid: sector i sits in column i % 3, row i // 3

CLIM = (0.0, hmax + land_lift)   # = np.max(GZ) in 2DT.py: tallest peak is hmax, lifted


def green_mountain_cmap(zmax=CLIM[1], sea=land_lift):
    """The colormap from build_terrain.show_terrain_time_trees (used by 2DT_trees.py)."""
    from matplotlib.colors import LinearSegmentedColormap

    s = sea / zmax                  # sea level as a fraction of the full range

    def land(f):
        """Map f in [0, 1] (0 = shoreline, 1 = highest peak) to a colormap position."""
        return s + f * (1 - s)

    stops = [
        (0.0,        "#1a5ab3"),    # deep water
        (s,          "#1a5ab3"),    # water up to sea level (hard edge)
        (land(0.00), "#8ccc66"),    # lowland: light green
        (land(0.30), "#2f8a3a"),    # hills: darker green
        (land(0.50), "#d9a441"),    # foothills: yellow-orange
        (land(0.65), "#c8662b"),    # mountains: orange
        (land(0.80), "#7d7d7d"),    # rock: gray
        (1.0,        "#e6e6e6"),    # peaks: light gray
    ]
    return LinearSegmentedColormap.from_list("green_mountain", stops, N=512)

HALF = cell_size / 2
HEIGHT_SCALE = 1.0  # 1.0 = terrain stands exactly as tall as in the flat map

# Precomputed terrain from 2DT.py: frame j is the day in row NPY_FIRST_ROW + j.
NPY_Z = Path("struct_Z.npy")
NPY_FIRST_ROW = window   # 2DT.py: time_values = vol["date"].iloc[window:window + 100]

# Which face each sector sits on: (axis of the face normal, +1 / -1 side).
# axis 0 = x, 1 = y, 2 = z. The other two axes are the face's (a, b) directions.
FACE_AXES = {
    "Consumer Discretionary": (0, 1),
    "Energy": (0, -1),
    "Financials": (1, 1),
    "Health Care": (1, -1),
    "Industrials": (2, 1),
    "Information Technology": (2, -1),
}
assert set(FACE_AXES) == set(sectors), f"sectors don't match FACE_AXES: {sectors}"


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------
# ONE SECTOR AT ONE DATE (the 2DT.py pipeline)
# --------------------------------------------------

@functools.lru_cache(maxsize=None)
def vmax_for(t):
    # 2DT.py normalises by the most volatile stock across all sectors on that date.
    all_tickers = np.concatenate(sector_tickers)
    return float(np.nanmax(vol.loc[vol["date"] == t, all_tickers].values.astype(float)))


def sector_layout(i, t):
    """Peak positions (local face coords), heights and mean returns for sector i on date t. Cheap."""
    tickers = sector_tickers[i]
    start_time = t - pd.Timedelta(days=window)

    last_vola = np.nan_to_num(vol.loc[vol["date"] == t, tickers].values.astype(float).flatten())

    window_returns = returns.loc[
        (returns["date"] >= start_time) & (returns["date"] <= t), tickers
    ].values
    window_returns = window_returns[np.isfinite(window_returns).all(axis=1)]
    mean_returns = window_returns.mean(axis=0)        # drives the trees (2DT_trees.py)
    corr = np.nan_to_num(np.corrcoef(window_returns, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    pos, _ = mds_rows(corr)                           # same placement as 2DT.py
    usable = cell_size - 2 * margin
    xy = margin + (pos + 1) / 2 * usable

    heights = np.maximum(hmax * last_vola / vmax_for(t), 0.3)
    return xy, heights, mean_returns


def sector_terrain(i, t):
    """Height map (res, res) for sector i on date t, built as in 2DT.py. Slow-ish."""
    xy, heights, mean_returns = sector_layout(i, t)
    X, Y, Z = build_terrain(
        xy, heights,
        extent=(0, cell_size, 0, cell_size),
        resolution=res,
        floor=0.0,
        seed=i,
    )
    d_edge = np.minimum.reduce([X, cell_size - X, Y, cell_size - Y])
    Z = Z * smoothstep(d_edge / beach)
    Z[Z > 0] += land_lift
    return Z, xy, heights, mean_returns


# --------------------------------------------------
# PRECOMPUTED TERRAIN (struct_Z.npy from 2DT.py)
# --------------------------------------------------

class SavedTerrain:
    """Reads finished tiles straight out of struct_Z.npy; nothing is grown."""

    def __init__(self, path=NPY_Z):
        self.Z = None
        self.first = self.last = None
        if not path.exists():
            print(f"{path} not found: every day will be grown (run 2DT.py to create it).")
            return
        Z = np.load(path, mmap_mode="r")             # 190 MB: read frames on demand
        if Z.ndim != 3 or Z.shape[:2] != (n_cols * res, (len(sectors) // n_cols) * res):
            print(f"{path} has shape {Z.shape}, expected ({n_cols * res}, "
                  f"{len(sectors) // n_cols * res}, T); ignoring it.")
            return
        self.Z = Z
        self.first = NPY_FIRST_ROW
        self.last = min(NPY_FIRST_ROW + Z.shape[2], len(vol)) - 1
        print(f"{path}: days {self.first}-{self.last} "
              f"({vol['date'].iloc[self.first].date()} to {vol['date'].iloc[self.last].date()}) "
              "load instantly.")

    def has(self, day):
        return self.Z is not None and self.first <= day <= self.last

    def nearest(self, day):
        return None if self.Z is None else int(np.clip(day, self.first, self.last))

    @functools.lru_cache(maxsize=64)
    def tiles(self, day):
        t = vol["date"].iloc[day]
        j = day - self.first
        out = []
        for i in range(len(sectors)):
            ci, ri = i % n_cols, i // n_cols          # same grid cell as in 2DT.py
            Z = np.asarray(self.Z[ci * res:(ci + 1) * res, ri * res:(ri + 1) * res, j],
                           dtype=np.float32)
            xy, heights, mean_returns = sector_layout(i, t)   # for hover and trees
            out.append((Z, xy, heights, mean_returns))
        return out


class Frame(list):
    """One day's six tiles, optionally carrying precomputed trees (`trees`)."""
    trees = None


class CachedTerrain:
    """Reads everything for a day out of cube_cache/ (written by build_cache.py).

    No terrain growing, MDS, pandas filtering or tree placement happens while
    sliding: a day is a handful of array slices.
    """

    def __init__(self, path=Path("cube_cache")):
        self.Z = None
        self.first = self.last = None
        try:
            done = np.load(path / "done.npy")
            Z = np.load(path / "Z.npy", mmap_mode="r")
            xy = np.load(path / "xy.npy", mmap_mode="r")
            heights = np.load(path / "heights.npy", mmap_mode="r")
            trees = np.load(path / "trees.npy", mmap_mode="r")
            tree_sec = np.load(path / "tree_sec.npy", mmap_mode="r")
        except (FileNotFoundError, ValueError):
            print(f"{path}/ not found: run `python build_cache.py` for smooth sliding.")
            return
        if Z.shape[1:] != (len(sectors), res, res) or len(done) != LAST_DAY - FIRST_DAY + 1:
            print(f"{path}/ doesn't match the current data/settings; re-run build_cache.py.")
            return
        self.Z, self.xy, self.heights = Z, xy, heights
        self.trees, self.tree_sec, self.done = trees, tree_sec, done
        self.days = np.flatnonzero(done) + FIRST_DAY
        self.first, self.last = FIRST_DAY, LAST_DAY
        self._splits = np.cumsum([len(t) for t in sector_tickers])[:-1]
        print(f"{path}/: {len(self.days)}/{len(done)} days precompiled.")
        if not len(self.days):
            self.Z = None

    def has(self, day):
        return self.Z is not None and FIRST_DAY <= day <= LAST_DAY and bool(self.done[day - FIRST_DAY])

    def nearest(self, day):
        if self.Z is None:
            return None
        return int(self.days[np.argmin(np.abs(self.days - day))])

    @functools.lru_cache(maxsize=256)
    def tiles(self, day):
        j = day - FIRST_DAY
        Z = np.asarray(self.Z[j], dtype=np.float32)
        xy = np.split(np.asarray(self.xy[j]), self._splits)
        heights = np.split(np.asarray(self.heights[j]), self._splits)
        out = Frame((Z[i], xy[i], heights[i], None) for i in range(len(sectors)))
        trees, sec = np.asarray(self.trees[j]), np.asarray(self.tree_sec[j])
        out.trees = [trees[sec == i] if (sec == i).any() else None for i in range(len(sectors))]
        return out


class Sources:
    """Cache first, then struct_Z.npy: whichever has the day."""

    def __init__(self, *sources):
        self.sources = [s for s in sources if s.Z is not None]
        self.Z = self.sources[0].Z if self.sources else None
        self.first = min((s.first for s in self.sources), default=None)
        self.last = max((s.last for s in self.sources), default=None)

    def has(self, day):
        return any(s.has(day) for s in self.sources)

    def tiles(self, day):
        return next(s for s in self.sources if s.has(day)).tiles(day)

    def nearest(self, day):
        cands = [n for n in (s.nearest(day) for s in self.sources) if n is not None]
        return min(cands, key=lambda d: abs(d - day)) if cands else None


# --------------------------------------------------
# CUBE GEOMETRY
# --------------------------------------------------

def face_to_xyz(sector, a, b, h):
    # Map face coordinates (a, b) plus a height h to 3D, pushing outward along the face normal.
    axis, sign = FACE_AXES[sector]
    first, second = [k for k in range(3) if k != axis]

    coords = [None, None, None]
    coords[axis] = sign * (HALF + h * HEIGHT_SCALE)
    coords[first] = a
    coords[second] = b
    return coords


u = np.linspace(-HALF, HALF, res)
A, B = np.meshgrid(u, u, indexing="ij")    # same layout as build_terrain's output

# Triangles of one res x res face, built once. Every day reuses them and only
# moves the points, so swapping days never rebuilds or re-merges meshes.
_idx = np.arange(res * res).reshape(res, res)
_i00, _i10 = _idx[:-1, :-1].ravel(), _idx[1:, :-1].ravel()
_i11, _i01 = _idx[1:, 1:].ravel(), _idx[:-1, 1:].ravel()
_tris = np.vstack([np.stack([_i00, _i10, _i11], 1), np.stack([_i00, _i11, _i01], 1)])
FACE_CELLS = np.hstack([np.full((len(_tris), 1), 3), _tris]).ravel()


# --------------------------------------------------
# TREES (same rules as 2DT_trees.py)
# --------------------------------------------------

MAX_SIDE = 8                     # max small trees per stock
MAIN_MIN, MAIN_MAX = 0.6, 3.0    # main tree height range (min for weakest, max for best)
SIDE_FRAC = 0.2                  # side tree height as a fraction of the main tree height
MAIN_FOR_LOSERS = True           # False -> stocks with negative returns get no trees at all
TREE_LAND = 0.6                  # trees only where the surface is above this (land, after the +0.5)

# Fixed per-stock random (angle, radial fraction) so side trees don't jump between
# frames. Drawn in the same order as 2DT_trees.py, so trees sit in the same places.
_tree_rng = np.random.default_rng(0)
TREE_OFFSETS = {(i, k): np.column_stack([_tree_rng.uniform(0, 2 * np.pi, MAX_SIDE),
                                         _tree_rng.uniform(0, 1, MAX_SIDE)])
                for i in range(len(sectors))
                for k in range(len(sector_tickers[i]))}

_u_local = np.linspace(0, cell_size, res)       # tile axis, as 2DT_trees.py's gx within a cell


def surface_z(Z, x, y):
    ix = np.clip(np.searchsorted(_u_local, x), 0, res - 1)
    iy = np.clip(np.searchsorted(_u_local, y), 0, res - 1)
    return Z[ix, iy]


def veg_scores(tiles):
    """Per sector: 0 for losers, else the stock's percentile rank of mean return across all sectors."""
    from scipy.stats import rankdata

    all_ret = np.concatenate([t[3] for t in tiles])
    pct = (rankdata(all_ret) - 1) / max(len(all_ret) - 1, 1)
    score = np.where(all_ret > 0, pct, 0.0)
    return np.split(score, np.cumsum([len(t[3]) for t in tiles])[:-1])


def tree_local(tiles):
    """Per sector: (m, 4) array of x, y, z, size in tile coordinates (None if no trees)."""
    out = []
    for i, ((Z, xy, _, _), sc) in enumerate(zip(tiles, veg_scores(tiles))):
        local = []                                   # (x, y, z, size) in tile coords
        for k, ((px, py), s) in enumerate(zip(xy, sc)):
            if s == 0 and not MAIN_FOR_LOSERS:
                continue
            # main tree, on the peak
            main_size = MAIN_MIN + (MAIN_MAX - MAIN_MIN) * s
            z_peak = surface_z(Z, np.array([px]), np.array([py]))[0]
            if z_peak <= TREE_LAND:
                continue                             # peak is not on land
            local.append((px, py, z_peak, main_size))
            # small trees around it
            n = int(round(MAX_SIDE * s))
            if n == 0:
                continue
            ang, frac = TREE_OFFSETS[(i, k)][:n, 0], TREE_OFFSETS[(i, k)][:n, 1]
            dist = 0.22 * main_size + 0.15 + frac * 0.6   # start outside the main canopy
            sx, sy = px + dist * np.cos(ang), py + dist * np.sin(ang)
            z = surface_z(Z, sx, sy)
            ok = z > TREE_LAND
            local.extend((x, y, zz, SIDE_FRAC * main_size) for x, y, zz in zip(sx[ok], sy[ok], z[ok]))
        out.append(np.array(local) if local else None)
    return out


def tree_frame(tiles, scene):
    """-> (points (m, 3), up directions (m, 3), sizes (m,)) for every tree that day.

    `scene` supplies the geometry (scene.place / scene.up), so the cube and the
    sphere share the tree rules and only differ in where a tile point ends up.
    """
    pts, nrm, sizes = [], [], []
    local = getattr(tiles, "trees", None)            # precompiled (cube_cache/)
    if local is None:
        local = tree_local(tiles)
    for sector, L in zip(sectors, local):
        if L is None:
            continue
        a, b = L[:, 0] - HALF, L[:, 1] - HALF
        pts.append(scene.place(sector, a, b, L[:, 2]))
        nrm.append(scene.up(sector, a, b))
        sizes.append(L[:, 3] * scene.tree_scale)
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.vstack(pts), np.vstack(nrm), np.concatenate(sizes)


def tree_meshes(points, normals, sizes):
    """Trunks and canopies as two glyph meshes, standing along each face's normal.

    Same templates as build_terrain.show_terrain_time_trees (height 1, base at
    the origin), but built along +x because glyphs turn +x onto the normal.
    """
    import pyvista as pv

    if len(points) == 0:                             # glyphing an empty cloud fails
        points, normals, sizes = np.zeros((1, 3)), np.array([[1.0, 0, 0]]), np.array([1e-6])
    cloud = pv.PolyData(points)
    cloud["normal"] = normals
    cloud["size"] = sizes
    trunk = pv.Cylinder(center=(0.15, 0, 0), direction=(1, 0, 0), radius=0.04, height=0.3,
                        resolution=8)
    canopy = pv.Cone(center=(0.65, 0, 0), direction=(1, 0, 0), height=0.7, radius=0.22,
                     resolution=10)
    return (cloud.glyph(geom=trunk, orient="normal", scale="size", factor=1.0),
            cloud.glyph(geom=canopy, orient="normal", scale="size", factor=1.0))


MAX_TREES = sum(len(t) for t in sector_tickers) * (1 + MAX_SIDE)


class TreeLayer:
    """Trunks and canopies as two fixed-size meshes whose points move each day.

    Same shapes as tree_meshes, but instead of glyphing (and adding new actors)
    every frame, each template is placed with numpy into slots for MAX_TREES
    trees; unused slots collapse to a point. Nothing in VTK is rebuilt.
    """

    def __init__(self, plotter):
        import pyvista as pv

        self.plotter = plotter
        trunk = pv.Cylinder(center=(0.15, 0, 0), direction=(1, 0, 0), radius=0.04, height=0.3,
                            resolution=8).triangulate()
        canopy = pv.Cone(center=(0.65, 0, 0), direction=(1, 0, 0), height=0.7, radius=0.22,
                         resolution=10).triangulate()
        canopy = canopy.compute_normals(cell_normals=False, split_vertices=False)
        self.meshes, self.templates, self.actors = [], [], []
        for tmpl, color, smooth in ((trunk, "saddlebrown", False), (canopy, "forestgreen", True)):
            n = tmpl.n_points
            faces = tmpl.faces.reshape(-1, 4)
            offs = (np.arange(MAX_TREES) * n)[:, None, None]
            all_faces = np.concatenate(
                [faces[None, :, :1].repeat(MAX_TREES, 0), faces[None, :, 1:] + offs], axis=2)
            mesh = pv.PolyData(np.zeros((MAX_TREES * n, 3)), all_faces.ravel())
            nrm = np.asarray(tmpl.point_data["Normals"]) if smooth else None
            if smooth:
                mesh.point_data["Normals"] = np.zeros((MAX_TREES * n, 3))
                mesh.point_data.active_normals_name = "Normals"
            actor = plotter.add_mesh(mesh, color=color, render=False)
            if smooth:
                actor.prop.interpolation = "phong"
            self.meshes.append(mesh)
            self.templates.append((np.asarray(tmpl.points), nrm))
            self.actors.append(actor)

    @staticmethod
    def _basis(up):
        """Rows: (up, side1, side2) per tree, so template +x maps onto `up`."""
        e1 = up / np.linalg.norm(up, axis=1, keepdims=True)
        helper = np.where(np.abs(e1[:, 2:3]) < 0.9, [[0.0, 0, 1]], [[0.0, 1, 0]])
        e2 = np.cross(e1, helper)
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
        return np.stack([e1, e2, np.cross(e1, e2)], axis=1)       # (m, 3, 3)

    def update(self, points, ups, sizes):
        m = min(len(points), MAX_TREES)
        R = self._basis(ups[:m]) if m else np.zeros((0, 3, 3))
        for mesh, (tp, tn) in zip(self.meshes, self.templates):
            n = len(tp)
            out = np.zeros((MAX_TREES, n, 3))
            # template point p -> centre + size * (p @ R)
            out[:m] = points[:m, None, :] + sizes[:m, None, None] * (tp @ R)   # (m, n, 3)
            if m:
                out[m:] = points[0]                              # unused slots: a single point
            mesh.points = out.reshape(-1, 3)
            if tn is not None:
                nn = np.zeros((MAX_TREES, n, 3))
                nn[:, :, 0] = 1.0
                nn[:m] = tn @ R
                mesh.point_data["Normals"] = nn.reshape(-1, 3)

    def set_visible(self, on):
        for a in self.actors:
            a.SetVisibility(on)


# --------------------------------------------------
# BACKGROUND PRECOMPUTATION (days not in struct_Z.npy)
# --------------------------------------------------

def tile_task(day, i):
    """Runs in a worker process: one sector's height map for one day."""
    Z, xy, heights, mean_returns = sector_terrain(i, vol["date"].iloc[day])
    return day, i, Z.astype(np.float32), xy, heights, mean_returns


class Precomputer:
    def __init__(self, first, last, saved: SavedTerrain, workers=None, keep_days=800):
        self.first, self.last = first, last
        self.saved = saved
        # 6 workers: about 3.5 s to start and ~6 days/s; more workers start much
        # slower (they all load the data at once) for little extra throughput.
        self.workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
        self.pool = None               # started on first use, so saved days need no workers
        self.in_flight = {}            # future -> (day, sector index)
        self.partial = {}              # day -> {sector index: tile}
        self.grown = {}                # day -> list of six tiles, in sector order
        self.keep_days = keep_days     # memory cap for grown days (~1 MB per day)
        self.target = first

    # -- lookups ---------------------------------------------------------
    def is_ready(self, day):
        return self.saved.has(day) or day in self.grown

    def tiles(self, day):
        return self.saved.tiles(day) if self.saved.has(day) else self.grown[day]

    def nearest_ready(self, day):
        best = self.saved.nearest(day)
        best_d = None if best is None else abs(best - day)
        for d in self.grown:
            if best_d is None or abs(d - day) < best_d:
                best, best_d = d, abs(d - day)
        return best

    # -- background work -------------------------------------------------
    def _wanted(self):
        """Days not yet available, nearest to the target first (within keep_days)."""
        busy = {d for d, _ in self.in_flight.values()}
        for dist in range(self.keep_days // 2):
            for d in (self.target - dist, self.target + dist + 1):
                if self.first <= d <= self.last and not self.is_ready(d) and d not in busy:
                    yield d

    def top_up(self):
        """Keep ~2 tiles per worker queued, always for the days nearest the target."""
        want = self.workers * 2
        queued = set(self.in_flight.values())
        for d in self._wanted():
            if len(self.in_flight) >= want:
                break
            if self.pool is None:
                self.pool = ProcessPoolExecutor(self.workers)
                print(f"Growing days outside struct_Z.npy on {self.workers} worker processes...")
            have = self.partial.get(d, {})
            for i in range(len(sectors)):
                if i not in have and (d, i) not in queued:
                    self.in_flight[self.pool.submit(tile_task, d, i)] = (d, i)

    def retarget(self, day):
        """Slider moved: drop queued work far from the new day, then refill around it."""
        self.target = day
        for f, (d, _) in list(self.in_flight.items()):
            if abs(d - day) > self.workers and f.cancel():
                del self.in_flight[f]
        self.top_up()

    def collect(self):
        """Move finished tiles into `grown`. Returns True if a new day completed."""
        new_day = False
        for f in [f for f in self.in_flight if f.done()]:
            del self.in_flight[f]
            if f.cancelled():
                continue
            day, i, *tile = f.result()
            tiles = self.partial.setdefault(day, {})
            tiles[i] = tuple(tile)
            if len(tiles) == len(sectors):
                self.grown[day] = [tiles[k] for k in range(len(sectors))]
                del self.partial[day]
                new_day = True
        if len(self.grown) > self.keep_days:     # forget the days furthest from the slider
            far = sorted(self.grown, key=lambda d: -abs(d - self.target))
            for d in far[:len(self.grown) - self.keep_days]:
                del self.grown[d]
        self.top_up()
        return new_day

    def build_here(self, day):
        """Grow a day in this process (no waiting for worker processes to start up)."""
        if not self.is_ready(day):
            self.grown[day] = [tile_task(day, i)[2:] for i in range(len(sectors))]

    def shutdown(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------

FIRST_DAY = window        # 2DT.py starts at row `window`, so the correlation window is full
LAST_DAY = len(vol) - 1

TICKER_INFO = univ.set_index("ticker")[["name", "sector"]]

# How close (in scene units) the mouse must be to a peak to show its stock.
HOVER_RADIUS = 1.2


class CubeScene:
    """Six face meshes that are created once and updated in place each day.

    Subclasses change the shape by overriding `place` / `up`.
    """
    hover_radius = HOVER_RADIUS
    tree_scale = 1.0               # tree size multiplier
    tooltip_font = 11

    def __init__(self, plotter):
        self.plotter = plotter
        self.meshes, self.actors = [], []
        self.tree, self.labels, self.day = None, None, None
        self.cmap = green_mountain_cmap()
        self.show_trees = True
        self.tiles = None
        self.trees = None          # TreeLayer, made on first draw
        self.peaks = None          # (n, 3) hover anchors, same order as self.labels

    # -- geometry (a, b are face coords centred on the face, h is height) ---
    def place(self, sector, a, b, h):
        x, y, z = face_to_xyz(sector, np.asarray(a, float), np.asarray(b, float),
                              np.asarray(h, float))
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1)

    def up(self, sector, a, b):
        axis, sign = FACE_AXES[sector]
        n = np.zeros(3)
        n[axis] = sign
        return np.broadcast_to(n, np.shape(a) + (3,)).copy()

    # -- hooks ----------------------------------------------------------------
    


    
  

# How close (in scene units) the mouse must be to a peak to show its stock.
HOVER_RADIUS = 1.2


class CubeScene:
    """Six face meshes that are created once and updated in place each day.

    Subclasses change the shape by overriding `place` / `up`.
    """
    hover_radius = HOVER_RADIUS
    tree_scale = 1.0               # tree size multiplier
    tooltip_font = 11

    def __init__(self, plotter):
        self.plotter = plotter
        self.meshes, self.actors = [], []
        self.tree, self.labels, self.day = None, None, None
        self.cmap = green_mountain_cmap()
        self.show_trees = True
        self.tiles = None
        self.trees = None          # TreeLayer, made on first draw
        self.peaks = None          # (n, 3) hover anchors, same order as self.labels

    # -- geometry (a, b are face coords centred on the face, h is height) ---
    def place(self, sector, a, b, h):
        x, y, z = face_to_xyz(sector, np.asarray(a, float), np.asarray(b, float),
                              np.asarray(h, float))
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1)

    def up(self, sector, a, b):
        axis, sign = FACE_AXES[sector]
        n = np.zeros(3)
        n[axis] = sign
        return np.broadcast_to(n, np.shape(a) + (3,)).copy()

    # -- hooks ----------------------------------------------------------------
    def decorate(self):
        """One-off scene additions after the first day is drawn."""
        self.plotter.add_axes()
        self.plotter.show_grid()

        # Put one sector name just outside the centre of each cube face.
        # The points sit beyond the tallest terrain so the labels remain
        # readable and do not get buried by changing peaks.
        label_points = []
        label_text = []
        label_offset = HALF + hmax + land_lift + 0.8

        for sector in sectors:
            axis, sign = FACE_AXES[sector]
            point = np.zeros(3, dtype=float)
            point[axis] = sign * label_offset
            label_points.append(point)
            label_text.append(sector)

        self.sector_label_actor = self.plotter.add_point_labels(
            np.asarray(label_points),
            label_text,
            font_size=14,
            text_color="black",
            shape=None,
            always_visible=True,
            render=False,
        )

    def highlight(self, index):
        """Hovered stock index (or None). The cube relies on the tooltip only."""

    def show(self, day, tiles):
        import pyvista as pv
        from scipy.spatial import cKDTree

        first = not self.meshes
        all_points, all_labels = [], []
        for i, (sector, (Z, xy, heights, _)) in enumerate(zip(sectors, tiles)):
            pts = self.place(sector, A, B, Z).reshape(-1, 3)
            if first:
                mesh = pv.PolyData(pts, FACE_CELLS)
            else:
                mesh = self.meshes[i]
                mesh.points = pts
            mesh.point_data["height"] = np.asarray(Z).ravel()
            # Point normals give smooth shading; recomputed because the shape changed.
            mesh.compute_normals(inplace=True, cell_normals=False, split_vertices=False,
                                 consistent_normals=False, auto_orient_normals=False)
            mesh.set_active_scalars("height")
            if first:
                self.meshes.append(mesh)
                self.actors.append(self.plotter.add_mesh(
                    mesh, scalars="height", cmap=self.cmap, clim=CLIM, show_edges=False,
                    show_scalar_bar=(i == 0), render=False))
            all_points.append(self.place(sector, xy[:, 0] - HALF, xy[:, 1] - HALF,
                                         heights + land_lift + 0.05))   # on top of each peak
            all_labels.extend(sector_tickers[i])
        self.peaks = np.vstack(all_points)
        self.tree, self.labels, self.day = cKDTree(self.peaks), all_labels, day
        self.tiles = tiles
        self.draw_trees()

    def draw_trees(self):
        if self.trees is None:
            self.trees = TreeLayer(self.plotter)       # created once, then moved in place
        self.trees.set_visible(self.show_trees)
        if self.show_trees:
            self.trees.update(*tree_frame(self.tiles, self))


def run(scene_cls=CubeScene):
    import pyvista as pv
    import vtk

    plotter = pv.Plotter()

    # BACKGROUND
    if USE_BACKGROUND:
        background_texture = pv.read_texture("background_2.jpeg")
        plotter.add_actor(background_texture.to_skybox())
        plotter.set_environment_texture(
        background_texture, is_srgb=True
        )

    scene = scene_cls(plotter)
    saved = Sources(CachedTerrain(), SavedTerrain())
    pre = Precomputer(FIRST_DAY, LAST_DAY, saved)

    # Start on the first saved day (instant); without struct_Z.npy, on the latest day.
    start = saved.first if saved.Z is not None else LAST_DAY
    state = {"want": start}
    pre.build_here(start)         # no-op for a saved day; else grown here, before workers start
    scene.show(start, pre.tiles(start))
    pre.retarget(start)           # workers start (lazily) only for days the npy doesn't cover

    # Picker that finds the exact spot on the cube under the mouse (cube only, not axes/grid).
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)
    picker.PickFromListOn()
    for a in scene.actors:
        picker.AddPickList(a)

    date_text = plotter.add_text("", name="date", position="upper_left", font_size=14)
    tooltip = plotter.add_text("", name="hover", position=(0, 0), font_size=scene.tooltip_font)
    tooltip.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
    tooltip.GetTextProperty().SetBackgroundOpacity(0.8)
    tooltip.GetTextProperty().SetColor(1, 1, 1)

    def date_str(day):
        return str(vol["date"].iloc[day].date())

    def refresh():
        """Show the wanted day if it's ready, otherwise the nearest one that is."""
        want = state["want"]
        day = want if pre.is_ready(want) else pre.nearest_ready(want)
        if day is not None and day != scene.day:
            scene.show(day, pre.tiles(day))
            tooltip.SetInput("")
        label = date_str(scene.day)
        if scene.day != want:
            label += f"   (growing {date_str(want)}...)"
        date_text.SetText(2, label)                  # corner 2 = upper left; one actor, updated in place
        plotter.render()

    def on_slide(value):
        want = int(round(value))
        if want != state["want"]:
            state["want"] = want
            pre.retarget(want)
            refresh()

    def on_timer(*_):
        if pre.collect():
            refresh()

    def on_mouse_move(interactor, _event):
        x, y = interactor.GetEventPosition()
        text, hit = "", None
        if picker.Pick(x, y, 0, plotter.renderer):
            distance, index = scene.tree.query(picker.GetPickPosition())
            if distance < scene.hover_radius:
                hit = int(index)
                ticker = scene.labels[index]
                name, sector = TICKER_INFO.loc[ticker]
                volatility = vol[ticker].iloc[scene.day]
                text = f"{ticker}  {name}\n{sector}\nVolatility: {volatility:.3f}"
        scene.highlight(hit)
        if text != tooltip.GetInput():
            tooltip.SetInput(text)
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()
        elif text:
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()

    plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)
    plotter.iren.add_observer("TimerEvent", on_timer)
    # On Windows a timer created before the interactor is initialised never fires,
    # and then finished days are never picked up (the terrain stays frozen).
    plotter.iren.initialize()
    plotter.iren.create_timer(150)          # pick up finished tiles ~7 times a second

    # Swapping in a ready day is cheap, so the slider updates while dragging.
    slider = plotter.add_slider_widget(
        on_slide,
        rng=[FIRST_DAY, LAST_DAY],
        value=start,
        title="Trading day",
        fmt="%.0f",
        pointa=(0.35, 0.93),
        pointb=(0.95, 0.93),
        style="modern",
        interaction_event="always",
    )

    def step(delta):
        day = int(np.clip(state["want"] + delta, FIRST_DAY, LAST_DAY))
        slider.GetRepresentation().SetValue(day)
        on_slide(day)

    plotter.add_key_event("Right", lambda: step(1))
    plotter.add_key_event("Left", lambda: step(-1))

    def toggle_trees():
        scene.show_trees = not scene.show_trees
        scene.draw_trees()
        plotter.render()

    plotter.add_key_event("t", toggle_trees)

    scene.decorate()
    refresh()
    try:
        plotter.show()
    finally:
        pre.shutdown()


if __name__ == "__main__":   # required: worker processes re-import this file on Windows
    run()

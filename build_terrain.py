import heapq
import math

import numpy as np
from numba import njit

# pyvista / matplotlib are imported inside the show_* functions, and the data is
# no longer loaded here: worker processes that only grow terrain import this
# module, and nothing used the module-level vol / univ / returns.

TAU = 2 * math.pi
N_TABLE = 200
PAD = 2   # border of cells around the grid, so no bounds checks are needed

# 16-neighbourhood: 8 adjacent cells plus the 8 "knight moves".
OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
]
SLOPE_SCALE_FACTOR = 5 # higher = steeper slopes

def smooth_noise(rng, n, scale):
    """Gaussian-blurred white noise (correlation length `scale` cells), unit std."""
    f2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
    spec = np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-2 * np.pi**2 * scale**2 * f2)
    field = np.fft.ifft2(spec).real
    return field / field.std()


def make_slope_table(rng, base_slope, anisotropy, n_waves=3, n_table=N_TABLE, min_slope=0.02):
    """Slope as a function of angle, as a lookup table over [0, 2*pi)."""
    phases = rng.uniform(0, 2 * np.pi, size=n_waves)
    amps = rng.uniform(0.5, 1, size=n_waves)
    freqs = np.arange(1, n_waves + 1)            # integer frequencies keep it periodic
    vals = np.linspace(0, 2 * np.pi, n_table, endpoint=False)

    x = sum(amps[k] * np.cos(freqs[k] * vals + phases[k]) for k in range(n_waves))
    x = x / np.max(np.abs(x)) * SLOPE_SCALE_FACTOR
    return np.maximum(base_slope + anisotropy * x, min_slope)


@njit(cache=True)
def _nearest_lower(xs, ys, peaks, heights, order):
    """Row k: for every cell, distance to / height of the closest of the k lowest peaks."""
    P, n = order.shape[0], xs.shape[0]
    near_d = np.zeros((P + 1, n))
    near_z = np.zeros((P + 1, n))
    best_d = np.full(n, np.inf)
    best_z = np.zeros(n)
    for k in range(1, P + 1):
        q = order[k - 1]
        px, py, hz = peaks[q, 0], peaks[q, 1], heights[q]
        for c in range(n):
            d = math.hypot(xs[c] - px, ys[c] - py)
            if d < best_d[c]:
                best_d[c] = d
                best_z[c] = hz
        near_d[k] = best_d
        near_z[k] = best_z
    return near_d, near_z


@njit(cache=True)
def _grow(Z, S, seeds, peaks, xs, ys, heights, tables, mult, edge_factor, slope_noise,
          sorted_heights, near_d, near_z, move_off, move_step,
          protect_peaks, rho, overshadow_margin, max_boost, floor):
    """The priority-queue growth loop, compiled. Fills Z and S in place.

    Identical logic to the original pure-Python loop: the heap holds
    (-height, flat_index, peak_id) tuples, the highest cell is popped and offers
    each of the 16 neighbours a lower height.
    """
    n_table = tables.shape[1]
    M = move_off.shape[0]
    n_peaks = sorted_heights.shape[0]

    heap = [(0.0, 0, 0)]          # typed list of (-h, cell, pid); seeded below
    heap.pop()
    for s in range(seeds.shape[0]):
        c, pid = seeds[s, 0], seeds[s, 1]
        z0 = heights[pid]
        if z0 > Z[c]:             # Z[c] starts at -inf
            Z[c] = z0             # sets cell to peak's exact height
            S[c] = tables[pid].sum() / n_table   # start from the peak's own mean slope
            heap.append((-z0, c, pid))
    heapq.heapify(heap)

    one_minus_rho = 1.0 - rho
    while len(heap) > 0:
        neg_h, c, pid = heapq.heappop(heap)
        h = -neg_h
        if h < Z[c]:
            continue                              # stale entry
        prev = S[c] + 1e-8                        # slope that produced this cell's height
        prev_pow = prev ** rho

        k = 0                                     # number of peaks lower than h (bisect_left)
        if protect_peaks:
            lo, hi = 0, n_peaks
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_heights[mid] < h:
                    lo = mid + 1
                else:
                    hi = mid
            k = lo
        c_edge = c * M
        px, py = peaks[pid, 0], peaks[pid, 1]

        for m in range(M):
            nc = c + move_off[m]
            e = c_edge + m
            th = math.atan2(ys[nc] - py, xs[nc] - px) % TAU
            t_idx = int(th / TAU * n_table) % n_table      # angle -> slope table index
            target = tables[pid, t_idx] * mult[nc] + 1e-8
            slope = prev_pow * target ** one_minus_rho * slope_noise[e]
            drop = move_step[m] * edge_factor[e]  # horizontal step, with noise

            eff = slope
            if k:
                # slope that would bring the terrain down to (summit - margin) at the
                # closest lower peak, if we kept descending through this neighbour
                s_req = (h - near_z[k, nc] + overshadow_margin) / (drop + near_d[k, nc])
                if s_req > slope:                 # would bury
                    eff = min(s_req, max_boost * slope)

            h_new = h - eff * drop
            if h_new > floor and h_new > Z[nc]:
                Z[nc] = h_new
                S[nc] = slope                     # store the unboosted slope, so the boost
                heapq.heappush(heap, (-h_new, nc, pid))  # does not propagate through inheritance


def build_terrain(
    peaks,                  # list of (x, y) positions
    heights,                # list of z values, one per peak
    slope_mean=1.5,         # mean of the base slope across peaks
    slope_std=0.3,          # std of the base slope across peaks
    anisotropy=0.20,        # how strongly the slope varies with direction
    n_harmonics=4,          # how wiggly that variation is
    edge_noise=0.15,        # relative noise on each step: (1 + N(0, edge_noise))
    rho=0.5,                # slope inheritance: 0 = ignore parent, 1 = copy parent
    sigma=0.10,             # log-normal noise on the slope of each edge
    field_strength=0.3,     # how strongly the smooth field modulates the slope
    field_scales=((15, 1.0), (5, 0.5)),   # (correlation length in cells, weight), is pairs of (scale, weight)
    # scale = correlation length in cells, ie. width of gaussian blur applied to white noise
    # weight = how strongly scale gets weighted
    protect_peaks=True,     # steepen slopes towards lower peaks so they are not swallowed
    overshadow_margin=0.3,  # terrain should be this far below a lower peak's summit at the peak
    max_boost=3.0,          # a slope is never increased by more than this factor
    final_noise_std=0.0,    # optional plain white noise added at the end
    extent=(-10, 10, -10, 10),
    resolution=200,
    floor=0.0,
    seed=0,
    return_slopes=False,
):
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, dtype=float)
    heights = np.asarray(heights, dtype=float)

    n = resolution
    W = n + 2 * PAD                       # padded grid width
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # ---------- precomputation (numpy, once) ----------
    base_slopes = np.abs(rng.normal(slope_mean, slope_std, size=len(peaks)))
    tables = np.array([
        make_slope_table(rng, bs, anisotropy, n_harmonics) for bs in base_slopes
    ]).reshape(len(peaks), N_TABLE)

    # Padded coordinates, so every cell (including the border) has a position
    xs_p = xmin + dx * np.arange(-PAD, n + PAD)
    ys_p = ymin + dy * np.arange(-PAD, n + PAD)
    Xp, Yp = np.meshgrid(xs_p, ys_p, indexing="ij")

    # The angle of each cell as seen from each peak is computed inside _grow, only
    # for the (peak, cell) pairs actually visited, instead of a full table per peak.
    xs_flat = Xp.ravel().copy()
    ys_flat = Yp.ravel().copy()

    # Smooth multiplier field, computed ONCE (exp keeps the slope positive)
    field = sum(w * smooth_noise(rng, W, s) for s, w in field_scales) # shape (W)
    mult = np.exp(field_strength * field).ravel()

    # Per-edge random numbers, drawn in one go (one value per cell-and-offset pair)
    M = len(OFFSETS)

    # edge_factor = multiplicative noise on horizontal step: drop = step * edge_factor[e]
    edge_factor = np.maximum(1.0 + rng.normal(0.0, edge_noise, size=W * W * M), 0.1) # one edge: (cell, offset) pair
    slope_noise = np.exp(sigma * rng.normal(size=W * W * M))

    # Peak protection: peaks sorted by height (ascending). Row k of near_d / near_z
    # holds, for every cell, the distance to and the height of the closest peak
    # among the k LOWEST peaks (row 0 is unused). For a cell at height h, the peaks
    # lower than h are exactly the first k = searchsorted(sorted_heights, h) ones.
    order = np.argsort(heights)
    sorted_heights = heights[order]
    if protect_peaks:
        near_d, near_z = _nearest_lower(xs_flat, ys_flat, peaks, heights, order)
    else:
        near_d = near_z = np.zeros((1, W * W))

    # Moves: flat index offset and horizontal step length of each neighbour
    move_off = np.array([di * W + dj for di, dj in OFFSETS], dtype=np.int64)
    move_step = np.array([math.hypot(di * dx, dj * dy) for di, dj in OFFSETS])

    # ---------- state ----------
    # Border cells hold +inf so no offer is ever accepted there.
    Z_init = np.full((W, W), np.inf) # inf = acts as a wall
    Z_init[PAD:-PAD, PAD:-PAD] = -np.inf          # -inf = not reached yet
    Z = Z_init.ravel()
    S = np.zeros(W * W) # slope of the winning offer per cell

    # Seed cells: (flat index, peak id) for every peak
    seeds = np.empty((len(peaks), 2), dtype=np.int64)
    for pid, (px, py) in enumerate(peaks):
        i = min(max(int(round((px - xmin) / dx)), 0), n - 1)
        j = min(max(int(round((py - ymin) / dy)), 0), n - 1)
        seeds[pid] = ((i + PAD) * W + (j + PAD), pid)

    # ---------- main loop (compiled with numba, see _grow) ----------
    _grow(Z, S, seeds, peaks, xs_flat, ys_flat, heights, tables, mult, edge_factor, slope_noise,
          sorted_heights, near_d, near_z, move_off, move_step,
          bool(protect_peaks), float(rho), float(overshadow_margin), float(max_boost), float(floor))

    # ---------- back to numpy ----------
    Zout = Z.reshape(W, W)[PAD:-PAD, PAD:-PAD].copy()
    Zout[np.isinf(Zout)] = floor                  # cells no peak reached before the floor

    if final_noise_std > 0:
        Zout = np.maximum(Zout + rng.normal(0.0, final_noise_std, size=Zout.shape), floor)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if return_slopes:
        slopes = S.reshape(W, W)[PAD:-PAD, PAD:-PAD].copy()
        return X, Y, Zout, slopes
    return X, Y, Zout


def show_terrain(X, Y, Z):
    """Display the complete terrain using PyVista."""
    import pyvista as pv

    grid = pv.StructuredGrid(X, Y, Z)

    grid["height"] = Z.flatten(order="F")

    plotter = pv.Plotter()

    plotter.add_mesh(
        grid,
        scalars="height",
        cmap="terrain",
        smooth_shading=True,
        show_edges=False,
    )

    plotter.add_axes()
    plotter.show_bounds(grid="back")

    plotter.show()



def show_terrain_time(GX, GY, GZ, time_values):
    """
    Display precomputed terrain frames using a PyVista slider.

    GX, GY: Fixed spatial coordinate grids (2D)
    GZ: Precomputed terrain heights (3D: x, y, time)
    time_values: Corresponding timestamps
    """

    import numpy as np
    import pyvista as pv

    n_times = GZ.shape[2]

    assert len(time_values) == n_times

    # ----------------------------------------------
    # Initialise first frame
    # ----------------------------------------------

    Z = GZ[:, :, 0]

    grid = pv.StructuredGrid(GX, GY, Z)

    grid["height"] = Z.flatten(order="F")

    plotter = pv.Plotter()

    # Fixed colour scale across all timestamps
    plotter.add_mesh(
        grid,
        scalars="height",
        cmap="terrain",
        clim=(0, np.max(GZ)),
        smooth_shading=True,
        show_edges=False,
    )

    plotter.add_axes()
    plotter.show_bounds(grid="back")

    # ----------------------------------------------
    # Date label
    # ----------------------------------------------

    date_label = plotter.add_text(
        str(time_values[0]),
        position="upper_left",
        font_size=12,
    )

    # ----------------------------------------------
    # Fixed spatial coordinates
    # ----------------------------------------------

    x_flat = GX.flatten(order="F")
    y_flat = GY.flatten(order="F")

    # ----------------------------------------------
    # Slider callback
    # ----------------------------------------------

    def update_terrain(value):

        t = int(round(value))

        # Simply select precomputed terrain
        Z_new = GZ[:, :, t]

        # Update mesh geometry
        grid.points = np.column_stack([
            x_flat,
            y_flat,
            Z_new.flatten(order="F")
        ])

        # Update height colouring
        grid["height"] = Z_new.flatten(order="F")

        # Update date label
        date_label.SetInput(str(time_values[t]))

        plotter.render()

    # ----------------------------------------------
    # Slider
    # ----------------------------------------------

    plotter.add_slider_widget(
        update_terrain,
        rng=(0, n_times - 1),
        value=0,
        title="Time",
        interaction_event="always",
    )

    plotter.show()


def make_trees(frame):
    """frame: (n, 4) array of x, y, z, size -> (trunk_mesh, canopy_mesh)"""
    import pyvista as pv
    if len(frame) == 0:
        # dummy invisible point, because glyphing an empty cloud fails
        frame = np.array([[0.0, 0.0, -100.0, 1e-3]])

    pts = pv.PolyData(frame[:, :3])
    pts["size"] = frame[:, 3]

    trunks = pts.glyph(geom=trunk_geom, scale="size", factor=1.0, orient=False)
    canopies = pts.glyph(geom=canopy_geom, scale="size", factor=1.0, orient=False)
    return trunks, canopies


def show_terrain_time_trees(GX, GY, GZ, time_values, trees=None, background_image=None):
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap

    print('background', background_image)

    # COLOR MAP

    # zmax = float(np.max(GZ))
    # sea = 0.5                      # sea level (land starts above this)
    # n = 256
    # n_water = int(np.clip(sea / zmax, 0, 1) * n)
    # n_land = n - n_water
    # water = np.tile([0.1, 0.35, 0.7, 1.0], (n_water, 1))
    # # land: light green at low elevation -> dark green at high elevation
    # light = np.array([0.55, 0.80, 0.40, 1.0])
    # dark  = np.array([0.10, 0.45, 0.15, 1.0])
    # land = np.linspace(light, dark, n_land)
    # land_cmap = ListedColormap(np.vstack([water, land]))

    zmax = float(np.max(GZ))
    sea = 0.5                       # sea level
    s = sea / zmax                  # sea level as a fraction of the full range

    def land(f):
        """Map f in [0, 1] (0 = shoreline, 1 = highest peak) to a colormap position."""
        return s + f * (1 - s)

    stops = [
        (0.0,       "#1a5ab3"),     # deep water
        (s,         "#1a5ab3"),     # water up to sea level (hard edge)
        (land(0.00), "#8ccc66"),    # lowland: light green
        (land(0.30), "#2f8a3a"),    # hills: darker green
        (land(0.50), "#d9a441"),    # foothills: yellow-orange
        (land(0.65), "#c8662b"),    # mountains: orange
        (land(0.80), "#7d7d7d"),    # rock: gray
        (1.0,       "#e6e6e6"),     # peaks: light gray
    ]
    terrain_cmap = LinearSegmentedColormap.from_list("green_mountain", stops, N=512)

    n_times = GZ.shape[2]
    assert len(time_values) == n_times
    if trees is not None:
        assert len(trees) == n_times

    Z = GZ[:, :, 0]
    grid = pv.StructuredGrid(GX, GY, Z)
    grid["height"] = Z.flatten(order="F")

    plotter = pv.Plotter()
    # plotter.add_mesh(
    #     grid, scalars="height", cmap="terrain",
    #     clim=(0, np.max(GZ)), smooth_shading=True, show_edges=False,
    # )
    # plotter.add_mesh(
    #     grid, scalars="height", cmap=land_cmap,
    #     clim=(0, zmax), smooth_shading=True, show_edges=False,
    # )

    if background_image is not None:
        background_texture = pv.read_texture(background_image)

        # Surrounds the scene; rotates naturally with the camera view
        plotter.add_actor(background_texture.to_skybox())

        # Optional: use the image for scene lighting and reflections
        plotter.set_environment_texture(background_texture, is_srgb=True)

    plotter.add_mesh(
        grid, scalars="height", cmap=terrain_cmap,
        clim=(0, zmax), smooth_shading=True, show_edges=False,
    )
    
    plotter.add_axes()
    plotter.show_bounds(grid="back")

    # ---- tree templates: height 1, base at the origin ----
    global trunk_geom, canopy_geom
    trunk_geom = pv.Cylinder(
        center=(0, 0, 0.15), direction=(0, 0, 1),
        radius=0.04, height=0.3, resolution=8,
    )
    canopy_geom = pv.Cone(
        center=(0, 0, 0.65), direction=(0, 0, 1),
        height=0.7, radius=0.22, resolution=10,
    )

    def draw_trees(t):
        if trees is None:
            return
        trunks, canopies = make_trees(trees[t])
        # same `name` replaces the previous actor instead of stacking a new one
        plotter.add_mesh(trunks, color="saddlebrown", name="trunks", render=False)
        plotter.add_mesh(canopies, color="forestgreen", name="canopies",
                         smooth_shading=True, render=False)

    draw_trees(0)

    date_label = plotter.add_text(str(time_values[0]), position="upper_left", font_size=12)

    x_flat = GX.flatten(order="F")
    y_flat = GY.flatten(order="F")

    def update_terrain(value):
        t = int(round(value))
        Z_new = GZ[:, :, t]
        grid.points = np.column_stack([x_flat, y_flat, Z_new.flatten(order="F")])
        grid["height"] = Z_new.flatten(order="F")
        draw_trees(t)
        date_label.SetInput(str(time_values[t]))
        plotter.render()

    plotter.add_slider_widget(
        update_terrain, rng=(0, n_times - 1), value=0,
        title="Time", interaction_event="always",
    )
    plotter.show()
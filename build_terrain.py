import bisect
import heapq
import math
import numpy as np
from scipy.ndimage import maximum_filter
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

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
    """Gaussian-blurred white noise (correlation length `scale` cells), unit std.

    1. white noise, to frequency domain, compute bins, build gaussian filter FFT(g) x FFT(f) = FFT(g * f)
    2. apply filter and go back: spec = fft(noise) * filter, then ifft2(spec) returns to space
    """
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
    tables = [
        make_slope_table(rng, bs, anisotropy, n_harmonics).tolist() for bs in base_slopes
    ]

    # Padded coordinates, so every cell (including the border) has a position
    xs_p = xmin + dx * np.arange(-PAD, n + PAD)
    ys_p = ymin + dy * np.arange(-PAD, n + PAD)
    Xp, Yp = np.meshgrid(xs_p, ys_p, indexing="ij")

    # For every peak: table index of the angle of each cell as seen from the peak
    theta_idx = []
    for px, py in peaks:
        th = np.arctan2(Yp - py, Xp - px) % TAU
        idx = (th / TAU * N_TABLE).astype(np.int64) % N_TABLE # angle -> table index
        theta_idx.append(idx.ravel().tolist())

    # Smooth multiplier field, computed ONCE (exp keeps the slope positive)
    field = sum(w * smooth_noise(rng, W, s) for s, w in field_scales) # shape (W)
    mult = np.exp(field_strength * field).ravel().tolist()

    # Per-edge random numbers, drawn in one go (one value per cell-and-offset pair)
    M = len(OFFSETS)

    # edge_factor = multiplicative noise on horizontal step: drop = step * edge_factor[e]
    edge_factor = np.maximum(1.0 + rng.normal(0.0, edge_noise, size=W * W * M), 0.1).tolist() # one edge: (cell, offset) pair
    slope_noise = np.exp(sigma * rng.normal(size=W * W * M)).tolist()

    # Peak protection: peaks sorted by height (ascending). For k = 1..P, near_d[k] and
    # near_z[k] hold, for every cell, the distance to and the height of the closest peak
    # among the k LOWEST peaks. For a cell at height h, the peaks lower than h are
    # exactly the first k = bisect_left(sorted_heights, h) ones.
    order = np.argsort(heights)
    sorted_heights = heights[order].tolist()
    near_d, near_z = [None], [None]
    if protect_peaks:
        best_d = np.full((W, W), np.inf)
        best_z = np.zeros((W, W))
        for q in order:
            d = np.hypot(Xp - peaks[q, 0], Yp - peaks[q, 1])
            closer = d < best_d
            best_d = np.where(closer, d, best_d)
            best_z = np.where(closer, heights[q], best_z)
            near_d.append(best_d.ravel().tolist())
            near_z.append(best_z.ravel().tolist())

    """
    near_d[k] = list where every entry = distance from that cell to closest of the k lowest peaks
    near_z[k] = list of same length, storing height of that closest peak
    """

    # Moves as (flat index offset, horizontal step length, offset number)
    moves = [
        (di * W + dj, math.hypot(di * dx, dj * dy), m)
        for m, (di, dj) in enumerate(OFFSETS)
    ] # neighbors as flat index offsets
    """
    each move has 3 values:
    -di * W + dj: flat index offset
    -math.hypot(...): horizontal dist covered by that step
    -m: position of offset in OFFSETS
    """

    # ---------- state: flat Python lists ----------
    # Border cells hold +inf so no offer is ever accepted there.
    Z_init = np.full((W, W), np.inf) # inf = acts as a wall
    Z_init[PAD:-PAD, PAD:-PAD] = -np.inf          # -inf = not reached yet
    Z = Z_init.ravel().tolist()
    S = [0.0] * (W * W) # slope of the winning offer per cell

    heap = []             
    # fill in heap with entries: (-height, flat_index, peak_id)
    for pid, ((px, py), z0) in enumerate(zip(peaks, heights)):
        i = min(max(int(round((px - xmin) / dx)), 0), n - 1)
        j = min(max(int(round((py - ymin) / dy)), 0), n - 1)
        c = (i + PAD) * W + (j + PAD) # flat index
        if z0 > Z[c]: # Z[c] starts at -inf
            Z[c] = z0 # sets cell to peak's exact height
            S[c] = sum(tables[pid]) / N_TABLE # start from the peak's own mean slope, not 0
            # S[c] is mean of slope table
            heap.append((-z0, c, pid))
    heapq.heapify(heap)

    # ---------- main loop (local names for speed) ----------
    heappop, heappush, bisect_left = heapq.heappop, heapq.heappush, bisect.bisect_left
    one_minus_rho = 1.0 - rho

    while heap:
        neg_h, c, pid = heappop(heap) # c = flat index, pid = unique idx per peak
        h = -neg_h
        if h < Z[c]:
            continue                              # stale entry
        tab = tables[pid]
        tidx = theta_idx[pid]
        prev = S[c] + 1e-8                        # slope that produced this cell's height
        prev_pow = prev ** rho
        """
        slope = prev^rho * target^(1-rho) * noise
        = exp( rho*log(prev) + (1-rho)*log(target) ) * noise
        """
        k = bisect_left(sorted_heights, h) if protect_peaks else 0 # num peaks lower than h
        if k:
            nd, nz = near_d[k], near_z[k]
        c_edge = c * M

        for off, step, m in moves: # step = dist
            nc = c + off
            e = c_edge + m
            target = tab[tidx[nc]] * mult[nc] + 1e-8 # tab[...] = base slope
            slope = prev_pow * target ** one_minus_rho * slope_noise[e]
            drop = step * edge_factor[e] # horizontal step, with noise

            eff = slope
            if k:
                # slope that would bring the terrain down to (summit - margin) at the
                # closest lower peak, if we kept descending through this neighbour
                s_req = (h - nz[nc] + overshadow_margin) / (drop + nd[nc])
                # L = drop + nd[nc]: drop = dist(popped peak, candidate), nd[nc] = dist(candidate, lower peak)
                # comes from: height_higher_peak - L*s = height_lower_peak
                if s_req > slope: # would bury
                    eff = min(s_req, max_boost * slope)

            h_new = h - eff * drop
            if h_new > floor and h_new > Z[nc]:
                Z[nc] = h_new
                S[nc] = slope                     # store the unboosted slope, so the boost
                heappush(heap, (-h_new, nc, pid)) # does not propagate through inheritance

    # ---------- back to numpy ----------
    Zout = np.array(Z).reshape(W, W)[PAD:-PAD, PAD:-PAD].copy()
    Zout[np.isinf(Zout)] = floor                  # cells no peak reached before the floor

    if final_noise_std > 0:
        Zout = np.maximum(Zout + rng.normal(0.0, final_noise_std, size=Zout.shape), floor)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if return_slopes:
        slopes = np.array(S).reshape(W, W)[PAD:-PAD, PAD:-PAD].copy()
        return X, Y, Zout, slopes
    return X, Y, Zout


def show_terrain(X, Y, Z):
    """Display the complete terrain using PyVista."""

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
        date_label.SetText(2, str(time_values[t]))

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
    if len(frame) == 0:
        # dummy invisible point, because glyphing an empty cloud fails
        frame = np.array([[0.0, 0.0, -100.0, 1e-3]])

    pts = pv.PolyData(frame[:, :3])
    pts["size"] = frame[:, 3]

    trunks = pts.glyph(geom=trunk_geom, scale="size", factor=1.0, orient=False)
    canopies = pts.glyph(geom=canopy_geom, scale="size", factor=1.0, orient=False)
    return trunks, canopies


def show_terrain_time_trees(
    GX, GY, GZ, time_values,
    trees=None,
    background_image=None,
    cloud_values=None,          # shape (nx, ny, nt); a 2D (nx, ny) array is also accepted
    dots=None                  # list of length nt; each entry is an (m, 2) array of stock x, y in GX/GY coordinates
):
    # --------------------------------------------------
    # TERRAIN COLOUR MAP
    # --------------------------------------------------

    zmax = float(np.nanmax(GZ))
    if zmax <= 0:
        raise ValueError("GZ must contain positive terrain heights.")


    sea = 0.5
    s = min(sea / zmax, 1.0)

    def land(f):
        return s + f * (1 - s)

    stops = [
        (0.0, "#1a5ab3"),
        (s, "#1a5ab3"),
        (land(0.00), "#8ccc66"),
        (land(0.30), "#2f8a3a"),
        (land(0.50), "#d9a441"),
        (land(0.65), "#c8662b"),
        (land(0.80), "#7d7d7d"),
        (1.0, "#e6e6e6"),
    ]

    terrain_cmap = LinearSegmentedColormap.from_list(
        "green_mountain", stops, N=512
    )

    n_times = GZ.shape[2]

    if len(time_values) != n_times:
        raise ValueError("time_values must match the time dimension of GZ.")

    if trees is not None and len(trees) != n_times:
        raise ValueError("trees must contain one entry per time step.")

    if GX.shape != GY.shape or GX.shape != GZ.shape[:2]:
        raise ValueError("GX and GY must match the first two dimensions of GZ.")

    # --------------------------------------------------
    # INITIAL TERRAIN
    # --------------------------------------------------

    Z = GZ[:, :, 0]

    grid = pv.StructuredGrid(GX, GY, Z)
    grid["height"] = Z.flatten(order="F")

    x_flat = GX.flatten(order="F")
    y_flat = GY.flatten(order="F")

    plotter = pv.Plotter()

    if background_image is not None:
        background_texture = pv.read_texture(background_image)
        plotter.add_actor(background_texture.to_skybox())
        plotter.set_environment_texture(
            background_texture, is_srgb=True
        )

    plotter.add_mesh(
        grid,
        scalars="height",
        cmap=terrain_cmap,
        clim=(0, zmax),
        smooth_shading=True,
        show_edges=False,
    )

    # --------------------------------------------------
    # CLOUDS
    # --------------------------------------------------

    CLOUD_ALTITUDE = zmax + 2.0   # fixed z for every cloud
    BETA_THRESHOLD = 0.15          # after normalization
    MIN_RADIUS = 1.0              # radius of the weakest cloud
    MAX_RADIUS = 3.0              # radius of the strongest cloud
    NEIGHBOURHOOD = 7             # cells; a cell must be the max within this window

    cloud_norm = None
    if cloud_values is not None:
        cloud_norm = np.abs(
            np.nan_to_num(cloud_values, nan=0.0, posinf=0.0, neginf=0.0)
        )

        # accept a static 2D field too: repeat it for every time step
        if cloud_norm.ndim == 2:
            cloud_norm = np.repeat(cloud_norm[:, :, None], n_times, axis=2)
        if cloud_norm.shape != GZ.shape:
            raise ValueError(
                "cloud_values must have shape (nx, ny) or (nx, ny, nt)."
            )

        # normalise by the global max so sizes are comparable across time
        gmax = cloud_norm.max()
        if gmax > 0:
            cloud_norm = cloud_norm / gmax

    cloud_template = pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16)

    def draw_clouds(t):
        if cloud_norm is None:
            return

        frame = cloud_norm[:, :, t]

        # a cell is a cloud centre if it is the max of its neighbourhood
        local_max = frame == maximum_filter(
            frame, size=NEIGHBOURHOOD, mode="nearest"
        )
        mask = local_max & (frame > BETA_THRESHOLD)

        if not mask.any():
            # no clouds at this time step: remove the previous ones
            plotter.remove_actor("clouds", render=False)
            return

        betas = frame[mask]

        poly = pv.PolyData(np.column_stack([
            GX[mask],
            GY[mask],
            np.full(betas.shape, CLOUD_ALTITUDE),
        ]))

        # radius grows linearly with beta
        poly["radius"] = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (
            (betas - BETA_THRESHOLD) / (1.0 - BETA_THRESHOLD)
        )

        glyphs = poly.glyph(
            scale="radius",
            factor=1.0,
            orient=False,
            geom=cloud_template,
        )

        # same name -> replaces the clouds of the previous frame
        plotter.add_mesh(
            glyphs,
            name="clouds",
            color="lightgray",
            opacity=0.5,
            smooth_shading=True,
            render=False,
        )

    # --------------------------------------------------
    # STOCK DOTS
    # --------------------------------------------------

    x_axis = GX[:, 0]          # assumes GX varies along axis 0 (meshgrid indexing="ij")
    y_axis = GY[0, :]
    DOT_LIFT = 0.05
    DOT_MIN_Z = sea            # only show stocks standing on land (above sea level)

    def nearest_index(axis, v):
        i = np.clip(np.searchsorted(axis, v), 1, len(axis) - 1)
        left_closer = np.abs(v - axis[i - 1]) < np.abs(v - axis[i])
        return np.where(left_closer, i - 1, i)

    def draw_dots(t):
        if dots is None:
            return

        xy = np.asarray(dots[t], dtype=float).reshape(-1, 2)
        if len(xy) == 0:
            plotter.remove_actor("dots", render=False)
            return

        ix = nearest_index(x_axis, xy[:, 0])
        iy = nearest_index(y_axis, xy[:, 1])
        z = GZ[ix, iy, t]                      # real terrain height under each stock

        on_land = z > DOT_MIN_Z
        if not on_land.any():
            plotter.remove_actor("dots", render=False)
            return

        pts = np.column_stack([xy[on_land, 0], xy[on_land, 1], z[on_land] + DOT_LIFT])

        # same name -> replaces the dots of the previous frame
        plotter.add_mesh(
            pv.PolyData(pts),
            name="dots",
            color="red",
            point_size=10,
            render_points_as_spheres=True,
            lighting=False,
            render=False,
        )

    # --------------------------------------------------
    # TREE TEMPLATES
    # --------------------------------------------------

    global trunk_geom, canopy_geom

    trunk_geom = pv.Cylinder(
        center=(0, 0, 0.15),
        direction=(0, 0, 1),
        radius=0.04,
        height=0.3,
        resolution=8,
    )

    canopy_geom = pv.Cone(
        center=(0, 0, 0.65),
        direction=(0, 0, 1),
        height=0.7,
        radius=0.22,
        resolution=10,
    )

    def draw_trees(t):

        if trees is None:
            return

        trunks, canopies = make_trees(trees[t])

        plotter.add_mesh(
            trunks,
            color="saddlebrown",
            name="trunks",
            render=False
        )

        plotter.add_mesh(
            canopies,
            color="forestgreen",
            name="canopies",
            smooth_shading=True,
            render=False
        )

    draw_trees(0)
    draw_clouds(0)
    draw_dots(0)

    # --------------------------------------------------
    # TIME LABEL
    # --------------------------------------------------

    date_label = plotter.add_text(
        str(time_values[0]),
        position="upper_left",
        font_size=12
    )

    # --------------------------------------------------
    # TIME UPDATE CALLBACK
    # --------------------------------------------------

    def update_terrain(value):

        t = int(round(value))
        t = max(0, min(t, n_times - 1))

        Z_new = GZ[:, :, t]

        # Update terrain geometry
        grid.points = np.column_stack([
            x_flat,
            y_flat,
            Z_new.flatten(order="F")
        ])

        grid["height"] = Z_new.flatten(order="F")

        # Update trees and clouds
        draw_trees(t)
        draw_clouds(t)
        draw_dots(t)

        # Update upper-left date label
        date_label.SetText(2, str(time_values[t]))

        plotter.render()

    # --------------------------------------------------
    # SLIDER
    # --------------------------------------------------

    plotter.add_slider_widget(
        update_terrain,
        rng=(0, n_times - 1),
        value=0,
        title="Time",
        interaction_event="always",
    )

    plotter.add_axes()
    plotter.show_bounds(grid="back")

    plotter.show()
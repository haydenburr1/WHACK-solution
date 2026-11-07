"""
Peak-based terrain grown with a priority queue (highest cell first).

All peaks sit in one max-heap. The highest cell is popped repeatedly and offers
each neighbour a lower height:

    target = slope_table_peak[angle] * smooth_field[neighbour]
    slope  = prev_slope^rho * target^(1-rho) * exp(sigma * N(0,1))     (inheritance)
    h_new  = h - slope * step_length * (1 + noise)

PEAK PROTECTION: for each edge we look up the closest peak (to the neighbour)
that is LOWER than the current cell's height h. If continuing at the current
slope would still be above that peak's summit by the time we reach it, the
slope is increased so the terrain drops to (summit - margin) at that peak. This
keeps lower peaks from being swallowed by a taller neighbour.

Everything that does not depend on the growth order is precomputed with numpy
before the loop; the loop only touches plain Python lists and floats.

Install:
    pip install pyvista numpy

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/peak_protec.py
"""

import bisect
import heapq
import math

import numpy as np

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
    print('mult:', len(mult))

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


def show_terrain(X, Y, Z, peaks, heights, marker_extra=2.0):
    """Show the terrain with a red vertical line and sphere at each peak."""
    import pyvista as pv

    grid = pv.StructuredGrid(X, Y, Z)
    grid["height"] = Z.flatten(order="F")  # StructuredGrid also uses Fortran order

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="height", cmap="terrain", smooth_shading=True)

    for (px, py), z0 in zip(peaks, heights):
        line = pv.Line((px, py, z0), (px, py, z0 + marker_extra))
        plotter.add_mesh(line, color="red", line_width=5, render_lines_as_tubes=True)
        plotter.add_mesh(pv.Sphere(radius=0.15, center=(px, py, z0)), color="red")

    plotter.add_axes()
    plotter.show_bounds(grid="back")
    plotter.show()


if __name__ == "__main__":
    import time

    # One dominant peak with small satellites
    peak_xy = [(0, 0), (-6, 5), (6, 5), (0, -7), (7, -4)]
    peak_z = [9.0, 4.0, 3.5, 4.5, 3.0]

    t0 = time.perf_counter()
    X, Y, Z = build_terrain(peak_xy, peak_z)
    print(f"terrain built in {time.perf_counter() - t0:.2f} s")
    show_terrain(X, Y, Z, peak_xy, peak_z)
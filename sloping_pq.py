"""
Peak-based terrain grown with a priority queue (Dijkstra-style, highest first).

All peaks are put in one max-heap. The highest cell is popped repeatedly and
offers each neighbour a lower height:

    h_new = h - slope(theta) * step_length * (1 + noise)

A neighbour keeps the best (highest) offer it receives, so overlapping peaks
merge automatically. theta is the angle of the neighbour as seen from the peak
the growth descends from, so each peak keeps its own direction-dependent slope.

Install:
    pip install pyvista numpy

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/sloping_pq.py
"""

import heapq
import math

import numpy as np
import pyvista as pv

def make_slope_fn(rng, base_slope, anisotropy, n_waves=3, n_table=200, min_slope=0.02):
    phases = rng.uniform(0, 2 * np.pi, size=n_waves)
    amps = rng.uniform(0.5, 1, size=n_waves)
    freqs = np.arange(1, n_waves + 1)            # integer frequencies keep it periodic
    vals = np.linspace(0, 2 * np.pi, n_table, endpoint=False)

    x = sum(amps[k] * np.cos(freqs[k] * vals + phases[k]) for k in range(n_waves))
    x = x / np.max(np.abs(x)) * 5                    # scale to [-1, 1]
    table = np.maximum(base_slope + anisotropy * x, min_slope).tolist()

    # def slope_fn(theta, ref_h, max_h): # ref_h should be <= max_height
    #     gamma = 1.0
    #     idx = int((theta % math.tau) / math.tau * n_table) % n_table
    #     return table[idx] * np.exp(-gamma*(max_h/ref_h - 1))

    def slope_fn(theta): # ref_h should be <= max_height
        idx = int((theta % math.tau) / math.tau * n_table) % n_table
        return table[idx]

    return slope_fn

def smooth_noise(rng, n, scale):          # scale = correlation length in cells
    f2 = np.fft.fftfreq(n)[:, None]**2 + np.fft.fftfreq(n)[None, :]**2
    spec = np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-2 * np.pi**2 * scale**2 * f2)
    field = np.fft.ifft2(spec).real
    return field / field.std()


# 16-neighbourhood: 8 adjacent cells plus the 8 "knight moves". More directions
# than 4 or 8 makes the grown peaks much rounder (fewer diamond/octagon shapes).
OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
]


def build_terrain(
    peaks,                  # list of (x, y) positions
    heights,                # list of z values, one per peak
    slope_mean=1.5,         # mean of the base slope across peaks
    slope_std=0.3,         # std of the base slope across peaks
    anisotropy=0.20,        # how strongly the slope varies with direction
    n_harmonics=4,          # how wiggly that variation is
    edge_noise=0.15,        # relative noise on each step: (1 + N(0, edge_noise))
    final_noise_std=0.0,    # optional plain white noise added at the end
    extent=(-10, 10, -10, 10),
    resolution=200,
    floor=0.0,
    seed=0,
):
    MAX_h = max(heights)
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, dtype=float)
    heights = np.asarray(heights, dtype=float)
    field = smooth_noise(rng, resolution, 15) + 0.5 * smooth_noise(rng, resolution, 5)

    base_slopes = np.abs(rng.normal(slope_mean, slope_std, size=len(peaks)))
    slope_fns = [
        make_slope_fn(rng, bs, anisotropy, n_harmonics) for bs in base_slopes
    ]

    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    Z = np.full((resolution, resolution), -np.inf)   # -inf = not reached yet
    heap = []                                        # entries: (-height, i, j, peak_id)

    # Seed the queue with every peak (heapq is a min-heap, so heights are negated)
    for pid, ((px, py), z0) in enumerate(zip(peaks, heights)):
        i = int(round((px - xmin) / dx))
        j = int(round((py - ymin) / dy))
        i = min(max(i, 0), resolution - 1)
        j = min(max(j, 0), resolution - 1)
        if z0 > Z[i, j]:
            Z[i, j] = z0
            heapq.heappush(heap, (-z0, i, j, pid))

    slopes_struct = np.zeros((resolution, resolution), dtype=float)

    while heap:
        #print(len(heap))
        neg_h, i, j, pid = heapq.heappop(heap)
        h = -neg_h
        if h < Z[i, j]:
            continue                 # stale entry: a better offer arrived after this push

        px, py = peaks[pid]
        slope_fn = slope_fns[pid]

        for di, dj in OFFSETS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < resolution and 0 <= nj < resolution):
                continue             # outside the grid
            # get closest peak that is lower than this point

            step = math.hypot(di * dx, dj * dy)                  # horizontal step length
            theta = math.atan2(ys[nj] - py, xs[ni] - px)         # neighbour as seen from the peak
            factor = max(1.0 + rng.normal(0.0, edge_noise), 0.1) # keep the drop positive
            prev_slope = slopes_struct[i,j]
            #target = slope_fn(theta, h, MAX_h)
            target = slope_fn(theta) * np.exp(0.3 * field[ni, nj])   # exp keeps it positive
            rho = 0.5
            sigma = 0.10
            slope = np.exp(rho * np.log(prev_slope + 1e-8) + (1 - rho) * np.log(target + 1e-8) + sigma * rng.normal())
            h_new = h - slope * step * factor

            if h_new > floor and h_new > Z[ni, nj]:
                Z[ni, nj] = h_new
                slopes_struct[ni, nj] = slope
                heapq.heappush(heap, (-h_new, ni, nj, pid))

    Z[np.isinf(Z)] = floor           # cells no peak reached before the floor

    if final_noise_std > 0:
        Z = np.maximum(Z + rng.normal(0.0, final_noise_std, size=Z.shape), floor)

    return X, Y, Z


def show_terrain(X, Y, Z, peaks, heights, marker_extra=2.0):
    """Show the terrain with a red vertical line and sphere at each peak."""
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
    
    # # 1. Single peak: shows the slope pattern with no overlap
    # peak_xy = [(0, 0)]
    # peak_z  = [8.0]

    # # 2. Twin peaks, equal height: the front where they meet is easy to see
    # peak_xy = [(-4, 0), (4, 0)]
    # peak_z  = [7.0, 7.0]

    # 3. One dominant peak with small satellites
    peak_xy = [(0, 0), (-6, 5), (6, 5), (0, -7), (7, -4)]
    peak_z  = [9.0, 4.0, 3.5, 4.5, 3.0]

    # # 4. Ridge: peaks in a line with varying heights
    # peak_xy = [(-8, -6), (-4, -3), (0, 0), (4, 3), (8, 6)]
    # peak_z  = [5.0, 7.0, 8.5, 7.5, 5.5]

    # # 5. Tight cluster: heavy overlap, merging happens high up
    # peak_xy = [(-1, -1), (1, 0), (0, 2), (-2, 1.5)]
    # peak_z  = [8.0, 7.0, 7.5, 6.0]

    # # 6. Scattered range: eight peaks of very different heights
    # peak_xy = [(-8, 7), (-3, 2), (2, 8), (7, 4), (-6, -6), (0, -3), (5, -7), (8, -1)]
    # peak_z  = [4.0, 6.5, 5.0, 7.5, 8.5, 3.0, 6.0, 4.5]

    # # 7. Ring of equal peaks around a lower centre peak (tests a valley between them)
    # peak_xy = [(6, 0), (3, 5.2), (-3, 5.2), (-6, 0), (-3, -5.2), (3, -5.2), (0, 0)]
    # peak_z  = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0]

    # # 8. Peaks near the edge: tests the grid boundary
    # peak_xy = [(-9, -9), (9, 9), (-9, 9), (9, -9), (0, 0)]
    # peak_z  = [6.0, 7.0, 5.0, 8.0, 4.0]


    X, Y, Z = build_terrain(peak_xy, peak_z)
    show_terrain(X, Y, Z, peak_xy, peak_z)
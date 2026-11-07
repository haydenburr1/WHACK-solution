"""
Level-by-level terrain growth.

We walk z from the highest peak down to the floor. At each level every peak
that has already "started" (z <= its height) has a region around it, described
in polar form by a radius r(theta). Going one level down, the radius grows, so
each region contains the previous one (the plateau we already have). A cell
gets the height of the first level at which any peak's region reaches it, so
overlapping peaks only merge below both peaks' heights.

Radius of a peak at level z:
    standard_dist = (z0 - z) / slope          (grows as z decreases)
    r(theta) = standard_dist + roughness * shape(theta)
where shape(theta) is a random periodic function in [0, 1], drawn ONCE per peak.

Install:
    pip install pyvista numpy

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/sloping_waves.py
 
"""

import numpy as np
import pyvista as pv

TAU = 2 * np.pi


def make_shape(rng, num_waves=5, num_points=200):
    """
    Random periodic function of the angle, as a lookup table over [0, 2*pi),
    normalised to [0, 1]. Drawn once per peak, so the outline keeps the same
    character at every level.
    """
    phases = rng.uniform(0, TAU, size=num_waves)
    amps = rng.uniform(0.1, 0.2, size=num_waves)
    freqs = np.arange(1, num_waves + 1)          # integer frequencies -> periodic
    x = np.linspace(0, TAU, num_points, endpoint=False)
    y = sum(amps[k] * np.cos(freqs[k] * x + phases[k]) for k in range(num_waves))
    if np.min(y) <= 0:
        return np.abs(np.min(y)) + 1 + y
    if np.min(y) > 1:
        return y
    return 1 + y


def waves(shape, standard_dist, roughness):
    """Radius table for one peak at one level (minimum value = standard_dist)."""
    return standard_dist + roughness * shape


def build_terrain(
    peaks,                 # list of (x, y) positions
    heights,               # list of z values, one per peak
    slope=0.5,             # height lost per unit of horizontal growth
    roughness=1.0,         # how irregular the outlines are (world units)
    dz=0.1,                # height step between levels (smaller = smoother)
    num_waves=5,
    extent=(-10, 10, -10, 10),
    resolution=200,
    floor=0.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, dtype=float)
    heights = np.asarray(heights, dtype=float)
    n_table = 200

    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    shapes = [make_shape(rng, num_waves, n_table) for _ in peaks]

    # Distance and angle-table index of every grid point, per peak (computed once)
    dists, idxs = [], []
    for px, py in peaks:
        dx, dy = X - px, Y - py
        dists.append(np.hypot(dx, dy))
        theta = np.arctan2(dy, dx) % TAU
        idxs.append((theta / TAU * n_table).astype(int) % n_table)

    # Levels from the top down; each peak's own height is included so that
    # it starts exactly at its summit.
    grid_levels = heights.max() - dz * np.arange(int((heights.max() - floor) / dz) + 1)
    levels = np.unique(np.concatenate([grid_levels, heights]))[::-1]
    levels = levels[levels > floor]

    Z = np.full(X.shape, np.nan)                  # nan = not reached yet
    for z in levels:
        for p in range(len(peaks)):
            if heights[p] < z:
                continue                          # this peak hasn't started yet
            depth = heights[p] - z
            radius = waves(shapes[p], depth / slope, roughness)
            inside = dists[p] <= radius[idxs[p]]  # region of peak p at level z
            Z[inside & np.isnan(Z)] = z           # first (highest) level wins

    Z[np.isnan(Z)] = floor
    return X, Y, Z


def show_terrain(X, Y, Z, peaks, heights, marker_extra=2.0):
    """Show the terrain with a red vertical line and sphere at each peak."""
    grid = pv.StructuredGrid(X, Y, Z)
    grid["height"] = Z.flatten(order="F")

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
    peak_xy = [(-5, -4), (0, 3), (6, -2), (4, 6)]
    peak_z = [8.0, 5.0, 6.5, 8.0]

    X, Y, Z = build_terrain(peak_xy, peak_z)
    show_terrain(X, Y, Z, peak_xy, peak_z)
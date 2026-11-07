"""
Peak-based terrain: each (x, y, z) peak gets a random slope drawn from a
normal distribution, and the surface falls away from the peak at that
slope, plus noise.

Install:
    pip install pyvista numpy

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/sloping_out_test.py
"""

import numpy as np
import pyvista as pv


def build_terrain(
    peaks,                 # list of (x, y) positions
    heights,               # list of z values, one per peak
    slope_mean=0.5,        # mean slope (height lost per unit horizontal distance)
    slope_std=0.15,        # standard deviation of the slope
    noise_std=0.05,        # standard deviation of the added height noise
    extent=(-10, 10, -10, 10),
    resolution=200,
    floor=0.0,             # terrain never drops below this height
    seed=0,
):
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, dtype=float)
    heights = np.asarray(heights, dtype=float)

    # One slope per peak, from a normal distribution (kept strictly positive)
    slopes = np.abs(rng.normal(slope_mean, slope_std, size=len(peaks)))
    slopes = np.maximum(slopes, 1e-3)
    print("Sampled slopes:", np.round(slopes, 3))

    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    """
    X and Y: values from xs and ys resp. are repeated down every column
    """

    # Each peak is a cone: z_i - slope_i * distance_to_peak_i
    # Where cones overlap, the highest one wins.
    Z = np.full_like(X, floor)
    for (px, py), z0, s in zip(peaks, heights, slopes):
        dist = np.hypot(X - px, Y - py) # computed for every point in X at once, is same shape as X and Y
        # dist[i,j] is distance between (xs[i], ys[j]) to peak (px, py)
        Z = np.maximum(Z, z0 - s * dist)

    # Add noise, then keep the floor
    Z = Z + rng.normal(0.0, noise_std, size=Z.shape)
    Z = np.maximum(Z, floor)

    return X, Y, Z


def show_terrain(X, Y, Z):
    grid = pv.StructuredGrid(X, Y, Z)
    grid["height"] = Z.flatten(order="F")  # StructuredGrid also uses Fortran order

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="height", cmap="terrain", smooth_shading=True)
    plotter.add_axes()
    plotter.show_bounds(grid="back")
    plotter.show()


if __name__ == "__main__":
    peak_xy = [(-5, -4), (0, 3), (6, -2), (4, 6)]
    peak_z = [8.0, 5.0, 6.5, 4.0]

    X, Y, Z = build_terrain(peak_xy, peak_z)
    show_terrain(X, Y, Z)
"""
Peak-based terrain with direction-dependent slopes.

Each peak gets its own slope function s(theta), where theta is the angle of
a grid point as seen from the peak. The height at a point is
    z0 - s(theta) * distance
and overlapping peaks are merged with a maximum.

Install:
    pip install pyvista numpy

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/sloping_out_2.py
"""

import numpy as np
import pyvista as pv


def make_slope_fn(rng, base_slope, anisotropy, n_harmonics, min_slope=0.02):
    """
    Returns a random function s(theta) that is periodic in theta, so there is
    no seam where the angle wraps from 2*pi back to 0.

    s(theta) = base + sum_k [a_k cos(k theta) + b_k sin(k theta)]
    with a_k, b_k ~ Normal(0, anisotropy / k): higher harmonics get smaller
    amplitudes, so the variation is smooth rather than jagged.
    """
    ks = np.arange(1, n_harmonics + 1)
    a = rng.normal(0.0, anisotropy / ks)
    b = rng.normal(0.0, anisotropy / ks)

    def slope_fn(theta):
        s = np.full_like(theta, base_slope)
        for k, ak, bk in zip(ks, a, b):
            s += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        return np.maximum(s, min_slope)

    return slope_fn


def build_terrain(
    peaks,                 # list of (x, y) positions
    heights,               # list of z values, one per peak
    slope_mean=0.5,        # mean of the base slope across peaks
    slope_std=0.15,        # std of the base slope across peaks
    anisotropy=0.35,       # how strongly the slope varies with direction
    n_harmonics=2,         # how wiggly that variation is
    noise_std=0.10,
    extent=(-10, 10, -10, 10),
    resolution=200,
    floor=0.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, dtype=float)
    heights = np.asarray(heights, dtype=float)

    base_slopes = np.abs(rng.normal(slope_mean, slope_std, size=len(peaks)))
    slope_fns = [
        make_slope_fn(rng, bs, anisotropy, n_harmonics) for bs in base_slopes
    ]

    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    Z = np.full_like(X, floor)
    for (px, py), z0, slope_fn in zip(peaks, heights, slope_fns):
        dx = X - px
        dy = Y - py
        dist = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)      # angle of each grid point seen from the peak
        s = slope_fn(theta)             # one slope per grid point, same shape as X
        Z = np.maximum(Z, z0 - s * dist)

    Z = Z + rng.normal(0.0, noise_std, size=Z.shape)
    Z = np.maximum(Z, floor)
    return X, Y, Z


def show_terrain(X, Y, Z):
    grid = pv.StructuredGrid(X, Y, Z)
    grid["height"] = Z.flatten(order="F")

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
"""
Ocean with six islands arranged in a 2 x 3 grid (starting point).

The map is split into 2 rows x 3 columns of equal cells, and each island gets
the centre of one cell (numbered 1-6, left to right, top to bottom).
The terrain of the islands is NOT filled in yet: island_height() is a stub, so
for now you only see the seabed, the water surface, and a ring + number
marking where each island will go.

./venv/bin/python /home/tobiasocula/Documents/whack/WHACK-solution/ocean_islands.py

Install:
    pip install pyvista numpy
"""

import numpy as np


def island_centers(extent, rows=2, cols=3):
    """Centre of every cell of a rows x cols partition of the map.

    Row 0 is at the top (largest y), so the numbering reads like text.
    Returns the centres and the (width, height) of one cell.
    """
    xmin, xmax, ymin, ymax = extent
    cell_w = (xmax - xmin) / cols
    cell_h = (ymax - ymin) / rows
    centers = [
        (xmin + (c + 0.5) * cell_w, ymax - (r + 0.5) * cell_h)
        for r in range(rows)
        for c in range(cols)
    ]
    return centers, (cell_w, cell_h)


def island_height(X, Y, center, radius, seabed_depth, rng):
    """
    TODO: terrain of ONE island.

    X, Y         : coordinate arrays of the whole map
    center       : (x, y) centre of this island
    radius       : space available to the island around its centre
    seabed_depth : the seabed lies at z = -seabed_depth, sea level is z = 0

    Return an array shaped like X with ABSOLUTE terrain heights of this island,
    including its underwater flanks (down to the seabed). Where the island has
    no influence, return -seabed_depth (or lower). It is combined with the other
    islands by a pointwise maximum, so the highest terrain always wins.

    The stub returns the bare seabed, i.e. no island at all.
    """
    return np.full_like(X, -seabed_depth)


def build_heightmap(
    extent=(-15, 15, -10, 10),
    cells_per_unit=10,        # grid resolution: sample points per world unit
    rows=2,
    cols=3,
    island_fill=0.8,          # fraction of a grid cell an island may occupy (as a diameter)
    seabed_depth=3.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = extent

    nx = int(round((xmax - xmin) * cells_per_unit)) + 1
    ny = int(round((ymax - ymin) * cells_per_unit)) + 1
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    centers, (cell_w, cell_h) = island_centers(extent, rows, cols)
    radius = island_fill * min(cell_w, cell_h) / 2

    Z = np.full(X.shape, -seabed_depth)               # flat seabed
    for center in centers:
        Z = np.maximum(Z, island_height(X, Y, center, radius, seabed_depth, rng))

    return X, Y, Z, centers, radius


def show_map(X, Y, Z, centers, radius, seabed_depth=3.0, max_height=5.0, top_down=True):
    """Seabed/terrain + translucent water surface + a numbered ring per island."""
    import pyvista as pv

    sea_level = 0.0
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    grid = pv.StructuredGrid(X, Y, Z)
    grid["height"] = Z.flatten(order="F")             # StructuredGrid uses Fortran order

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid, scalars="height", cmap="terrain",
        clim=(-seabed_depth, max_height), smooth_shading=True,
        scalar_bar_args={"title": "height"},
    )

    water = pv.Plane(
        center=((xmin + xmax) / 2, (ymin + ymax) / 2, sea_level),
        direction=(0, 0, 1),
        i_size=xmax - xmin, j_size=ymax - ymin,
    )
    plotter.add_mesh(water, color="royalblue", opacity=0.5)

    # Placeholder markers: a thin ring of the island's footprint and its number
    for (cx, cy) in centers:
        ring = pv.Disc(
            center=(cx, cy, sea_level + 0.02), inner=radius * 0.97, outer=radius,
            normal=(0, 0, 1), r_res=1, c_res=72,
        )
        plotter.add_mesh(ring, color="white")

    label_points = np.array([(cx, cy, sea_level + 0.3) for cx, cy in centers])
    labels = [str(i) for i in range(1, len(centers) + 1)]
    plotter.add_point_labels(label_points, labels, font_size=24,
                             show_points=False, shape=None)

    plotter.add_axes()
    if top_down:
        plotter.view_xy()                             # map view; you can still rotate
    plotter.show()


if __name__ == "__main__":
    SEABED_DEPTH = 3.0

    X, Y, Z, centers, radius = build_heightmap(seabed_depth=SEABED_DEPTH)
    print("island centres:", [(round(x, 2), round(y, 2)) for x, y in centers])
    print(f"grid: {Z.shape}, island footprint radius: {radius:.2f}")

    show_map(X, Y, Z, centers, radius, seabed_depth=SEABED_DEPTH)
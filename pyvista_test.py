"""
Sanity check for pyvisa + pyvista.

Install:
    pip install pyvisa pyvisa-py pyvista numpy

pyvisa-py is a pure-Python backend, so this works even without NI-VISA
installed. No instrument is required: the script just lists whatever
VISA resources it can find (an empty list is fine).

./venv/bin/python /home/tobiasocula/Documents/whack/pyvista_test.py
"""

import numpy as np
import pyvisa
import pyvista as pv


def check_pyvisa():
    print(f"pyvisa version: {pyvisa.__version__}")
    try:
        rm = pyvisa.ResourceManager()      # NI-VISA or other system backend
    except Exception as err:
        print(f"Default backend not available ({err}); trying pyvisa-py (@py)")
        rm = pyvisa.ResourceManager("@py")

    print(f"Backend: {rm}")
    resources = rm.list_resources()
    print(f"Resources found: {resources if resources else 'none (that is OK)'}")
    rm.close()


def show_noise_grid(n=40, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.random((n, n, n))

    # ImageData was called UniformGrid in older pyvista versions
    GridClass = getattr(pv, "ImageData", None) # or pv.UniformGrid
    grid = GridClass(dimensions=(n, n, n))
    print('shape:', noise.flatten(order="F").shape) # (64000) (flat)
    grid.point_data["noise"] = noise.flatten(order="F")  # VTK uses Fortran order

    """
    point_data is a dictionary-like container of arrays living on the grid’s points.
    Assigning to a new key adds an array named "noise". PyVista checks that the array has exactly one
    value per point (here n³ = 64,000 values), and then you can refer to it by name, 
    as in add_volume(grid, scalars="noise").
    """

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="noise", cmap="viridis", opacity="sigmoid")
    plotter.add_axes()
    plotter.show_bounds(grid="back")
    plotter.show()


if __name__ == "__main__":
    check_pyvisa()
    show_noise_grid()
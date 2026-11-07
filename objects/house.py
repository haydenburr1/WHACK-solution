
import numpy as np
import pyvista as pv

from objects.base import local_transform


def create_house(position, normal, plotter=None):

    transform = local_transform(normal, position)

    house = pv.Cube(
        center=(0, 0, 0.16),
        x_length=0.38,
        y_length=0.32,
        z_length=0.32
    )

    house.transform(transform, inplace=True)

    roof_points = np.array([
        [-0.19, -0.16, 0.32],
        [ 0.19, -0.16, 0.32],
        [ 0.19,  0.16, 0.32],
        [-0.19,  0.16, 0.32],
        [ 0.00,  0.00, 0.55]
    ])

    roof_faces = np.array([
        3, 0, 1, 4,
        3, 1, 2, 4,
        3, 2, 3, 4,
        3, 3, 0, 4
    ])

    roof = pv.PolyData(roof_points, roof_faces)
    roof.transform(transform, inplace=True)

    if plotter is not None:
        plotter.add_mesh(house, color="#D9B58C")
        plotter.add_mesh(roof, color="#963F35")

    return house, roof

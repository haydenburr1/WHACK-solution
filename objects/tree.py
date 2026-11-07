
import pyvista as pv


def create_tree(
    position,
    normal,
    scale=1.0,
    plotter=None
):
    trunk_height = 0.4 * scale
    crown_height = 0.7 * scale

    trunk = pv.Cylinder(
        center=position + normal * trunk_height / 2,
        direction=normal,
        radius=0.055 * scale,
        height=trunk_height,
        resolution=8
    )

    crown = pv.Cone(
        center=position + normal * (
            trunk_height + crown_height / 2
        ),
        direction=normal,
        height=crown_height,
        radius=0.28 * scale,
        resolution=8
    )

    if plotter is not None:
        plotter.add_mesh(trunk, color="#765039")
        plotter.add_mesh(crown, color="#286B42")

    return trunk, crown

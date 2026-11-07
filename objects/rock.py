
import numpy as np
import pyvista as pv

from objects.base import local_transform


def create_rock(position, normal, size=0.2, plotter=None, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    rock = pv.Icosphere(
        radius=size,
        center=(0, 0, 0),
        nsub=1
    )

    rock.points *= np.array([
        rng.uniform(1.0, 1.5),
        rng.uniform(0.8, 1.2),
        rng.uniform(0.45, 0.7)
    ])

    rock.points *= rng.uniform(
        0.9, 1.1, size=(rock.n_points, 1)
    )

    transform = local_transform(
        normal,
        position + normal * size * 0.18
    )

    rock.transform(transform, inplace=True)

    if plotter is not None:
        plotter.add_mesh(
            rock,
            color="#777B83",
            smooth_shading=False
        )

    return rock


import numpy as np


def local_transform(normal, position):
    """Align an object's local Z axis with the planet surface."""

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    tangent = np.cross(normal, [0, 0, 1])

    if np.linalg.norm(tangent) < 0.001:
        tangent = np.cross(normal, [0, 1, 0])

    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    matrix = np.eye(4)
    matrix[:3, 0] = tangent
    matrix[:3, 1] = bitangent
    matrix[:3, 2] = normal
    matrix[:3, 3] = position

    return matrix


def random_direction(rng):
    """Generate a random point direction on a sphere."""

    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction)
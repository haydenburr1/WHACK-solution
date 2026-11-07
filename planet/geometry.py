"""Geometria: kockalapok (és ugyanaz gömbre vetítve), zaj, színátmenet, kellékek.

Egy 'lap' = egy szektor. A lap pontjait (u, v) in [-1, 1] koordináta adja:
    cube pont  c = n + u*a + v*b        (n = lap normálisa, a x b = n)
    irány      d = c / |c|              (egységvektor; a témák EZT kapják -> kocka és gömb is működik)
Lapsorrend megegyezik a repo cube.py-jával:
    0 Consumer Discretionary (+x)   1 Energy (-x)        2 Financials (+y)
    3 Health Care (-y)              4 Industrials (+z)   5 Information Technology (-z)
"""
import numpy as np
import pyvista as pv

TWO_PI = 2 * np.pi
X, Y, Z = np.eye(3)
FACE_NORMALS = np.array([X, -X, Y, -Y, Z, -Z])
FACE_AXES = [(Y, Z), (Z, Y), (Z, X), (X, Z), (X, Y), (Y, X)]     # (a, b), a x b = n


def face_cube_points(k, uv):
    a, b = FACE_AXES[k]
    return FACE_NORMALS[k] + uv[:, :1] * a + uv[:, 1:2] * b


def face_dirs(k, uv):
    c = face_cube_points(k, uv)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def face_grid(N):
    """N x N rács egy lapon: (uv (N*N,2), háromszögek (m,3))."""
    u = np.linspace(-1, 1, N)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    idx = np.arange(N * N).reshape(N, N)
    i00, i10 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    i11, i01 = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    tris = np.vstack([np.stack([i00, i10, i11], 1), np.stack([i00, i11, i01], 1)])
    return uv, tris


def edge_distance(uv):
    """Távolság a lap szélétől (0 a szélen, 1 a közepén)."""
    return 1 - np.maximum(np.abs(uv[:, 0]), np.abs(uv[:, 1]))


def angular_distance(p, c):
    return np.arccos(np.clip(p @ c, -1, 1))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def gradient(t, stops):
    """t: (n,) 0..1; stops: [(pozíció, (r,g,b)), ...] -> (n,3) szín (0..255)."""
    t = np.clip(t, 0, 1)
    pos = [s[0] for s in stops]
    return np.stack([np.interp(t, pos, [s[1][c] for s in stops]) for c in range(3)], axis=1)


def fbm(p, seed, octaves=5, freq=2.0, gain=0.5, ridged=False):
    """Függőségmentes zaj egységvektorokon (síkhullámok összege). Determinisztikus a seedre."""
    rng = np.random.default_rng(seed)
    total = np.zeros(len(p))
    amp, norm = 1.0, 0.0
    for _ in range(octaves):
        v = np.zeros(len(p))
        for _ in range(4):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            v += np.sin(freq * np.pi * (p @ d) + rng.uniform(0, TWO_PI))
        v = np.clip(v / 2.0, -1, 1)
        v = 1 - np.abs(v) if ridged else 0.5 + 0.5 * v
        total += amp * v
        norm += amp
        amp *= gain
        freq *= 2.0
    return total / norm


def stretch(x, k=2.2):
    return np.clip((x - 0.5) * k + 0.5, 0, 1)


def make_prop(kind, pos, normal, height, width, color, resolution=12):
    """Kis 3D kellék (épület, fa, szikla...) a felszínre állítva.
    kind: 'box' | 'cone' | 'cyl' | 'sphere' | 'rock'. Az alja a pos-nál van, a teteje a normal felé néz."""
    if kind == "box":
        m = pv.Cube(center=(0, 0, height / 2), x_length=width, y_length=width, z_length=height)
    elif kind == "cone":
        m = pv.Cone(center=(0, 0, height / 2), direction=(0, 0, 1), height=height,
                    radius=width / 2, resolution=resolution)
    elif kind == "cyl":
        m = pv.Cylinder(center=(0, 0, height / 2), direction=(0, 0, 1), radius=width / 2,
                        height=height, resolution=resolution)
    elif kind == "sphere":
        m = pv.Sphere(radius=width / 2, center=(0, 0, width / 2),
                      theta_resolution=12, phi_resolution=12)
    elif kind == "rock":
        m = pv.Icosphere(radius=width / 2, nsub=1)
        pts = np.array(m.points)
        rng = np.random.default_rng(int(abs(width) * 1e6) % (2**32))
        pts += rng.normal(scale=width * 0.10, size=pts.shape)
        pts[:, 2] = pts[:, 2] * 0.65 + width * 0.25
        m.points = pts
    else:
        raise ValueError(kind)

    m = m.triangulate()
    m.clear_data()

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(a, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = t1, t2, n, pos
    m = m.transform(M, inplace=False)
    m.point_data["rgb"] = np.tile(np.asarray(color, dtype=np.uint8), (m.n_points, 1))
    return m


def merge_props(meshes):
    if not meshes:
        return None
    return pv.merge(meshes, merge_points=False).extract_surface()

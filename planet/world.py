"""A bolygó: 6 lap, laponként téma, a magasság a részvények volatilitásából jön.

shape="cube"   -> kocka (mint a repo cube.py-ja)
shape="sphere" -> ugyanaz gömbre vetítve (cube-sphere)
Minden más (témák, adat, kellékek) azonos a kettőben.

Magasság egy pontban:  h = RELIEF * theme.relief * amp(pont) * theme.shape(irány) * szélmaszk
ahol amp(pont) = 0.08 + 0.92 * (a lap részvényeinek normált volatilitása, Gauss-súlyozva a
távolság szerint) -> minden részvény 'körül' ott a hatása; mode="sector"-nál a lap átlaga.
"""
import numpy as np
from geometry import (face_grid, face_cube_points, face_dirs, edge_distance, smoothstep,
                      merge_props, FACE_NORMALS)
from themes import THEME_CLASSES

RELIEF = 0.28
EDGE_BLEND = 0.10
SIGMA = 0.28        # egy részvény hatósugara a lapon (stock mód)
AMP_MIN = 0.08


def amp_from_norm(x):
    return AMP_MIN + (1 - AMP_MIN) * x


class World:
    def __init__(self, layout, shape="cube", N=80, seed=0, mode="stock"):
        """layout: data.build_layout(...) kimenete (laponként (oszlopindexek, helyek))."""
        assert shape in ("cube", "sphere") and mode in ("stock", "sector")
        self.shape_mode, self.mode = shape, mode
        self.themes = [cls(k) for k, cls in enumerate(THEME_CLASSES)]
        uv, tris = face_grid(N)
        nv = len(uv)
        rng = np.random.default_rng(seed)

        C, D, NRM, H, M, REL, RGB = [], [], [], [], [], [], []
        self.idx = [l[0] for l in layout]
        self.W, self.Wp, self.Wc = [], [], []
        self.P = []                       # kellékek tárolt adatai laponként
        for k, t in enumerate(self.themes):
            c = face_cube_points(k, uv)
            d = c / np.linalg.norm(c, axis=1, keepdims=True)
            h = t.shape(d)
            m = smoothstep(0, EDGE_BLEND, edge_distance(uv))
            rgb = t.colors(h, d) * (0.55 + 0.45 * m)[:, None]
            C.append(c); D.append(d); H.append(h); M.append(m); REL.append(np.full(nv, t.relief))
            NRM.append(np.tile(FACE_NORMALS[k], (nv, 1)) if shape == "cube" else d)
            RGB.append(rgb.clip(0, 255).astype(np.uint8))
            pos_k = layout[k][1]
            self.W.append(self._weights(uv, pos_k))
            # kellékek
            puv = t.scatter(rng)
            pc = face_cube_points(k, puv) if len(puv) else np.zeros((0, 3))
            pd_ = pc / np.linalg.norm(pc, axis=1, keepdims=True) if len(puv) else pc
            self.Wp.append(self._weights(puv, pos_k))
            cuv = np.zeros((1, 2))
            self.Wc.append(self._weights(cuv, pos_k))
            self.P.append(dict(
                c=pc, d=pd_, n=(np.tile(FACE_NORMALS[k], (len(puv), 1)) if shape == "cube" else pd_),
                S=t.shape(pd_) if len(puv) else np.zeros(0), R=rng.random(len(puv)),
                cc=face_cube_points(k, cuv), cd=face_dirs(k, cuv),
                cS=t.shape(face_dirs(k, cuv))))
        self.C, self.D, self.NRM = map(np.vstack, (C, D, NRM))
        self.H, self.M, self.REL = map(np.concatenate, (H, M, REL))
        self.rgb = np.vstack(RGB)
        self.nv = nv
        self.faces = np.vstack([tris + k * nv for k in range(6)])
        self.faces_flat = np.hstack([np.full((len(self.faces), 1), 3), self.faces]).ravel()

    def _weights(self, uv, pos):
        if self.mode == "sector" or len(uv) == 0:
            return np.full((len(uv), len(pos)), 1.0 / max(len(pos), 1))
        d2 = ((uv[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * SIGMA ** 2)) + 1e-12
        return w / w.sum(1, keepdims=True)

    # ------------------------------------------------------------ geometria
    def _place(self, c, d, n, h):
        if self.shape_mode == "cube":
            return c + n * h[:, None]
        return d * (1 + h)[:, None]

    def terrain_points(self, x):
        """x: (n_tickers,) normált volatilitás 0..1 -> (N_pont,3) pozíciók."""
        a = np.concatenate([amp_from_norm(self.W[k] @ x[self.idx[k]]) for k in range(6)])
        h = RELIEF * self.REL * a * self.H * self.M
        return self._place(self.C, self.D, self.NRM, h)

    def props(self, k, x):
        p, t = self.P[k], self.themes[k]
        if len(p["R"]) == 0:
            return None
        amp = amp_from_norm(self.Wp[k] @ x[self.idx[k]])
        h = RELIEF * t.relief * amp * p["S"] - 0.004
        pos = self._place(p["c"], p["d"], p["n"], h)
        camp = amp_from_norm((self.Wc[k] @ x[self.idx[k]])[0])
        ch = RELIEF * t.relief * camp * p["cS"] - 0.004
        n0 = FACE_NORMALS[k] if self.shape_mode == "cube" else p["cd"][0]
        cpos = self._place(p["cc"], p["cd"], n0[None], ch)[0]
        return merge_props(t.build_props(amp, pos, p["n"], p["R"], (cpos, n0, camp)))

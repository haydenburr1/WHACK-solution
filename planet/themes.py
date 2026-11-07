"""Szektor-témák: minden szektornak saját tája, színe és kellékei.

Új téma / módosítás: egy Theme-osztály három dolgot ad meg
  shape(d)                      -> 0..1 'alakzat' (d: egységvektorok), ezt skálázza a volatilitás
  colors(h, d)                  -> (n,3) RGB 0..255
  build_props(amp, pos, nrm, R, center) -> kellékek (épület, fa...) listája
      amp: kellékenként 0.08..1 'volatilitás-szint', pos/nrm: hely és felfelé irány,
      R: kellékenként stabil véletlen 0..1, center=(pos, nrm, amp) a lap közepén
A THEME_CLASSES sorrendje = lapsorrend (lásd geometry.py).
"""
import numpy as np
from geometry import (FACE_NORMALS, face_dirs, angular_distance, fbm, stretch, gradient, make_prop)


class Theme:
    sector = "?"          # pontosan a universe.csv 'sector' értéke
    short = "?"
    seed = 0
    n_props = 40
    relief = 1.0          # a szektor magasságának szorzója
    min_edge = 0.16       # kellékek ennél közelebb nem kerülnek a laphoz széléhez

    def __init__(self, k):
        self.k = k
        self.center = FACE_NORMALS[k].astype(float)

    def shape(self, d):
        raise NotImplementedError

    def colors(self, h, d):
        raise NotImplementedError

    def accept(self, S):
        return np.ones(len(S), dtype=bool)

    def scatter(self, rng):
        """Kellékek (u,v) helyei a lapon."""
        pts, tries = [], 0
        lim = 1 - self.min_edge
        while len(pts) < self.n_props and tries < 60:
            tries += 1
            uv = rng.uniform(-lim, lim, size=(4000, 2))
            uv = uv[self.accept(self.shape(face_dirs(self.k, uv)))]
            pts.extend(uv)
        return np.array(pts[: self.n_props]).reshape(-1, 2)

    def build_props(self, amp, pos, nrm, R, center):
        return []


# ------------------------------------------------------------ Consumer Discretionary: cukros dombok + sátrak
class Consumer(Theme):
    sector, short, seed, n_props = "Consumer Discretionary", "Consumer", 55, 60

    def shape(self, d):
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        bumps = 0.5 + 0.5 * np.sin(8 * x + 1) * np.sin(8 * y + 2) * np.sin(8 * z + 3) * 1.8
        return np.clip(0.7 * np.clip(bumps, 0, 1) + 0.3 * stretch(fbm(d, self.seed, octaves=2)), 0, 1)

    def colors(self, h, d):
        return gradient(h, [(0, (255, 150, 200)), (0.5, (255, 205, 130)), (1, (150, 220, 255))])

    def build_props(self, amp, pos, nrm, R, center):
        pal = [(255, 70, 110), (255, 200, 40), (70, 200, 255), (160, 90, 255), (60, 220, 150)]
        return [make_prop("cone", q, n, 0.04 + 0.04 * r, 0.05 + 0.02 * r, pal[i % len(pal)])
                for i, (q, n, r) in enumerate(zip(pos, nrm, R))]


# ------------------------------------------------------------ Energy: vulkán
class Energy(Theme):
    sector, short, seed, n_props = "Energy", "Energy", 11, 30

    def shape(self, d):
        dist = angular_distance(d, self.center)
        prof = np.clip(1 - dist / 0.60, 0, 1) ** 1.25
        prof = np.where(prof > 0.88, 0.88 - (prof - 0.88) * 1.6, prof)       # kráter
        rough = stretch(fbm(d, self.seed, octaves=5, freq=2.5))
        return np.clip(0.85 * prof + 0.15 * rough * (0.3 + prof), 0, 1)

    def colors(self, h, d):
        h = h / 0.85
        return gradient(h, [(0.0, (38, 30, 30)), (0.35, (70, 42, 36)), (0.65, (150, 55, 25)),
                            (0.85, (235, 100, 20)), (1.0, (255, 210, 70))])

    def accept(self, S):
        return S < 0.55        # sziklák a vulkán lábánál, ne a kráterben

    def build_props(self, amp, pos, nrm, R, center):
        out = [make_prop("rock", q, n, 0.02 + 0.03 * r, 0.02 + 0.035 * r, (55, 48, 48))
               for q, n, r in zip(pos, nrm, R)]
        cpos, cnrm, camp = center
        if camp > 0.45:                                                       # kitörés
            out.append(make_prop("cone", cpos, cnrm, 0.45 * (camp - 0.4), 0.05 + 0.05 * camp, (255, 130, 25)))
        return out


# ------------------------------------------------------------ Financials: zikkurat + arany oszlopok
class Financials(Theme):
    sector, short, seed, n_props = "Financials", "Financials", 33, 40

    def shape(self, d):
        dist = angular_distance(d, self.center)
        base = np.clip(1 - dist / 0.8, 0, 1) * 0.85 + 0.15 * stretch(fbm(d, self.seed, octaves=3))
        return np.floor(np.clip(base, 0, 1) * 6) / 6                          # teraszok

    def colors(self, h, d):
        return gradient(h, [(0, (95, 75, 35)), (0.5, (170, 130, 40)), (1, (255, 225, 120))])

    def build_props(self, amp, pos, nrm, R, center):
        out = []
        for q, n, r, a in zip(pos, nrm, R, amp):
            h = 0.02 + 0.24 * a * (0.2 + 0.8 * r)
            c = gradient(np.array([h / 0.26]), [(0, (150, 110, 20)), (1, (255, 225, 110))])[0]
            out.append(make_prop("cyl", q, n, h, 0.045, c, resolution=6))      # hatszög oszlop = oszlopdiagram
        return out


# ------------------------------------------------------------ Health Care: zöld dombok + fák
class Healthcare(Theme):
    sector, short, seed, n_props = "Health Care", "Healthcare", 44, 110

    def shape(self, d):
        return stretch(fbm(d, self.seed, octaves=3, freq=1.5), 2.0) ** 1.1

    def colors(self, h, d):
        return gradient(h, [(0, (50, 125, 70)), (0.6, (110, 185, 100)), (1, (190, 230, 150))])

    def build_props(self, amp, pos, nrm, R, center):
        greens = [(30, 110, 55), (45, 140, 65), (25, 95, 60)]
        return [make_prop("cone", q, n, 0.035 + 0.03 * r, 0.025 + 0.01 * r, greens[i % 3])
                for i, (q, n, r) in enumerate(zip(pos, nrm, R))]


# ------------------------------------------------------------ Industrials: sziklás hegyek + kémények
class Industrials(Theme):
    sector, short, seed, n_props = "Industrials", "Industrials", 66, 10

    def shape(self, d):
        return stretch(fbm(d, self.seed, octaves=5, freq=2.2, ridged=True), 2.6) ** 1.4

    def colors(self, h, d):
        return gradient(h, [(0, (70, 72, 78)), (0.55, (130, 130, 138)), (0.75, (175, 175, 182)), (0.85, (245, 245, 250))])

    def accept(self, S):
        return S < 0.3          # gyárak az alacsony részeken

    def build_props(self, amp, pos, nrm, R, center):
        return [make_prop("cyl", q, n, 0.06 + 0.06 * r, 0.02, (150, 70, 60), resolution=10)
                for q, n, r in zip(pos, nrm, R)]


# ------------------------------------------------------------ Information Technology: nagyváros
class Technology(Theme):
    sector, short, seed, n_props, relief = "Information Technology", "Technology", 22, 120, 0.5

    def shape(self, d):
        return 0.25 * stretch(fbm(d, self.seed, octaves=3, freq=2.0))

    def colors(self, h, d):
        return np.array([22, 32, 52.0]) + 40 * (h[:, None] / 0.25) * np.array([0.3, 0.5, 1.0])

    def build_props(self, amp, pos, nrm, R, center):
        out = []
        for q, n, r, a in zip(pos, nrm, R, amp):
            h = 0.015 + 0.34 * a * (0.12 + 0.88 * r ** 2)
            c = gradient(np.array([h / 0.36]), [(0, (0, 200, 255)), (1, (255, 60, 200))])[0]
            out.append(make_prop("box", q, n, h, 0.035 + 0.02 * r, c))
        return out


THEME_CLASSES = [Consumer, Energy, Financials, Healthcare, Industrials, Technology]
SECTOR_ORDER = [c.sector for c in THEME_CLASSES]

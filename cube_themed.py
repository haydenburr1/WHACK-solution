"""cube.py + szektor-témák (vulkán, város, arany tornyok, fák, ...).

Ugyanaz a kocka, ugyanaz a csúszka és hover, mint a cube.py-ban, de:
  - a magasság továbbra is a volatilitásból jön (cube.py get_sector_map), a téma csak finoman formálja
  - minden szektornak saját színe és kellékei vannak (planet/themes.py)
  - a kellékek (épületek, fák...) mérete a helyi volatilitást követi

Futtatás a repo gyökeréből (ahol a cube.py és a planet/ mappa van):
    python cube_themed.py
    python cube_themed.py --no-props       # csak a színezett terep, gyorsabb
A cube.py-t NEM módosítja, csak importálja.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "planet"))

import numpy as np
import pyvista as pv
import vtk
from scipy.ndimage import zoom, map_coordinates
from scipy.spatial import cKDTree

import cube as C                                   # a csapat cube.py-ja: adat + geometria
from themes import THEME_CLASSES                   # planet/themes.py
from geometry import smoothstep, merge_props       # planet/geometry.py

# ------------------------------------------------------------------ beállítások
THEME_MIX = 0.5       # 0 = tiszta cube.py-s terep, 1 = a téma alakja erősen formálja a magasságot
EDGE_BLEND = 0.10     # a lap széle felé ennyire (egységnyi lapfélszélességben) sötétedik/lelapul
AMP_MIN = 0.08        # a kellékek 'nyugodt' szintje
SINK = 0.004          # a kellékek kicsit a talajba süllyednek
NO_PROPS = "--no-props" in sys.argv

HALF = C.HALF
R = C.RESOLUTION
UNIT_PER_DENSE = C.HEIGHT_SCALE / HALF            # 1 'dense' egység magassága egységkocka-méretben
VOL_MAX = C.COLOR_LIMITS[1]

THEMES = {cls.sector: cls(k) for k, cls in enumerate(THEME_CLASSES)}


def unit_point(sector, a_u, b_u):
    """Lapkoordináták (-1..1) -> egységkocka pontja a cube.py tengely-konvenciójával."""
    axis, sign = C.FACE_AXES[sector]
    first, second = [i for i in range(3) if i != axis]
    out = np.zeros(np.shape(a_u) + (3,))
    out[..., axis] = sign
    out[..., first] = a_u
    out[..., second] = b_u
    return out


def normal_of(sector):
    axis, sign = C.FACE_AXES[sector]
    n = np.zeros(3)
    n[axis] = sign
    return n


class FaceStatic:
    """Ami nem függ a naptól: téma-alakzat, alapszínek, kellékek helye."""

    def __init__(self, sector, rng):
        self.sector = sector
        t = self.theme = THEMES[sector]
        u = np.linspace(-1, 1, R)
        a, b = np.meshgrid(u, u)
        pts = unit_point(sector, a, b).reshape(-1, 3)
        d = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        raw = t.shape(d)
        self.shape_max = max(float(raw.max()), 1e-9)
        self.shape_n = (raw / self.shape_max).reshape(R, R)
        self.mask = smoothstep(0, EDGE_BLEND, 1 - np.maximum(np.abs(a), np.abs(b)))
        base = t.colors(raw, d).reshape(R, R, 3) * (0.6 + 0.4 * self.mask)[..., None]
        self.base_rgb = base
        self.factor = (1 - THEME_MIX) + 1.4 * THEME_MIX * self.shape_n

        # kellékek helye: (a_u, b_u) a lapon, a téma szabályai szerint
        self.prop_ab = np.zeros((0, 2))
        if not NO_PROPS and t.n_props:
            lim = 1 - t.min_edge
            chosen, tries = [], 0
            while len(chosen) < t.n_props and tries < 60:
                tries += 1
                ab = rng.uniform(-lim, lim, size=(4000, 2))
                dd = unit_point(sector, ab[:, 0], ab[:, 1])
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                chosen.extend(ab[t.accept(t.shape(dd))])
            self.prop_ab = np.array(chosen[: t.n_props]).reshape(-1, 2)
            pp = unit_point(sector, self.prop_ab[:, 0], self.prop_ab[:, 1])
            self.prop_dir = pp / np.linalg.norm(pp, axis=1, keepdims=True)
            self.prop_shape_n = t.shape(self.prop_dir) / self.shape_max
            self.prop_rand = rng.random(len(self.prop_ab))
            cc = unit_point(sector, np.array([0.0]), np.array([0.0]))
            self.center_shape_n = float(t.shape(cc / np.linalg.norm(cc, axis=1, keepdims=True))[0] / self.shape_max)


def _sample(field, ab):
    """Bilineáris minta a (R,R) mezőből (sor = b, oszlop = a), ab: (n,2) -1..1."""
    cols = (ab[:, 0] + 1) / 2 * (R - 1)
    rows = (ab[:, 1] + 1) / 2 * (R - 1)
    return map_coordinates(field, [rows, cols], order=1)


def compute_face(fs, day):
    """Egy lap adatai egy napra (csak numpy, nincs pyvista)."""
    sector, t = fs.sector, fs.theme
    hm, tickers = C.get_sector_map(C.sector_data, sector, C.ticker_data, day)

    # ugyanaz a feldolgozás, mint a cube.create_cube_face-ben
    padded = np.pad(hm, 1)
    n = padded.shape[0]
    dense = np.clip(zoom(padded, R / n, order=3), 0, None)
    dense[0, :] = dense[-1, :] = dense[:, 0] = dense[:, -1] = 0

    disp = dense * fs.factor                         # a téma formálja az adat-magasságot
    u = np.linspace(-HALF, HALF, R)
    a, b = np.meshgrid(u, u)
    x, y, z = C.face_to_xyz(sector, a, b, disp)

    tn = np.clip(dense / VOL_MAX, 0, 1)              # volatilitás 0..1
    rgb = np.clip(fs.base_rgb * (0.65 + 0.55 * tn)[..., None], 0, 255).astype(np.uint8)

    # hover-pontok: a cellák közepe, a TÉNYLEGES (témázott) felszín fölött
    coarse = np.linspace(-HALF, HALF, n)
    side = hm.shape[0]
    idx = np.arange(len(tickers))
    rows, cols = idx // side, idx % side
    di = np.stack([(rows + 1) * (R - 1) / (n - 1), (cols + 1) * (R - 1) / (n - 1)])
    h_cell = map_coordinates(disp, di, order=1) + 0.0125
    labels = np.array([C.face_to_xyz(sector, coarse[c + 1], coarse[r + 1], h)
                       for r, c, h in zip(rows, cols, h_cell)]).reshape(-1, 3)

    props = None
    if len(fs.prop_ab):
        dp = _sample(dense, fs.prop_ab)
        amp = AMP_MIN + (1 - AMP_MIN) * np.clip(dp / VOL_MAX, 0, 1)
        h_unit = dp * ((1 - THEME_MIX) + 1.4 * THEME_MIX * fs.prop_shape_n) * UNIT_PER_DENSE - SINK
        nrm = normal_of(sector)
        base = unit_point(sector, fs.prop_ab[:, 0], fs.prop_ab[:, 1])
        pos = base + nrm * h_unit[:, None]
        dc = float(_sample(dense, np.zeros((1, 2)))[0])
        camp = AMP_MIN + (1 - AMP_MIN) * min(dc / VOL_MAX, 1)
        ch = dc * ((1 - THEME_MIX) + 1.4 * THEME_MIX * fs.center_shape_n) * UNIT_PER_DENSE - SINK
        cpos = unit_point(sector, np.array([0.0]), np.array([0.0]))[0] + nrm * ch
        props = dict(amp=amp, pos=pos, nrm=np.tile(nrm, (len(pos), 1)), R=fs.prop_rand,
                     center=(cpos, nrm, camp))

    return dict(x=x, y=y, z=z, rgb=rgb, labels=labels, tickers=tickers, props=props)


# ------------------------------------------------------------------ pyvista rész
def face_mesh(f):
    grid = pv.StructuredGrid(f["x"], f["y"], f["z"])
    # a StructuredGrid pontsorrendje Fortran-sorrend -> a színt is úgy kell sorba rakni
    flat = np.stack([f["rgb"][..., c].ravel("F") for c in range(3)], axis=1)
    grid.point_data["rgb"] = flat
    return grid.extract_surface(algorithm="dataset_surface").triangulate()


def build_all(statics, day):
    faces = [compute_face(fs, day) for fs in statics]
    cube = pv.merge([face_mesh(f) for f in faces], merge_points=True)
    labels = np.vstack([f["labels"] for f in faces])
    names = [t for f in faces for t in f["tickers"]]
    prop_meshes = {}
    for fs, f in zip(statics, faces):
        if f["props"] is None:
            continue
        p = f["props"]
        m = merge_props(fs.theme.build_props(p["amp"], p["pos"], p["nrm"], p["R"], p["center"]))
        if m is not None:
            prop_meshes[fs.sector] = m.scale(HALF, inplace=False)    # egységkocka -> jelenet méret
    return cube, prop_meshes, labels, names


FIRST_DAY, LAST_DAY = C.FIRST_DAY, C.LAST_DAY

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    statics = [FaceStatic(s, rng) for s in C.FACE_AXES]
    cache = {}

    plotter = pv.Plotter(window_size=(1200, 900))
    texture = pv.read_texture("background_2.jpeg")
    plotter.add_actor(texture.to_skybox())
    state = {"day": FIRST_DAY}

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)
    picker.PickFromListOn()

    tooltip = plotter.add_text("", name="hover", position=(0, 0), font_size=11)
    tooltip.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
    tooltip.GetTextProperty().SetBackgroundOpacity(0.8)
    tooltip.GetTextProperty().SetColor(1, 1, 1)

    def show_day(day, first=False):
        if day not in cache:
            if len(cache) > 20:
                cache.clear()
            cache[day] = build_all(statics, day)
        cube, prop_meshes, labels, names = cache[day]
        actor = plotter.add_mesh(cube, name="cube", scalars="rgb", rgb=True, smooth_shading=True,
                                 reset_camera=first, render=False)
        for s in C.FACE_AXES:
            if s in prop_meshes:
                plotter.add_mesh(prop_meshes[s], name=f"props_{s}", scalars="rgb", rgb=True,
                                 reset_camera=False, render=False)
            else:
                plotter.remove_actor(f"props_{s}", render=False)
        plotter.add_text(str(C.ticker_data["date"].iloc[day]), name="date", position="upper_left", font_size=14)
        state.update(tree=cKDTree(labels), labels=names)
        picker.InitializePickList()
        picker.AddPickList(actor)
        tooltip.SetInput("")
        plotter.render()

    show_day(FIRST_DAY, first=True)

    def on_slide(value):
        day = int(round(value))
        if day != state["day"]:
            state["day"] = day
            show_day(day)

    def on_mouse_move(interactor, _event):
        x, y = interactor.GetEventPosition()
        text = ""
        if picker.Pick(x, y, 0, plotter.renderer):
            distance, index = state["tree"].query(picker.GetPickPosition())
            if distance < C.HOVER_RADIUS:
                ticker = state["labels"][index]
                name, sector = C.TICKER_INFO.loc[ticker]
                vol = C.ticker_data[ticker].iloc[state["day"]]
                text = f"{ticker}  {name}\n{sector}\nVolatility: {vol:.3f}"
        if text != tooltip.GetInput():
            tooltip.SetInput(text)
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()
        elif text:
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()

    plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)

    slider = plotter.add_slider_widget(
        on_slide, rng=[FIRST_DAY, LAST_DAY], value=FIRST_DAY, title="Trading day", fmt="%.0f",
        pointa=(0.35, 0.93), pointb=(0.95, 0.93), style="modern",
        interaction_event="end",      # elengedéskor frissül (a kellékek építése lassabb); "always" = folyamatos
    )

    def step(delta):
        day = int(np.clip(state["day"] + delta, FIRST_DAY, LAST_DAY))
        slider.GetRepresentation().SetValue(day)
        on_slide(day)

    plotter.add_key_event("Right", lambda: step(1))
    plotter.add_key_event("Left", lambda: step(-1))
    plotter.add_axes()
    plotter.show()

"""Bolygó-vizualizáció: 6 szektor, a magasság a részvények volatilitását követi.

Futtatás a repo gyökeréből (ahol a data/ mappa van):
    python planet/main.py                      # a repo adataival, kockán
    python planet/main.py --shape sphere       # ugyanez gömbön
    python planet/main.py --demo               # kitalált adat (ha nincs data/)
    python planet/main.py --mode sector        # csak szektorátlag számít (nem részvényenként)
    python planet/main.py --gif out.gif        # forgó animáció fájlba
    python planet/main.py --screenshot a.png --step 1500
"""
import argparse
import numpy as np
import pyvista as pv

import data as D
from themes import SECTOR_ORDER, THEME_CLASSES
from world import World


def draw(pl, world, x, sector_vol, title):
    pts = world.terrain_points(x)
    mesh = pv.PolyData(pts, world.faces_flat)
    mesh.point_data["rgb"] = world.rgb
    pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, name="terrain", reset_camera=False)
    for k in range(6):
        m = world.props(k, x)
        if m is None:
            pl.remove_actor(f"props{k}")
        else:
            pl.add_mesh(m, scalars="rgb", rgb=True, name=f"props{k}", reset_camera=False)
    cent = np.array([t.center for t in world.themes]) * 1.7
    labels = [f"{c.short}\n{v:.0%}" for c, v in zip(THEME_CLASSES, sector_vol)]
    pl.add_point_labels(pv.PolyData(cent), labels, font_size=14, shape=None, show_points=False,
                        text_color="white", name="labels")
    pl.add_text(title, position="upper_left", font_size=12, color="white", name="title")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--shape", choices=["cube", "sphere"], default="cube")
    ap.add_argument("--mode", choices=["stock", "sector"], default="stock")
    ap.add_argument("--grid", type=int, default=80, help="rács/lap (nagyobb = finomabb, lassabb)")
    ap.add_argument("--gif"); ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--screenshot"); ap.add_argument("--step", type=int, help="melyik nap (screenshothoz)")
    args = ap.parse_args()

    vol, uni = D.demo_dataset() if args.demo else D.load_repo(args.data)
    raw = vol.to_numpy(float)
    raw = np.where(np.isfinite(raw), raw, np.nanmedian(raw))
    vmax = np.quantile(raw, 0.98)                      # közös skála minden szektorra és napra
    X = np.clip(raw / vmax, 0, 1)
    T = len(X)
    dates = [str(i)[:10] for i in vol.index]

    layout = D.build_layout(uni, list(vol.columns))
    world = World(layout, shape=args.shape, N=args.grid, mode=args.mode)
    sec_mean = lambda r: [r[idx].mean() for idx, _ in layout]

    def show(t):
        t = float(np.clip(t, 0, T - 1)); i = int(t); j = min(i + 1, T - 1); f = t - i
        x = (1 - f) * X[i] + f * X[j]
        r = (1 - f) * raw[i] + f * raw[j]
        draw(pl, world, x, sec_mean(r), f"{dates[i]}   ({args.shape}, {args.mode})")

    off = bool(args.gif or args.screenshot)
    pl = pv.Plotter(off_screen=off, window_size=(1100, 900))
    pl.set_background("black", top="midnightblue")
    show(0)
    pl.reset_camera()

    if args.gif:
        pl.open_gif(args.gif)
        for f in range(args.frames):
            show((T - 1) * f / max(args.frames - 1, 1))
            pl.camera.Azimuth(360 / args.frames)
            pl.write_frame()
        pl.close()
    elif args.screenshot:
        show(args.step if args.step is not None else T // 2)
        pl.screenshot(args.screenshot)
        pl.close()
    else:
        pl.add_slider_widget(lambda v: show(v), rng=[0, T - 1], value=0, title="Nap",
                             fmt="%0.0f", pointa=(0.2, 0.07), pointb=(0.8, 0.07))
        pl.show()


if __name__ == "__main__":
    main()

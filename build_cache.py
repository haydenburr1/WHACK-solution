"""
Precompile every trading day for cube.py into cube_cache/.

Run once (takes a few minutes; it is resumable, so Ctrl+C and re-run is fine):

    python build_cache.py

What gets stored, per day (rows FIRST_DAY..LAST_DAY of volatility.csv):
  Z.npy        (days, 6, res, res) float16  finished height maps (beach + land lift applied)
  xy.npy       (days, 48, 2)       float32  peak positions in tile coords
  heights.npy  (days, 48)          float32  peak heights
  trees.npy    (days, MAX_TREES, 4) float32 tree x, y, z, size in tile coords (NaN = no tree)
  tree_sec.npy (days, MAX_TREES)   int8     sector index of each tree (-1 = no tree)
  done.npy     (days,)             bool     which days are finished

Days covered by struct_Z.npy are copied from it, so they look exactly as before.
With the cache in place the viewer never grows terrain or runs MDS while sliding.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

import cube
from cube import (FIRST_DAY, LAST_DAY, MAX_SIDE, SavedTerrain, res, sector_layout,
                  sector_terrain, sector_tickers, sectors, tree_local, vol)

CACHE = Path("cube_cache")
N_DAYS = LAST_DAY - FIRST_DAY + 1
N_STOCKS = sum(len(t) for t in sector_tickers)
MAX_TREES = N_STOCKS * (1 + MAX_SIDE)


def day_task(day):
    """Worker: all six tiles for one day."""
    t = vol["date"].iloc[day]
    return day, [sector_terrain(i, t) for i in range(len(sectors))]


def saved_task(day, saved):
    t = vol["date"].iloc[day]
    return day, [(np.asarray(Z, np.float32), *sector_layout(i, t))
                 for i, (Z, *_) in enumerate(saved.tiles(day))]


def _arr(name, shape, dtype, fill):
    path = CACHE / name
    if path.exists():
        a = open_memmap(path, mode="r+")
        if a.shape == shape and a.dtype == dtype:
            return a
    a = open_memmap(path, mode="w+", shape=shape, dtype=dtype)
    a[:] = fill
    return a


def main():
    CACHE.mkdir(exist_ok=True)
    Z = _arr("Z.npy", (N_DAYS, len(sectors), res, res), np.float16, 0)
    XY = _arr("xy.npy", (N_DAYS, N_STOCKS, 2), np.float32, np.nan)
    H = _arr("heights.npy", (N_DAYS, N_STOCKS), np.float32, np.nan)
    TR = _arr("trees.npy", (N_DAYS, MAX_TREES, 4), np.float32, np.nan)
    TS = _arr("tree_sec.npy", (N_DAYS, MAX_TREES), np.int8, -1)
    done_path = CACHE / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(N_DAYS, bool)
    if done.shape != (N_DAYS,):
        done = np.zeros(N_DAYS, bool)

    def store(day, tiles):
        j = day - FIRST_DAY
        Z[j] = np.stack([t[0] for t in tiles]).astype(np.float16)
        XY[j] = np.concatenate([t[1] for t in tiles])
        H[j] = np.concatenate([t[2] for t in tiles])
        TR[j], TS[j] = np.nan, -1
        k = 0
        for i, L in enumerate(tree_local(tiles)):
            if L is None:
                continue
            TR[j, k:k + len(L)] = L
            TS[j, k:k + len(L)] = i
            k += len(L)
        done[j] = True

    todo = [d for d in range(FIRST_DAY, LAST_DAY + 1) if not done[d - FIRST_DAY]]
    print(f"{N_DAYS - len(todo)}/{N_DAYS} days already cached, {len(todo)} to go.")

    saved = SavedTerrain()
    from_npy = [d for d in todo if saved.has(d)]
    for d in from_npy:
        store(*saved_task(d, saved))
    todo = [d for d in todo if not saved.has(d)]

    t0, n = time.perf_counter(), 0
    workers = max(1, (os.cpu_count() or 2) - 1)
    try:
        with ProcessPoolExecutor(workers) as pool:
            for f in as_completed([pool.submit(day_task, d) for d in todo]):
                store(*f.result())
                n += 1
                if n % 25 == 0 or n == len(todo):
                    np.save(done_path, done)
                    rate = n / (time.perf_counter() - t0)
                    print(f"  {n}/{len(todo)} days  ({rate:.1f}/s, "
                          f"~{(len(todo) - n) / rate / 60:.1f} min left)")
    finally:
        for a in (Z, XY, H, TR, TS):
            a.flush()
        np.save(done_path, done)
    print(f"Done: {int(done.sum())}/{N_DAYS} days in {CACHE}/")


if __name__ == "__main__":
    main()

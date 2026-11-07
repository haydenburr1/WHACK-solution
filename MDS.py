"""Multidimensional scaling of stocks from rolling return correlations.

Distance between two stocks i, j over a window: d_ij = sqrt(2 * (1 - rho_ij)),
so perfectly correlated stocks sit on top of each other (d = 0) and
anti-correlated ones are farthest apart (d = 2). Classical (Torgerson) MDS
then gives 2-D coordinates whose pairwise distances approximate d.

Steps, per sector, at every keyframe (every KEY_STEP trading days):
  1. correlation of the last WINDOW days of returns
  2. squared distance  D2 = 2 * (1 - rho)
  3. classical MDS     B = -1/2 * J @ D2 @ J, top-2 eigenvectors * sqrt(eigenvalues)
  4. Procrustes        rotate/flip to match the previous keyframe (stops spinning)
then, over all keyframes:
  5. smoothing         centred rolling mean over SMOOTH_KEYFRAMES keyframes
  6. fixed scale       one constant for every sector and keyframe, so clusters
                       really tighten when correlations jump (e.g. March 2020)

Run:  python MDS.py
Output: data/mds_positions.npz
  tickers   (n,)        str
  sector    (n,)        str
  sub       (n,)        str
  face      (n,)        int, index into SECTORS
  dates     (NK,)       str, YYYY-MM-DD, one per keyframe
  key_idx   (NK,)       int, row of each keyframe in the returns table
  pos       (NK, n, 2)  positions, roughly within [-1, 1] per face
  avg_corr  (NK, 6)     mean pairwise correlation per sector
  fit       (NK, 6)     share of the distance structure the 2-D map explains
"""
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

WINDOW = 150            # trading days of returns per correlation window
KEY_STEP = 5            # trading days between keyframes
FIRST_DATE = "2018-01-01"
SMOOTH_KEYFRAMES = 5    # ~1 month of keyframes (5 x 5 trading days)
SCALE_PERCENTILE = 98   # this percentile of distance-from-centre maps to 1.0

SECTORS = [             # face order; must match the sectors in universe.csv
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Industrials",
    "Energy",
]

OUTPUT_FILE = "data/mds_positions.npz"


# ------------------------------------------------------------- core maths ---

def classical_mds(d2: np.ndarray, dims: int = 2):
    """Positions (n, dims) from a matrix of SQUARED distances, plus the fit.

    fit = share of the positive eigenvalue mass kept by the top `dims` axes:
    1.0 means the 2-D map reproduces the distances exactly.
    """
    n = len(d2)
    j = np.eye(n) - np.full((n, n), 1.0 / n)        # centring matrix
    b = -0.5 * j @ d2 @ j                           # inner products between stocks
    vals, vecs = np.linalg.eigh(b)                  # ascending order
    vals, vecs = vals[::-1], vecs[:, ::-1]          # largest first
    top = np.clip(vals[:dims], 0.0, None)           # rounding can give tiny negatives
    positive = vals[vals > 0].sum()
    fit = top.sum() / positive if positive > 0 else 1.0
    return vecs[:, :dims] * np.sqrt(top), fit


def align(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Rotate/reflect centred `cur` to best match centred `prev` (Procrustes)."""
    r, _ = orthogonal_procrustes(cur, prev)
    return cur @ r


def corr_to_d2(rho: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (1.0 - rho), 0.0, None)


# -------------------------------------------------------------- pipeline ----

def keyframe_rows(returns: pd.DataFrame) -> np.ndarray:
    """Rows to compute layouts at: from FIRST_DATE (or the first full window)."""
    before = returns.index.searchsorted(pd.Timestamp(FIRST_DATE))   # return days before FIRST_DATE
    first = max(before, WINDOW - 1)
    if first > before:
        print(f"Note: only {before} days of returns before {FIRST_DATE} "
              f"(need {WINDOW - 1}); "
              f"first keyframe is {returns.index[first].date()} "
              f"(start prices earlier to fix).")
    return np.arange(first, len(returns), KEY_STEP)


def sector_layouts(r: np.ndarray, rows: np.ndarray, key_idx: np.ndarray):
    """Raw aligned layouts (NK, len(rows), 2), mean correlation and fit for one sector."""
    nk, m = len(key_idx), len(rows)
    pos = np.zeros((nk, m, 2))
    avg_corr = np.zeros(nk)
    fit = np.zeros(nk)
    prev = None

    for k, t in enumerate(key_idx):
        window = r[t - WINDOW + 1:t + 1, rows]                 # WINDOW days x m stocks
        rho = np.nan_to_num(np.corrcoef(window, rowvar=False), nan=0.0)
        np.fill_diagonal(rho, 1.0)
        avg_corr[k] = rho[np.triu_indices(m, 1)].mean()

        xy, fit[k] = classical_mds(corr_to_d2(rho))
        xy -= xy.mean(axis=0)
        if prev is not None:
            xy = align(xy, prev)
        pos[k] = prev = xy

    return pos, avg_corr, fit


def smooth(pos: np.ndarray, width: int) -> np.ndarray:
    """Centred rolling mean along the keyframe axis (edges use what's available)."""
    if width <= 1:
        return pos
    out = np.empty_like(pos)
    half = width // 2
    for k in range(len(pos)):
        out[k] = pos[max(0, k - half):k + half + 1].mean(axis=0)
    return out


def build(prices: pd.DataFrame, uni: pd.DataFrame):
    uni = uni[uni["sector"].isin(SECTORS)].copy()
    uni["face"] = uni["sector"].map({s: i for i, s in enumerate(SECTORS)})
    uni = uni.sort_values(["face", "sub_industry", "ticker"]).reset_index(drop=True)
    tickers = uni["ticker"].tolist()

    returns = prices[tickers].pct_change().iloc[1:]   # drop the first row, which is NaN
    r = returns.to_numpy()
    key_idx = keyframe_rows(returns)

    n, nk = len(tickers), len(key_idx)
    pos = np.zeros((nk, n, 2))
    avg_corr = np.zeros((nk, len(SECTORS)))
    fit = np.zeros((nk, len(SECTORS)))

    for face in range(len(SECTORS)):
        rows = np.flatnonzero(uni["face"].to_numpy() == face)
        if len(rows) < 3:
            raise ValueError(f"{SECTORS[face]} has {len(rows)} stocks; MDS needs at least 3.")
        p, avg_corr[:, face], fit[:, face] = sector_layouts(r, rows, key_idx)
        pos[:, rows] = smooth(p, SMOOTH_KEYFRAMES)

    # One fixed scale for everything, so a sector that tightens in a crash
    # visibly shrinks instead of being stretched back to full size.
    radius = np.linalg.norm(pos, axis=2)
    pos /= np.percentile(radius, SCALE_PERCENTILE)

    return {
        "tickers": np.array(tickers),
        "sector": uni["sector"].to_numpy(str),
        "sub": uni["sub_industry"].to_numpy(str),
        "face": uni["face"].to_numpy(int),
        "dates": np.array([d.strftime("%Y-%m-%d") for d in returns.index[key_idx]]),
        "key_idx": key_idx,
        "pos": pos,
        "avg_corr": avg_corr,
        "fit": fit,
    }


# ---------------------------------------------------------------- checks ----

def self_test():
    """MDS must recover a known 2-D shape exactly (up to rotation/flip)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(6, 2))
    d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    xy, fit = classical_mds(d2)
    d2_back = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    assert np.allclose(d2, d2_back), "classical_mds did not reproduce known distances"
    assert np.isclose(fit, 1.0), "a true 2-D shape should have fit 1.0"


def report(world: dict, date: str = "2020-03-16", sector: str = "Financials"):
    """Print each stock's nearest neighbour in one sector on one date."""
    k = int(np.searchsorted(world["dates"], date))
    k = min(k, len(world["dates"]) - 1)
    face = SECTORS.index(sector)
    rows = np.flatnonzero(world["face"] == face)
    xy = world["pos"][k, rows]
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    same = 0
    print(f"\n{sector} on {world['dates'][k]} (avg corr {world['avg_corr'][k, face]:.2f}, "
          f"2-D fit {world['fit'][k, face]:.0%}): nearest neighbours")
    for i, row in enumerate(rows):
        j = rows[np.argmin(d[i])]
        match = world["sub"][row] == world["sub"][j]
        same += match
        print(f"  {world['tickers'][row]:<6} -> {world['tickers'][j]:<6} "
              f"{'same sub-industry' if match else ''}")
    print(f"  {same}/{len(rows)} nearest neighbours share a sub-industry")


def main():
    self_test()
    prices = pd.read_parquet("data/prices.parquet")
    uni = pd.read_csv("data/universe.csv")

    world = build(prices, uni)
    np.savez_compressed(OUTPUT_FILE, **world)

    nk, n, _ = world["pos"].shape
    print(f"Saved {OUTPUT_FILE}: {n} stocks x {nk} keyframes "
          f"({world['dates'][0]} to {world['dates'][-1]})")
    print("Mean 2-D fit per sector:")
    for face, s in enumerate(SECTORS):
        print(f"  {s:<24} {world['fit'][:, face].mean():.0%}")
    report(world)


if __name__ == "__main__":
    main()

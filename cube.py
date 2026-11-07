import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

sector_data = pd.read_csv("data/universe.csv")
ticker_data = pd.read_csv("data/volatility.csv")

CUBE_SIZE = 10.0
HALF = CUBE_SIZE / 2
HEIGHT_SCALE = 4.0
RESOLUTION = 60  # points per side on every face (same for all, so edges line up)

# Which face each sector sits on: (axis of the face normal, +1 / -1 side).
# axis 0 = x, 1 = y, 2 = z. The other two axes are the face's (a, b) directions.
FACE_AXES = {
    "Consumer Discretionary": (0, 1),
    "Energy": (0, -1),
    "Financials": (1, 1),
    "Health Care": (1, -1),
    "Industrials": (2, 1),
    "Information Technology": (2, -1),
}


def get_sector_map(sector_data, sector, ticker_data, day=2):
    # Tickers belonging to this sector only
    tickers = sector_data.loc[sector_data["sector"] == sector, "ticker"].tolist()

    # Square grid big enough to hold every ticker (round up, don't drop any)
    side = int(np.ceil(np.sqrt(len(tickers))))
    height_map = np.zeros((side, side))  # spare cells sit at 0 (flat)

    # One cell per ticker, filled row by row
    # day=2 because the first two rows of volatility.csv are NaN
    for i, ticker in enumerate(tickers):
        height_map[i // side, i % side] = ticker_data[ticker].iloc[day]

    return height_map, tickers


def face_to_xyz(sector, a, b, h):
    # Map face coordinates (a, b) plus a height h to 3D, pushing outward along the face normal.
    if sector not in FACE_AXES:
        raise ValueError(f"Unknown sector: {sector}")

    axis, sign = FACE_AXES[sector]
    first, second = [i for i in range(3) if i != axis]

    coords = [None, None, None]
    coords[axis] = sign * (HALF + h * HEIGHT_SCALE)
    coords[first] = a
    coords[second] = b
    return coords


def create_cube_face(sector, height_map, tickers):
    # Surround the grid with a ring of zeros so every face's border sits exactly
    # on the cube edge. Neighbouring faces then meet with no gaps.
    padded = np.pad(height_map, 1)
    n = padded.shape[0]

    # Upsample the coarse per-ticker grid into a smooth, dense surface.
    # Cubic interpolation can dip slightly below 0, so clip to keep heights non-negative.
    dense = zoom(padded, RESOLUTION / n, order=3)
    dense = np.clip(dense, 0, None)
    dense[0, :] = dense[-1, :] = dense[:, 0] = dense[:, -1] = 0  # pin the border exactly

    u = np.linspace(-HALF, HALF, RESOLUTION)
    a, b = np.meshgrid(u, u)

    x, y, z = face_to_xyz(sector, a, b, dense)

    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["height"] = dense.ravel()
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()

    # Label positions: the centre of each ticker's cell, just above the terrain.
    coarse = np.linspace(-HALF, HALF, n)
    side = height_map.shape[0]
    label_points = []
    for i in range(len(tickers)):
        row, col = i // side, i % side
        h = height_map[row, col] + 0.05  # small lift so labels don't sink into the surface
        label_points.append(face_to_xyz(sector, coarse[col + 1], coarse[row + 1], h))

    return surface, np.array(label_points)


# -----------------------------------------
# Generate complete cube
# -----------------------------------------

def create_terrain_cube(day=2):

    faces = []
    all_points = []
    all_labels = []

    for sector in FACE_AXES:

        height_map, tickers = get_sector_map(sector_data, sector, ticker_data, day)

        face, label_points = create_cube_face(sector, height_map, tickers)

        faces.append(face)
        all_points.append(label_points)
        all_labels.extend(tickers)

    cube = pv.merge(faces, merge_points=True)

    return cube, np.vstack(all_points), all_labels


# -----------------------------------------
# Visualize
# -----------------------------------------

FIRST_DAY = 2  # the first two rows of volatility.csv are NaN
LAST_DAY = len(ticker_data) - 1

# Fixed colour range for every day, so colours can be compared across time.
# Uses the 99th percentile so a few extreme days don't wash everything else out;
# anything above it shows as the top colour.
_all_vol = ticker_data.drop(columns="date").iloc[FIRST_DAY:].to_numpy().ravel()
COLOR_LIMITS = [0.0, float(np.nanpercentile(_all_vol, 99))]


TICKER_INFO = sector_data.set_index("ticker")[["name", "sector"]]

# How close (in scene units) the mouse must be to a stock's cell centre to show it.
# Roughly one cell width, so empty filler cells and the bare cube edge show nothing.
HOVER_RADIUS = 1.2


def draw_day(plotter, day, first=False):
    # Rebuild the cube for one trading day and swap it into the scene.
    # Using the same `name` each time replaces the previous actor instead of stacking them.
    # Returns the stock positions for this day so hover can find the nearest one.
    cube, label_points, labels = create_terrain_cube(day)

    actor = plotter.add_mesh(
        cube,
        name="cube",
        scalars="height",
        cmap="terrain",
        clim=COLOR_LIMITS,
        smooth_shading=True,
        reset_camera=first,
        render=False,
    )

    plotter.add_text(str(ticker_data["date"].iloc[day]), name="date", position="upper_left", font_size=14)

    return actor, cKDTree(label_points), labels


if __name__ == "__main__":

    plotter = pv.Plotter()
    state = {"day": FIRST_DAY}

    # Picker that finds the exact spot on the cube under the mouse (cube only, not axes/grid).
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)
    picker.PickFromListOn()

    # One reusable tooltip; we just change its text and position as the mouse moves.
    tooltip = plotter.add_text("", name="hover", position=(0, 0), font_size=11)
    tooltip.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
    tooltip.GetTextProperty().SetBackgroundOpacity(0.8)
    tooltip.GetTextProperty().SetColor(1, 1, 1)

    def show_day(day, first=False):
        actor, tree, labels = draw_day(plotter, day, first)
        state.update(tree=tree, labels=labels)
        picker.InitializePickList()
        picker.AddPickList(actor)  # the cube actor is new each day
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
            if distance < HOVER_RADIUS:
                ticker = state["labels"][index]
                name, sector = TICKER_INFO.loc[ticker]
                volatility = ticker_data[ticker].iloc[state["day"]]
                text = f"{ticker}  {name}\n{sector}\nVolatility: {volatility:.3f}"
        if text != tooltip.GetInput():
            tooltip.SetInput(text)
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()
        elif text:
            tooltip.SetPosition(x + 15, y + 15)
            plotter.render()

    plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)

    slider = plotter.add_slider_widget(
        on_slide,
        rng=[FIRST_DAY, LAST_DAY],
        value=FIRST_DAY,
        title="Trading day",
        fmt="%.0f",
        pointa=(0.35, 0.93),  # top of the window, clear of the colour bar
        pointb=(0.95, 0.93),
        style="modern",
        interaction_event="always",
    )

    # Left / Right arrow keys step one trading day at a time.
    def step(delta):
        day = int(np.clip(state["day"] + delta, FIRST_DAY, LAST_DAY))
        slider.GetRepresentation().SetValue(day)
        on_slide(day)

    plotter.add_key_event("Right", lambda: step(1))
    plotter.add_key_event("Left", lambda: step(-1))

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

import numpy as np
import pandas as pd
import pyvista as pv

sector_data = pd.read_csv("data/universe.csv")
ticker_data = pd.read_csv("data/volatility.csv")

def get_sector_map(sector_data, sector, ticker_data, ticker):
    # Get all the sectors so that we can place each sector on a different face of the cube.
    sectors_data = sector_data[sector_data["sector"] == sector]

    # Get all of the tickers and store it so that we can access it for volatility data.
    tickers_data = ticker_data[sector_data["ticker"]]

    # Get the count so we can get the x and y positions on each face.
    count = len(sectors_data)

    # Set them as sqrt since the area must add up to the total number of tickers in the sector.
    width = int(np.sqrt(count))
    height = int(np.sqrt(count))

    height_map = np.full((height, width), np.nan)

    for ticker in sector_data["ticker"]:
        # Stores each volatility data for each ticker in the height map.
        # Use .iloc[2] because the first two lines are NaN.
        volatility_data = tickers_data[ticker].iloc[2]

    for y in range(height):
        for x in range(width):
            height_map[y, x] = volatility_data

    return height_map

def create_cube_face(sector, height_map):
    rows, cols = height_map.shape
    CUBE_SIZE = 10.0

    u = np.linspace(-CUBE_SIZE / 2, CUBE_SIZE / 2, cols)
    v = np.linspace(-CUBE_SIZE / 2, CUBE_SIZE / 2, rows)

    a, b = np.meshgrid(u, v)

    half = CUBE_SIZE / 2

    if sector == "Consumer Discretionary":
        x, y, z = np.full_like(a, half), a, b

    elif sector == "Energy":
        x, y, z = np.full_like(a, -half), a, b

    elif sector == "Financials":
        x, y, z = a, np.full_like(a, half), b

    elif sector == "Health Care":
        x, y, z = a, np.full_like(a, -half), b

    elif sector == "Industrials":
        x, y, z = a, b, np.full_like(a, half)

    elif sector == "Information Technology":
        x, y, z = a, b, np.full_like(a, -half)

    else:
        raise ValueError(f"Unknown sector: {sector}")

    HEIGHT_SCALE = 0.5

    # Displace vertices along the face normal
    if sector in ["Consumer Discretionary", "Energy"]:
        x += np.sign(x) * height_map * HEIGHT_SCALE

    elif sector in ["Financials", "Health Care"]:
        y += np.sign(y) * height_map * HEIGHT_SCALE

    else:
        z += np.sign(z) * height_map * HEIGHT_SCALE

    # Create structured grid
    grid = pv.StructuredGrid(x, y, z)

    grid.point_data["height"] = height_map.ravel()

    return grid.extract_surface().triangulate()


# -----------------------------------------
# Generate complete cube
# -----------------------------------------

def create_terrain_cube():

    faces = []
    SECTORS = [
        "Consumer Discretionary",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology"
    ]

    for sector in SECTORS:

        height_map = get_sector_map(sector_data, sector, ticker_data, sector)

        face = create_cube_face(sector, height_map)

        faces.append(face)

    cube = pv.merge(faces, merge_points=True)

    return cube


# -----------------------------------------
# Visualize
# -----------------------------------------

if __name__ == "__main__":

    cube = create_terrain_cube()

    plotter = pv.Plotter()

    plotter.add_mesh(
        cube,
        scalars="height",
        cmap="terrain",
        show_edges=True,
        smooth_shading=True
    )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
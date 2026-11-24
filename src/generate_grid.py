"""
generate_grid.py
Generates a 0.25-degree cell list (cells.csv) for Turkey from
Natural Earth (Admin 1 - States, Provinces, 10m) boundaries.

Usage:
    pip install geopandas matplotlib numpy pandas
    python generate_grid.py
First, update the SHP_PATH variable according to your own file path.
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- SETTINGS
SHP_PATH = "data/boundary/ne_10m_admin_1_states_provinces.shp"
OUTPUT_CSV = "cells.csv"
OUTPUT_PNG = "cells_map.png"

ORIGIN_LAT = 35.0   # grid origin (latitude)
ORIGIN_LON = 25.0   # grid origin (longitude)
STEP = 0.5          # cell size (degrees)
BLOCK_SIZE = 4      # block = 8x8 cells = 2x2 degrees (for spatial splitting)


# ------------------------------------------------- HELPER FOR FIRMS
def calculate_cell_id(lat, lon):
    """Returns the ID of the cell into which a coordinate falls.
    Used to assign FIRMS detections to a cell."""
    row = math.floor((lat - ORIGIN_LAT) / STEP)
    col = math.floor((lon - ORIGIN_LON) / STEP)
    return f"r{row:02d}_c{col:03d}"


def run_all_steps():
    # ------------------------------------------------ 1. READ BOUNDARIES
    gdf = gpd.read_file(SHP_PATH)
    print("Columns:", list(gdf.columns))

    # Select Turkey (column name may vary depending on version)
    mask = None
    for col, val in [("admin", "Turkey"), ("iso_a2", "TR"), ("adm0_a3", "TUR")]:
        if col in gdf.columns:
            m = gdf[col] == val
            if m.sum() > 0:
                mask = m
                print(f"Filter: {col} == {val}")
                break
    if mask is None:
        raise SystemExit("Turkey not found. Manually inspect values in columns "
                         "(e.g., gdf['admin'].unique()).")

    tr = gdf[mask].to_crs(epsg=4326).copy()
    print("Number of provinces:", len(tr), "(expected 81)")

    province_col = "name" if "name" in tr.columns else "name_en"
    tr = tr[[province_col, "geometry"]].rename(columns={province_col: "province"})

    # ------------------------------------------------ 2. GENERATE GRID
    minx, miny, maxx, maxy = tr.total_bounds
    row_min = math.floor((miny - ORIGIN_LAT) / STEP)
    row_max = math.floor((maxy - ORIGIN_LAT) / STEP)
    col_min = math.floor((minx - ORIGIN_LON) / STEP)
    col_max = math.floor((maxx - ORIGIN_LON) / STEP)

    rows = np.arange(row_min, row_max + 1)
    cols = np.arange(col_min, col_max + 1)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    R, C = R.ravel(), C.ravel()

    df = pd.DataFrame({"row": R, "col": C})
    df["lat_min"] = ORIGIN_LAT + df["row"] * STEP
    df["lat_max"] = df["lat_min"] + STEP
    df["lon_min"] = ORIGIN_LON + df["col"] * STEP
    df["lon_max"] = df["lon_min"] + STEP
    df["lat"] = df["lat_min"] + STEP / 2
    df["lon"] = df["lon_min"] + STEP / 2
    print("Total candidate cells in bounding box:", len(df))

    # ------------------------------------------------ 3. LAND FILTER + PROVINCE
    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    merged = gpd.sjoin(points, tr, how="inner", predicate="within")
    merged = merged.drop(columns=["index_right", "geometry"])
    # Points falling on borders may match two provinces, deduplicate
    merged = merged.drop_duplicates(subset=["row", "col"])

    # ------------------------------------------------ 4. ID AND BLOCK
    merged["cell_id"] = [f"r{r:02d}_c{c:03d}"
                         for r, c in zip(merged["row"], merged["col"])]
    merged["block_id"] = [f"b{r // BLOCK_SIZE:02d}_{c // BLOCK_SIZE:02d}"
                          for r, c in zip(merged["row"], merged["col"])]

    col_order = ["cell_id", "lat", "lon", "province", "block_id",
                 "lat_min", "lat_max", "lon_min", "lon_max"]
    output = merged[col_order].sort_values("cell_id").reset_index(drop=True)
    for c in ["lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max"]:
        output[c] = output[c].round(4)

    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ------------------------------------------------ 5. CHECKS
    print("\n--- CHECKS ---")
    print("Number of land cells:", len(output), "(expected ~1200-1400)")
    print("Is cell_id unique:", output["cell_id"].is_unique)
    print("Missing provinces:", output["province"].isna().sum())
    empty_provinces = set(tr["province"]) - set(output["province"])
    print("Provinces with no cells:", sorted(empty_provinces))
    print("Cells per province (top 10):")
    print(output["province"].value_counts().head(10))

    # Map
    plt.figure(figsize=(12, 5))
    plt.scatter(output["lon"], output["lat"], s=6)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title(f"{len(output)} cells")
    plt.gca().set_aspect("equal")
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_CSV}, {OUTPUT_PNG}")


if __name__ == "__main__":
    run_all_steps()

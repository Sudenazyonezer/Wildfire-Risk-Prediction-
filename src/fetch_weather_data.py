import os
import time
import pandas as pd
import requests

CELLS_FILE = "hucreler.csv"
OUTPUT_FILE = "cells_weather_data_2023_2025.csv"
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

if not os.path.exists(CELLS_FILE):
    raise FileNotFoundError(f"'{CELLS_FILE}' not found! Please make sure it is in the project folder.")

df_cells = pd.read_csv(CELLS_FILE)
total_cells = len(df_cells)

def fetch_cell_weather(cell_id, lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "dew_point_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "et0_fao_evapotranspiration",
            "soil_moisture_0_to_7cm_mean",
            "soil_moisture_7_to_28cm_mean",
            "vapor_pressure_deficit_max"
        ],
        "timezone": "auto"
    }

    # Smart retry loop that never gives up on rate limits
    wait_times = [30, 60, 90, 120]
    attempt = 0

    while True:
        try:
            res = requests.get(url, params=params, timeout=45)
            if res.status_code == 200:
                data = res.json()["daily"]
                df = pd.DataFrame(data)
                df["cell_id"] = cell_id
                df["latitude"] = lat
                df["longitude"] = lon
                return df
            elif res.status_code == 429:
                wait = wait_times[min(attempt, len(wait_times) - 1)]
                print(f"   [Rate Limit - 429] Server busy, waiting {wait} seconds...")
                time.sleep(wait)
                attempt += 1
            else:
                print(f"   Server Error ({res.status_code}), waiting 10 seconds...")
                time.sleep(10)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print("   [Connection/Timeout] Retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            print(f"   Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    # Check already downloaded cells from the output file
    completed_cells = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE, usecols=["cell_id"])
            completed_cells = set(df_existing["cell_id"].dropna().unique())
            print(f"Info: Found {len(completed_cells)} already downloaded cells, skipping them.")
        except Exception:
            pass

    print(f"Data collection in progress ({START_DATE} - {END_DATE})...\n")

    for idx, row in df_cells.iterrows():
        cell_id = row["hucre_id"]
        lat = row["enlem"]
        lon = row["boylam"]
        province = row.get("il", "")

        # Skip if already downloaded
        if cell_id in completed_cells:
            continue

        print(f"[{idx + 1}/{total_cells}] Fetching cell: {cell_id} ({province} - {lat}, {lon})...")
        df_cell = fetch_cell_weather(cell_id, lat, lon, START_DATE, END_DATE)

        if df_cell is not None:
            df_cell.rename(columns={
                "time": "date",
                "temperature_2m_max": "temp_max",
                "temperature_2m_mean": "temp_mean",
                "relative_humidity_2m_mean": "humidity_mean",
                "dew_point_2m_mean": "dew_point_mean",
                "precipitation_sum": "precipitation_sum",
                "wind_speed_10m_max": "wind_speed_max",
                "wind_gusts_10m_max": "wind_gusts_max",
                "et0_fao_evapotranspiration": "evapotranspiration_et0",
                "soil_moisture_0_to_7cm_mean": "soil_moisture_surface",
                "soil_moisture_7_to_28cm_mean": "soil_moisture_root_zone",
                "vapor_pressure_deficit_max": "vpd_max"
            }, inplace=True)

            df_cell["province"] = province
            df_cell["block_id"] = row.get("blok_id", "")

            is_first = not os.path.exists(OUTPUT_FILE)
            df_cell.to_csv(OUTPUT_FILE, mode="a", header=is_first, index=False)
            completed_cells.add(cell_id)
            print(f"   -> {cell_id} saved successfully.")

        # Stable 3.5-second sleep between requests
        time.sleep(3.5)

    print(f"\nAll {total_cells} cells completed! File: '{OUTPUT_FILE}'")

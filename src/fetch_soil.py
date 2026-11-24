"""
fetch_soil.py
Fetches soil/agro-meteorological data from the Open-Meteo Archive API for
2023-01-01 .. 2025-12-31, aggregates it to daily values and appends every
point to ONE file (soil_2023_2025.csv). If the run is interrupted (or the daily
limit is hit), running it again resumes: points already in the file are skipped.

Usage:
    pip install requests pandas
    python fetch_soil.py --test 2        # try the first 2 points only
    python fetch_soil.py                 # fetch all points (resumable)
    python fetch_soil.py --check         # sanity checks on the single file
    python fetch_soil.py --merge-legacy  # only if you have old per-point files
"""

import argparse
import os
import time

import pandas as pd
import requests

# ============================================================== TO BE FILLED IN
# Option A (recommended for all of Türkiye): read the points from the grid file
# produced by generate_grid.py. It needs the columns: cell_id, lat, lon.
POINTS_CSV = "cells.csv"   # set to "" to use the manual list below instead

# Option B: manual list of points: (point_id, latitude, longitude)
# Example:  ("point_01", 37.0, 28.0),
POINTS = [
    # only used when POINTS_CSV is ""
]

# API key: leave empty for the free tier. If you get a key, put it here.
API_KEY = ""
# =============================================================================

OUTPUT_DIR = "data/raw/soil"   # only used by --merge-legacy
MERGED_CSV = "soil_2023_2025.csv"
FAILED_LIST = "data/raw/soil_failed.txt"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Istanbul"  # must match what the weather-data teammates use

# Different endpoints for free and keyed (commercial) usage.
# If you use a key, verify the endpoint in the Open-Meteo documentation.
URL_FREE = "https://archive-api.open-meteo.com/v1/archive"
URL_KEYED = "https://customer-archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "soil_moisture_0_to_7cm",
    "soil_moisture_28_to_100cm",
    "soil_temperature_0_to_7cm",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
]

# Output column names (change the right-hand side if the team agrees on others)
COLUMNS = {
    "point_id": "point_id",
    "lat": "latitude",
    "lon": "longitude",
    "date": "date",
    "soil_moisture_surface": "soil_moisture_surface",
    "soil_moisture_root": "soil_moisture_root",
    "soil_temperature": "soil_temperature",
    "vpd": "vpd",
    "et0": "et0",
    "elevation": "elevation",
}

SLEEP_BETWEEN_REQUESTS = 1.0  # seconds
MAX_ATTEMPTS = 4              # attempts per point
RATE_LIMIT_WAIT = 65          # seconds to wait when the per-minute limit is hit


class DailyLimitError(Exception):
    pass


def to_daily(response_json):
    """Converts the hourly API response into a daily table.
    Soil moisture, soil temperature and VPD: daily mean. ET0: daily sum."""
    hourly = pd.DataFrame(response_json["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.strftime("%Y-%m-%d")

    groups = hourly.groupby("date")
    daily = pd.DataFrame({
        COLUMNS["soil_moisture_surface"]: groups["soil_moisture_0_to_7cm"].mean(),
        COLUMNS["soil_moisture_root"]: groups["soil_moisture_28_to_100cm"].mean(),
        COLUMNS["soil_temperature"]: groups["soil_temperature_0_to_7cm"].mean(),
        COLUMNS["vpd"]: groups["vapour_pressure_deficit"].mean(),
        # min_count=1: a fully empty day becomes NaN instead of 0
        COLUMNS["et0"]: groups["et0_fao_evapotranspiration"].sum(min_count=1),
    }).reset_index().rename(columns={"date": COLUMNS["date"]})
    daily[COLUMNS["elevation"]] = response_json.get("elevation")
    return daily


def fetch_point(lat, lon):
    """Requests one point and retries on errors."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }
    url = URL_FREE
    if API_KEY:
        params["apikey"] = API_KEY
        url = URL_KEYED

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            try:
                reason = r.json().get("reason", "")
            except Exception:
                reason = r.text[:200]
            if r.status_code == 429:
                if "daily" in reason.lower():
                    raise DailyLimitError(reason)
                print(f"   rate limit (429): {reason} -> waiting {RATE_LIMIT_WAIT} s")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print(f"   HTTP {r.status_code}: {reason}")
                time.sleep(5 * attempt)
            last_error = f"HTTP {r.status_code}: {reason}"
        except DailyLimitError:
            raise
        except requests.RequestException as e:
            last_error = str(e)
            print(f"   connection error: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(last_error)


def append_to_csv(daily, attempts=15, wait=20):
    """Appends one point's rows to the single output file. If the file is locked
    (typically open in Excel or being synced), waits and retries; if it still
    fails, stops with a clear message (running again resumes safely)."""
    for attempt in range(1, attempts + 1):
        write_header = not (os.path.exists(MERGED_CSV)
                            and os.path.getsize(MERGED_CSV) > 0)
        try:
            daily.to_csv(MERGED_CSV, mode="a", header=write_header, index=False)
            return
        except PermissionError:
            print(f"   cannot write {MERGED_CSV} (is it open in Excel or another "
                  f"program?). Close it. Retry {attempt}/{attempts} in {wait} s")
            time.sleep(wait)
    raise SystemExit(f"Could not write {MERGED_CSV} after {attempts} attempts. "
                     "Close any program using it and run the script again; "
                     "it will resume.")


def load_points():
    """Returns a list of (point_id, latitude, longitude)."""
    if POINTS_CSV:
        if not os.path.exists(POINTS_CSV):
            raise SystemExit(f"{POINTS_CSV} not found. Run generate_grid.py first "
                             "or set POINTS_CSV to the right path.")
        grid = pd.read_csv(POINTS_CSV)
        return list(zip(grid["cell_id"], grid["lat"], grid["lon"]))
    return list(POINTS)


def fetch_all(test_count=None):
    all_points = load_points()
    if not all_points:
        raise SystemExit("No points to fetch. Set POINTS_CSV to the grid file "
                         "or add (point_id, latitude, longitude) rows to POINTS.")
    points = all_points[:test_count] if test_count else all_points

    # Points already in the single output file are skipped (resume support)
    done = set()
    if os.path.exists(MERGED_CSV) and os.path.getsize(MERGED_CSV) > 0:
        done = set(pd.read_csv(MERGED_CSV,
                               usecols=[COLUMNS["point_id"]])[COLUMNS["point_id"]])
        print(f"{len(done)} points already in {MERGED_CSV}, they will be skipped.")

    total = len(points)
    failures = []
    for i, (pid, lat, lon) in enumerate(points, start=1):
        if pid in done:
            continue
        t0 = time.time()
        try:
            daily = to_daily(fetch_point(lat, lon))
        except DailyLimitError as e:
            print(f"\nDaily limit reached: {e}\n"
                  "Run the script again tomorrow; it resumes where it stopped.")
            break
        except Exception as e:
            print(f"[{i}/{total}] {pid} FAILED: {e}")
            failures.append(f"{pid}\t{e}")
            continue

        # The coordinates you provided are written, not the ones the API returns
        daily.insert(0, COLUMNS["lon"], lon)
        daily.insert(0, COLUMNS["lat"], lat)
        daily.insert(0, COLUMNS["point_id"], pid)
        append_to_csv(daily)
        print(f"[{i}/{total}] {pid} done "
              f"({len(daily)} days, {time.time() - t0:.1f} s)")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if failures:
        os.makedirs(os.path.dirname(FAILED_LIST), exist_ok=True)
        with open(FAILED_LIST, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"\n{len(failures)} points failed, list: {FAILED_LIST}. "
              "Running again retries only the missing ones.")


def report(df):
    """Prints sanity checks for the fetched data."""
    pid, date = COLUMNS["point_id"], COLUMNS["date"]
    value_columns = [COLUMNS[k] for k in
                     ["soil_moisture_surface", "soil_moisture_root",
                      "soil_temperature", "vpd", "et0", "elevation"]]
    print("--- CHECKS ---")
    print("Number of points:", df[pid].nunique())
    print("Number of rows:", len(df))
    print("min date:", df[date].min(), "| max date:", df[date].max())
    days = df.groupby(pid)[date].nunique()
    print("Days per point (expected 1096): min", days.min(), "max", days.max())
    print("Duplicate (point, date) rows:", int(df.duplicated([pid, date]).sum()))
    print("Missing ratios:")
    print(df.isna().mean().round(4))
    print(df[value_columns].describe().loc[["min", "max"]])


def check():
    """Checks the single output file."""
    if not os.path.exists(MERGED_CSV):
        raise SystemExit(f"{MERGED_CSV} not found yet.")
    report(pd.read_csv(MERGED_CSV))


def merge_legacy():
    """Only needed if you already have per-point files from an earlier version:
    merges them into the single output file. Afterwards you can delete the folder;
    running the script again continues by appending to the single file."""
    if os.path.exists(MERGED_CSV) and os.path.getsize(MERGED_CSV) > 0:
        raise SystemExit(f"{MERGED_CSV} already exists, not overwriting it.")
    if not os.path.isdir(OUTPUT_DIR):
        raise SystemExit("No per-point files found.")
    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv"))
    if not files:
        raise SystemExit("No per-point files found.")
    df = pd.concat((pd.read_csv(os.path.join(OUTPUT_DIR, f)) for f in files),
                   ignore_index=True)
    df.to_csv(MERGED_CSV, index=False)
    print(f"Merged {len(files)} files into {MERGED_CSV}")
    report(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, help="fetch only the first N points")
    ap.add_argument("--check", action="store_true",
                    help="run sanity checks on the single output file")
    ap.add_argument("--merge-legacy", action="store_true",
                    help="merge old per-point files into the single output file")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.merge_legacy:
        merge_legacy()
    else:
        fetch_all(args.test)

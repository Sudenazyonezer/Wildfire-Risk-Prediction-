"""
fetch_firms.py
Fetches VIIRS active-fire detections from the NASA FIRMS area API for a bounding
box around Türkiye, 2023-01-01 .. 2025-12-31, in small date chunks, and appends
everything to ONE raw file (firms_raw_2023_2025.csv). Resumable: finished chunks
are recorded in firms_progress.txt and skipped on the next run.

Usage:
    pip install requests pandas
    set the key in your terminal (do NOT paste it into a file you commit):
        PowerShell:  $env:FIRMS_MAP_KEY="your_key_here"
    python fetch_firms.py --availability   # which dates does each source cover?
    python fetch_firms.py --test 3         # fetch only the first 3 chunks
    python fetch_firms.py                  # fetch everything (resumable)
    python fetch_firms.py --check          # sanity checks on the raw file

The raw file still contains the whole bounding box (it also covers parts of
neighbouring countries and sea). Filtering to Turkish grid cells, removing
low-confidence detections and fixed industrial sources is a separate next step.
"""
#35f9756268a3d7c4e85685b8c8fe32b6
import argparse
import datetime as dt
import io
import os
import time

import pandas as pd
import requests

# ================================================================== SETTINGS
# The key is read from the environment so it never ends up in the code/repo.
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")

# Satellite/product. VIIRS NOAA-20 standard processing covers 2020-present, so it
# covers the whole 2023-2025 window with ONE satellite. (Suomi NPP stops new
# deliveries on 2026-11-02; its history stays available, but NOAA-20 is the
# safer single-satellite choice. NOAA-21 only starts in 2024.)
SOURCE = "VIIRS_NOAA20_SP"

# west, south, east, north (a box around Türkiye)
BBOX = "25.5,35.5,45.0,42.5"

START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date(2025, 12, 31)

# Days per request. The FIRMS area page says 1..10, some sources report 1..5.
# 5 is safe under both; raise to 10 only if the API accepts it.
DAY_RANGE = 5
# ===========================================================================

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"
RAW_CSV = "firms_raw_2023_2025.csv"
PROGRESS_FILE = "firms_progress.txt"
FAILED_LIST = "firms_failed.txt"

SLEEP_BETWEEN_REQUESTS = 2.0   # seconds
MAX_ATTEMPTS = 4
LIMIT_WAIT = 600               # seconds; the transaction budget is per 10 minutes
EXPECTED_HEADER_START = "latitude,longitude"


def safe(text):
    """Removes the key from any message before printing/saving it."""
    return str(text).replace(MAP_KEY, "***") if MAP_KEY else str(text)


def get_text(url):
    """GET request; returns (status_code, body_text). The key never leaves here."""
    try:
        r = requests.get(url, timeout=120)
        return r.status_code, r.text
    except requests.RequestException as e:
        return None, safe(e)


def chunk_starts():
    day = START_DATE
    while day <= END_DATE:
        yield day
        day += dt.timedelta(days=DAY_RANGE)


def load_done():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(start):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(start.isoformat() + "\n")


def fetch_chunk(start):
    """Returns a DataFrame (possibly empty) for [start, start+days-1]."""
    days = min(DAY_RANGE, (END_DATE - start).days + 1)
    url = (f"{BASE_URL}/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{days}/"
           f"{start.isoformat()}")
    last_problem = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        status, body = get_text(url)
        text = (body or "").strip()

        if status == 200 and text.startswith(EXPECTED_HEADER_START):
            return pd.read_csv(io.StringIO(text))
        if status == 200 and text == "":
            return pd.DataFrame()          # no detections in this chunk

        low = text.lower()
        if "invalid" in low and "map_key" in low.replace(" ", "_"):
            raise SystemExit("FIRMS says the MAP_KEY is invalid. Check the "
                             "FIRMS_MAP_KEY environment variable.")
        last_problem = f"HTTP {status}: {safe(text[:200])}"
        if "transaction" in low or "limit" in low or "exceed" in low:
            print(f"   transaction limit: {safe(text[:120])} -> waiting "
                  f"{LIMIT_WAIT // 60} min")
            time.sleep(LIMIT_WAIT)
        else:
            print(f"   problem ({last_problem}), retry {attempt}/{MAX_ATTEMPTS}")
            time.sleep(5 * attempt)
    raise RuntimeError(last_problem)


def append_to_csv(df, attempts=15, wait=20):
    """Appends rows to the raw file; retries if the file is locked (e.g. Excel)."""
    for attempt in range(1, attempts + 1):
        header = not (os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0)
        try:
            df.to_csv(RAW_CSV, mode="a", header=header, index=False)
            return
        except PermissionError:
            print(f"   cannot write {RAW_CSV} (open in Excel or another program?)."
                  f" Close it. Retry {attempt}/{attempts} in {wait} s")
            time.sleep(wait)
    raise SystemExit(f"Could not write {RAW_CSV}. Close any program using it and "
                     "run the script again; it will resume.")


def fetch_all(test_chunks=None):
    if not MAP_KEY:
        raise SystemExit("MAP_KEY is empty. In PowerShell run:\n"
                         '  $env:FIRMS_MAP_KEY="your_key_here"\n'
                         "then start the script again in the same window.")
    starts = list(chunk_starts())
    if test_chunks:
        starts = starts[:test_chunks]
    done = load_done()
    if done:
        print(f"{len(done)} chunks already done, they will be skipped.")

    total, failures, rows = len(starts), [], 0
    for i, start in enumerate(starts, start=1):
        if start.isoformat() in done:
            continue
        t0 = time.time()
        try:
            df = fetch_chunk(start)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{i}/{total}] {start} FAILED: {safe(e)}")
            failures.append(f"{start}\t{safe(e)}")
            continue
        if len(df):
            df["source"] = SOURCE
            append_to_csv(df)
        mark_done(start)
        rows += len(df)
        print(f"[{i}/{total}] {start} done ({len(df)} detections, "
              f"{time.time() - t0:.1f} s)")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if failures:
        with open(FAILED_LIST, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"\n{len(failures)} chunks failed, list: {FAILED_LIST}. "
              "Running again retries only the missing ones.")
    print(f"\nNew detections this run: {rows}")


def availability():
    """Shows which dates each source covers (does not consume a map transaction
    in a meaningful way)."""
    if not MAP_KEY:
        raise SystemExit("Set FIRMS_MAP_KEY first (see the top of this file).")
    status, body = get_text(f"{BASE_URL}/data_availability/csv/{MAP_KEY}/ALL")
    print(f"HTTP {status}")
    print(safe(body))
    print(f"\nThe script currently uses SOURCE = {SOURCE}. Its date range must "
          f"contain {START_DATE} .. {END_DATE}.")


def check():
    if not os.path.exists(RAW_CSV):
        raise SystemExit(f"{RAW_CSV} not found yet.")
    df = pd.read_csv(RAW_CSV)
    print("--- CHECKS ---")
    print("Rows:", len(df))
    print("min acq_date:", df["acq_date"].min(), "| max acq_date:",
          df["acq_date"].max())
    print("Detections per year:")
    print(df["acq_date"].str[:4].value_counts().sort_index())
    print("Confidence values:", df["confidence"].value_counts().to_dict())
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    print("lat range:", df["latitude"].min(), df["latitude"].max(),
          "| lon range:", df["longitude"].min(), df["longitude"].max())
    done = load_done()
    expected = len(list(chunk_starts()))
    print(f"Finished chunks: {len(done)} / {expected}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, help="fetch only the first N chunks")
    ap.add_argument("--availability", action="store_true",
                    help="show the date coverage of the FIRMS sources")
    ap.add_argument("--check", action="store_true",
                    help="run sanity checks on the raw file")
    args = ap.parse_args()
    if args.availability:
        availability()
    elif args.check:
        check()
    else:
        fetch_all(args.test)
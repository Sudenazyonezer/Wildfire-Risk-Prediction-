# config.py  (Written by Person A, imported by B and C)

RANDOM_STATE = 42

# --- Files ---
MASTER_FILE = "cleaned_master_table_v2.csv"
FEATURES_FILE = "features_table.csv"
TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

# --- Target and columns EXCLUDED from the model ---
TARGET = "fire_occurrence"
ID_COLS = ["cell_id", "block_id", "province"]      # identifier
DATE_COL = "date"                                   # for splitting only
LEAK_COLS = ["fire_count", "max_frp"]               # derived from target
DROP_COLS = ID_COLS + [DATE_COL] + LEAK_COLS

# --- Time-based split ---
# Tuning phase: 2023 train, 2024 val
TUNE_TRAIN = ("2023-01-01", "2023-12-31")
TUNE_VAL = ("2024-01-01", "2024-12-31")
# Final: 2023-2024 train, 2025 test
FINAL_TRAIN = ("2023-01-01", "2024-12-31")
FINAL_TEST = ("2025-01-01", "2025-12-31")

# --- Features ---
BASE_WEATHER = [
    "temp_max", "temp_mean", "rel_humidity_mean", "dew_point_mean",
    "precip_sum", "wind_speed_max", "wind_gust_max", "et0",
    "soil_moisture_surface", "soil_moisture_root", "vpd", "soil_temperature",
]
STATIC = ["elevation"]
LATLON = ["lat", "lon"]

ROLL_COLS = ["temp_mean", "precip_sum", "rel_humidity_mean", "vpd", "soil_moisture_root"]
ROLL_WINDOWS = [3, 7, 14, 30]
LAG_COLS = ["vpd", "temp_max", "rel_humidity_mean", "precip_sum", "soil_moisture_root"]
LAG_STEPS = [1, 3, 7]

def _roll_name(col, n):
    return f"{col}_roll{n}_{'sum' if col == 'precip_sum' else 'mean'}"

ROLLING = [_roll_name(c, n) for c in ROLL_COLS for n in ROLL_WINDOWS]
LAGS = [f"{c}_lag{n}" for c in LAG_COLS for n in LAG_STEPS]
DROUGHT = ["days_since_rain"]
TIME = ["month_sin", "month_cos", "doy_sin", "doy_cos"]

ENGINEERED = ROLLING + LAGS + DROUGHT + TIME

# For tree models (RF, XGBoost/LightGBM)
FEATURES_NO_LATLON = BASE_WEATHER + STATIC + ENGINEERED      # main set
FEATURES_WITH_LATLON = FEATURES_NO_LATLON + LATLON           # location comparison

# Simplified set for logistic regression (reduces multicollinearity)
FEATURES_LOGREG = [
    "temp_max", "rel_humidity_mean", "precip_sum", "wind_speed_max",
    "soil_moisture_root", "elevation", "days_since_rain",
    "vpd_roll7_mean", "vpd_roll30_mean", "precip_sum_roll30_sum",
    "soil_moisture_root_roll14_mean", "month_sin", "month_cos",
]

FEATURES = FEATURES_NO_LATLON   # default
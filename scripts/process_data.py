import os
import sys
import hashlib
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_snake(s: str) -> str:
    s = s.strip().replace("/", "_").replace("-", "_").replace(" ", "_").lower()
    while "__" in s:
        s = s.replace("__", "_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    df = fill_missing_with_flags(df)
    return df


def hashed(value: str, salt: str) -> str:
    if pd.isna(value):
        return value
    h = hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest()
    return h[:16]


def anonymize_columns(df: pd.DataFrame, cols, salt: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).apply(lambda v: hashed(v, salt))
    return df


def parse_datetimes(df: pd.DataFrame, datetime_cols) -> pd.DataFrame:
    df = df.copy()
    for c in datetime_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def fill_missing_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            was_missing_col = f"{c}_was_missing"
            df[was_missing_col] = df[c].isna().astype(int)
            if df[c].notna().any():
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            # Leave NaT as-is; downstream time-based operations can handle
            pass
        else:
            was_missing_col = f"{c}_was_missing"
            df[was_missing_col] = df[c].isna().astype(int)
            df[c] = df[c].fillna("unknown")
    return df


def remove_dupes_and_sort(df: pd.DataFrame, by_cols) -> pd.DataFrame:
    df = df.copy()
    if by_cols:
        existing = [c for c in by_cols if c in df.columns]
        if existing:
            df = df.drop_duplicates(subset=existing)
            df = df.sort_values(by=existing)
        else:
            df = df.drop_duplicates()
    else:
        df = df.drop_duplicates()
    return df


def cap_outliers(df: pd.DataFrame, cols, z_thresh: float = 4.0) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = df[c].astype(float)
            m, sd = s.mean(), s.std(ddof=0)
            if pd.isna(sd) or sd == 0:
                continue
            z = (s - m) / sd
            df[c] = np.where(z > z_thresh, m + z_thresh * sd,
                     np.where(z < -z_thresh, m - z_thresh * sd, s))
    return df


def nearest_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, tolerance: str = "60min") -> pd.DataFrame:
    if on not in left.columns or on not in right.columns:
        return left
    l = left.copy().sort_values(on)
    r = right.copy().sort_values(on)
    merged = pd.merge_asof(l, r, on=on, direction="nearest", tolerance=pd.Timedelta(tolerance))
    return merged


# -----------------------------
# Loaders
# -----------------------------

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_DIR = os.path.join(DATA_DIR, "output", "processed")
ensure_dir(OUT_DIR)

SALT = os.environ.get("DPWORLD_ANON_SALT", "dpworld_default_salt_change_me")


def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        df = normalize_columns(df)
        if parse_dates:
            df = parse_datetimes(df, parse_dates)
        df = clean_strings(df)
        return df
    except FileNotFoundError:
        print(f"WARN: File not found: {path}")
        return pd.DataFrame()


# -----------------------------
# Domain processing
# -----------------------------

def process_ais(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Common AIS fields guess
    dt_cols = [c for c in ["timestamp", "time", "datetime", "event_time"] if c in df.columns]
    id_cols = [c for c in ["mmsi", "imo", "vessel_id", "vessel"] if c in df.columns]

    # Ensure datetime
    df = parse_datetimes(df, dt_cols)
    main_time = dt_cols[0] if dt_cols else None

    # Numeric clean
    num_candidates = ["sog", "cog", "lat", "lon", "draft"]
    for c in num_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Outlier capping for speed, draft
    df = cap_outliers(df, [c for c in ["sog", "draft"] if c in df.columns])

    # Anonymize IDs
    df = anonymize_columns(df, id_cols, SALT)

    # Keep essential
    keep = list(dict.fromkeys(([main_time] if main_time else []) + id_cols + [c for c in num_candidates if c in df.columns]))
    keep = [c for c in keep if c in df.columns]
    df = remove_dupes_and_sort(df[keep], by_cols=[main_time] + id_cols if main_time else id_cols)
    df = fill_missing_with_flags(df)
    return df


def process_port_calls(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Expected columns (various aliases)
    arrival = next((c for c in ["ata", "arrival_time", "arrived", "arrival"] if c in df.columns), None)
    berth_start = next((c for c in ["berth_start", "atb", "berthed", "start_berth"] if c in df.columns), None)
    departure = next((c for c in ["atd", "departure_time", "departed", "departure"] if c in df.columns), None)
    call_id = next((c for c in ["port_call_id", "call_id", "visit_id"] if c in df.columns), None)
    vessel_cols = [c for c in ["mmsi", "imo", "vessel_id", "vessel"] if c in df.columns]

    df = parse_datetimes(df, [c for c in [arrival, berth_start, departure] if c])

    # Compute durations
    if arrival and berth_start:
        df["berth_wait_hrs"] = (df[berth_start] - df[arrival]).dt.total_seconds() / 3600.0
    if berth_start and departure:
        df["service_time_hrs"] = (df[departure] - df[berth_start]).dt.total_seconds() / 3600.0
    if arrival and departure:
        df["turnaround_hrs"] = (df[departure] - df[arrival]).dt.total_seconds() / 3600.0

    # Sanity caps
    df = cap_outliers(df, [c for c in ["berth_wait_hrs", "service_time_hrs", "turnaround_hrs"] if c in df.columns])

    # Anonymize
    df = anonymize_columns(df, vessel_cols, SALT)

    keep = list(dict.fromkeys(([call_id] if call_id else []) + vessel_cols + [arrival, berth_start, departure] + [c for c in ["berth_wait_hrs", "service_time_hrs", "turnaround_hrs"] if c in df.columns]))
    keep = [c for c in keep if c in df.columns]
    by = [arrival] if arrival else ( [berth_start] if berth_start else ( [departure] if departure else []))
    df = remove_dupes_and_sort(df[keep], by_cols=by + ([call_id] if call_id else []) )
    df = fill_missing_with_flags(df)
    return df


def process_cranes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Expect: timestamp, moves, cranes_active, berth_id
    df = parse_datetimes(df, [c for c in ["timestamp", "time", "event_time"] if c in df.columns])
    for c in ["moves", "cranes_active", "throughput_tph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Rolling congestion proxy
    time_col = next((c for c in ["timestamp", "time", "event_time"] if c in df.columns), None)
    if time_col:
        df = df.sort_values(time_col)
        for w in [24, 48, 72]:
            if "moves" in df.columns:
                df[f"moves_roll_{w}h"] = df["moves"].rolling(f"{w}h", on=time_col).mean()
            if "cranes_active" in df.columns:
                df[f"cranes_active_roll_{w}h"] = df["cranes_active"].rolling(f"{w}h", on=time_col).mean()
    # Utilization ratio if capacity available
    capacity_col = next((c for c in ["crane_capacity", "max_cranes", "planned_cranes"] if c in df.columns), None)
    if capacity_col and "cranes_active" in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["crane_utilization_ratio"] = (df["cranes_active"].astype(float) / pd.to_numeric(df[capacity_col], errors="coerce")).clip(0, 1)
    df = fill_missing_with_flags(df)
    return df


def process_yard_gate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = parse_datetimes(df, [c for c in ["timestamp", "time"] if c in df.columns])
    for c in ["gate_in", "gate_out", "queue_len", "avg_wait_min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    time_col = next((c for c in ["timestamp", "time"] if c in df.columns), None)
    if time_col:
        df = df.sort_values(time_col)
        for w in [24, 48, 72]:
            for c in ["gate_in", "gate_out", "queue_len", "avg_wait_min"]:
                if c in df.columns:
                    df[f"{c}_roll_{w}h"] = df[c].rolling(f"{w}h", on=time_col).mean()
    # Utilization ratios
    yard_occ = next((c for c in ["yard_occupied_teu", "yard_occupied", "yard_used"] if c in df.columns), None)
    yard_cap = next((c for c in ["yard_capacity_teu", "yard_capacity"] if c in df.columns), None)
    if yard_occ and yard_cap:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["yard_utilization_ratio"] = (pd.to_numeric(df[yard_occ], errors="coerce") / pd.to_numeric(df[yard_cap], errors="coerce")).clip(0, 1)
    if "gate_in" in df.columns or "gate_out" in df.columns:
        gate_flow = df.get("gate_in", 0).astype(float) + df.get("gate_out", 0).astype(float)
        if "gate_capacity_per_hour" in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                df["gate_utilization_ratio"] = (gate_flow / pd.to_numeric(df["gate_capacity_per_hour"], errors="coerce")).clip(lower=0)
    # Congestion indicator (normalized)
    if "queue_len" in df.columns:
        q = pd.to_numeric(df["queue_len"], errors="coerce")
        df["congestion_index"] = (q - q.mean()) / (q.std(ddof=0) if q.std(ddof=0) else 1.0)
    elif "avg_wait_min" in df.columns:
        w = pd.to_numeric(df["avg_wait_min"], errors="coerce")
        df["congestion_index"] = (w - w.mean()) / (w.std(ddof=0) if w.std(ddof=0) else 1.0)
    df = fill_missing_with_flags(df)
    return df


def process_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = parse_datetimes(df, [c for c in ["timestamp", "time", "obs_time"] if c in df.columns])
    for c in ["wind_speed", "gust", "wave_height", "visibility", "precip_mm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = cap_outliers(df, [c for c in ["wind_speed", "gust", "wave_height", "visibility", "precip_mm"] if c in df.columns])
    return df


def process_cost_emissions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ["cost_per_hour", "co2_per_hour_kg", "sox_per_hour_kg", "nox_per_hour_kg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = fill_missing_with_flags(df)
    return df


# -----------------------------
# Merge and export
# -----------------------------

def export(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return
    base = os.path.join(OUT_DIR, name)
    # Save parquet and csv
    try:
        df.to_parquet(base + ".parquet", index=False)
    except Exception as e:
        print(f"WARN: Failed parquet for {name}: {e}")
    try:
        df.to_csv(base + ".csv", index=False)
    except Exception as e:
        print(f"WARN: Failed csv for {name}: {e}")


def build_time_aligned_dataset(port_calls: pd.DataFrame, cranes: pd.DataFrame, yard: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if port_calls.empty:
        return pd.DataFrame()

    # Choose time column for alignment
    time_col = next((c for c in ["ata", "arrival_time", "arrived", "arrival", "berth_start", "atb", "departure_time", "atd"] if c in port_calls.columns), None)
    if not time_col:
        return pd.DataFrame()

    features = port_calls.copy()

    # Merge nearest with cranes, yard, weather
    if not cranes.empty:
        cranes_time = next((c for c in ["timestamp", "time", "event_time"] if c in cranes.columns), None)
        if cranes_time:
            tmp = cranes.rename(columns={cranes_time: time_col})
            features = nearest_merge(features, tmp, on=time_col, tolerance="2h")

    if not yard.empty:
        yard_time = next((c for c in ["timestamp", "time"] if c in yard.columns), None)
        if yard_time:
            tmp = yard.rename(columns={yard_time: time_col})
            features = nearest_merge(features, tmp, on=time_col, tolerance="2h")

    if not weather.empty:
        weather_time = next((c for c in ["timestamp", "time", "obs_time"] if c in weather.columns), None)
        if weather_time:
            tmp = weather.rename(columns={weather_time: time_col})
            features = nearest_merge(features, tmp, on=time_col, tolerance="3h")

    return features


def main():
    # File paths
    ais_path = os.path.join(DATA_DIR, "ais_tracking.csv")
    cranes_path = os.path.join(DATA_DIR, "berth_crane_operations.csv")
    cost_path = os.path.join(DATA_DIR, "cost_emission.csv")
    port_calls_path = os.path.join(DATA_DIR, "vessel_port_calls.csv")
    weather_path = os.path.join(DATA_DIR, "weather_data.csv")
    yard_path = os.path.join(DATA_DIR, "yard_gate_congestion.csv")

    # Load
    ais_df = read_csv_safe(ais_path)
    cranes_df = read_csv_safe(cranes_path)
    cost_df = read_csv_safe(cost_path)
    port_calls_df = read_csv_safe(port_calls_path)
    weather_df = read_csv_safe(weather_path)
    yard_df = read_csv_safe(yard_path)

    # Process
    ais_p = process_ais(ais_df)
    cranes_p = process_cranes(cranes_df)
    cost_p = process_cost_emissions(cost_df)
    port_calls_p = process_port_calls(port_calls_df)
    weather_p = process_weather(weather_df)
    yard_p = process_yard_gate(yard_df)

    # Export individual processed datasets
    export(ais_p, "ais_processed")
    export(cranes_p, "berth_crane_processed")
    export(cost_p, "cost_emission_processed")
    export(port_calls_p, "port_calls_processed")
    export(weather_p, "weather_processed")
    export(yard_p, "yard_gate_processed")

    # Build time-aligned merged dataset (no targets/estimates)
    merged_df = build_time_aligned_dataset(port_calls_p, cranes_p, yard_p, weather_p)
    export(merged_df, "time_aligned_processed")

    print("Processing complete. Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

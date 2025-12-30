import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_DIR / "output" / "processed"
OUT_PARQUET = PROC_DIR / "ml_features_targets.parquet"
OUT_CSV = PROC_DIR / "ml_features_targets.csv"
OUT_PARQUET_REFINED = PROC_DIR / "ml_features_targets_refined.parquet"
OUT_CSV_REFINED = PROC_DIR / "ml_features_targets_refined.csv"
OUT_PARQUET_REGR_REFINED = PROC_DIR / "ml_features_targets_regression_refined.parquet"
OUT_CSV_REGR_REFINED = PROC_DIR / "ml_features_targets_regression_refined.csv"

BASE_PARQUET = PROC_DIR / "time_aligned_processed.parquet"
BASE_CSV = PROC_DIR / "time_aligned_processed.csv"

# -----------------------------
# Helpers
# -----------------------------

def choose_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "ata", "arrival_time", "arrived", "arrival",
        "berth_start", "atb", "departure_time", "atd",
        "timestamp", "time"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def coerce_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def build_delay_risk_dataset(base: pd.DataFrame, tcol: str) -> pd.DataFrame:
    """
    Step 4.6: Reformulate regression to a categorical delay risk at +24h.
    delay_risk_24h =
      0 if delta<=0.5h, 1 if 0.5<delta<=2.0h, 2 if delta>2.0h
    Uses the same wait base selection as build_refined_targets. Excludes
    wait-related features from X to avoid leakage.
    """
    # Ensure we have refined wait deltas computed
    ref = build_refined_targets(base, tcol)
    if "wait_delta_24h" not in ref.columns:
        return pd.DataFrame()
    df = ref.copy()
    # Target binning
    delta = pd.to_numeric(df["wait_delta_24h"], errors="coerce")
    bins = [-np.inf, 0.5, 2.0, np.inf]
    labels = [0, 1, 2]
    df["delay_risk_24h"] = pd.cut(delta, bins=bins, labels=labels, include_lowest=True).astype("Int64")

    # Features: temporal + operational drivers (exclude wait-related terms)
    df = add_temporal_features(df, tcol)
    # rolling on driver columns only
    roll_driver_cols = [c for c in [
        "yard_utilization_ratio", "crane_utilization_ratio", "avg_truck_wait_min",
        "temperature_c", "wind_speed_mps", "visibility_km", "rainfall_mm", "wave_height_m"
    ] if c in df.columns]
    df = rolling_features(df, tcol, roll_driver_cols, [24, 48, 72])

    # Drop leakage: any columns with 'wait_base', 'berth_wait', 'wait_time', 'wait_delta'
    drop_like = ["_wait_base_hrs", "berth_wait", "wait_time", "wait_delta"]
    extra_drop = [c for c in df.columns if any(p in c for p in drop_like)]
    feat_df, _ = select_non_leaky_features(df, tcol, extra_drop=extra_drop + ["_fut_congestion_index_24h"]) 

    # Attach target (avoid duplicate columns)
    if "delay_risk_24h" in feat_df.columns:
        feat_df = feat_df.drop(columns=["delay_risk_24h"])  # ensure single source of truth
    final = feat_df.merge(df[[tcol, "delay_risk_24h"]], on=tcol, how="left", suffixes=("", "_y"))
    # Consolidate any suffixes
    if "delay_risk_24h" not in final.columns:
        if "delay_risk_24h_y" in final.columns:
            final = final.rename(columns={"delay_risk_24h_y": "delay_risk_24h"})
        elif "delay_risk_24h_x" in final.columns:
            final = final.rename(columns={"delay_risk_24h_x": "delay_risk_24h"})
    # Remove missing targets
    final = final[final["delay_risk_24h"].notna()]
    final = add_time_based_split(final, tcol)
    return final


# -----------------------------
# Refined targets (Step 4.5)
# -----------------------------

def _future_value(df: pd.DataFrame, tcol: str, base_col: str, horizon_h: int) -> pd.Series:
    if tcol not in df.columns or base_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    tmp = (
        df[[tcol, base_col]]
        .sort_values(tcol)
        .set_index(tcol)[base_col]
        .resample("1h").mean().interpolate(limit=6)
        .shift(-horizon_h)
        .to_frame(name=f"_fut_{base_col}_{horizon_h}h")
        .reset_index()
    )
    merged = pd.merge_asof(
        df.sort_values(tcol),
        tmp.sort_values(tcol),
        on=tcol,
        direction="nearest",
        tolerance=pd.Timedelta("3h")
    )
    return merged[f"_fut_{base_col}_{horizon_h}h"]


def build_refined_targets(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    df = df.copy()
    # Choose base for waiting-time deltas
    base_col = None
    for c in ["berth_wait_hrs", "turnaround_hrs", "service_time_hrs", "avg_truck_wait_min"]:
        if c in df.columns:
            base_col = c
            break
    if base_col is None:
        return df
    # Convert minutes to hours for consistency
    if base_col == "avg_truck_wait_min":
        df["_wait_base_hrs"] = pd.to_numeric(df[base_col], errors="coerce") / 60.0
    else:
        df["_wait_base_hrs"] = pd.to_numeric(df[base_col], errors="coerce")

    # Future values
    fut24 = _future_value(df, tcol, "_wait_base_hrs", 24)
    fut48 = _future_value(df, tcol, "_wait_base_hrs", 48)
    fut72 = _future_value(df, tcol, "_wait_base_hrs", 72)

    # Delta targets: future - current (force index alignment)
    df["wait_delta_24h"] = pd.Series(fut24.to_numpy(), index=df.index) - df["_wait_base_hrs"]
    df["wait_delta_48h"] = pd.Series(fut48.to_numpy(), index=df.index) - df["_wait_base_hrs"]
    df["wait_delta_72h"] = pd.Series(fut72.to_numpy(), index=df.index) - df["_wait_base_hrs"]

    # Future congestion state label at +24h (fallbacks if congestion_index missing)
    cong_base_col = None
    for c in ["congestion_index", "yard_utilization_ratio", "avg_truck_wait_min"]:
        if c in df.columns:
            cong_base_col = c
            break
    if cong_base_col is not None:
        fut_name = f"_fut_{cong_base_col}_24h"
        if cong_base_col == "avg_truck_wait_min":
            # Convert to hours to reduce scale; thresholds are z-based so scale matters less
            df[cong_base_col] = pd.to_numeric(df[cong_base_col], errors="coerce") / 60.0
        df[fut_name] = _future_value(df, tcol, cong_base_col, 24)
        s = pd.to_numeric(df[cong_base_col], errors="coerce")
        mean, sd = s.mean(), s.std(ddof=0) if s.std(ddof=0) else 1.0
        bins = [-np.inf, mean - 0.5*sd, mean + 0.5*sd, np.inf]
        labels = ["Low", "Medium", "High"]
        df["future_congestion_level"] = pd.cut(df[fut_name], bins=bins, labels=labels, include_lowest=True)
    return df


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_temporal_features(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    if not tcol or tcol not in df.columns:
        return df
    df = df.copy()
    ts = pd.to_datetime(df[tcol], utc=True)
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["weekday"] = ts.dt.weekday
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def rolling_features(df: pd.DataFrame, tcol: str, num_cols: List[str], windows: List[int]) -> pd.DataFrame:
    if not tcol or tcol not in df.columns:
        return df
    df = df.copy().sort_values(tcol)
    # Use DatetimeIndex for time-based rolling windows
    idx_df = df.set_index(tcol)
    for w in windows:
        win = f"{w}h"
        for c in num_cols:
            if c in idx_df.columns:
                s = pd.to_numeric(idx_df[c], errors="coerce")
                roll_mean = s.rolling(win).mean()
                roll_std = s.rolling(win).std()
                df[f"{c}_roll_mean_{w}h"] = roll_mean.reindex(idx_df.index).values
                df[f"{c}_roll_std_{w}h"] = roll_std.reindex(idx_df.index).values
                # Simple lag of previous observation in time order (not time-aware gap)
                df[f"{c}_lag_{w}h"] = df.sort_values(tcol)[c].shift(1)
    return df


def define_wait_time_targets(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    """
    Define regression targets as future operational waiting time at +24/+48/+72 hours.
    Primary source: berth_wait_hrs. Fallbacks: turnaround_hrs, then service_time_hrs.
    We align by nearest timestamp after shifting the series backward in time
    (so that each row only uses future values for y and current/past for X).
    """
    if not tcol:
        return df
    base_col = None
    for c in ["berth_wait_hrs", "turnaround_hrs", "service_time_hrs", "avg_truck_wait_min"]:
        if c in df.columns:
            base_col = c
            break
    if base_col is None:
        return df
    df = df.copy().sort_values(tcol)
    # Resample to hourly to create a consistent time index for forward lookup
    hourly = df.set_index(tcol)[[base_col]].resample("1H").mean().interpolate(limit=6)
    # If using minutes, convert to hours to keep target naming consistent
    if base_col == "avg_truck_wait_min":
        hourly[base_col] = hourly[base_col] / 60.0
    for h in [24, 48, 72]:
        hourly[f"_tmp_future_{h}h"] = hourly[base_col].shift(-h)
    # Map future values back to original timestamps by nearest merge
    hourly = hourly.reset_index()
    for h in [24, 48, 72]:
        tmp = hourly[[tcol, f"_tmp_future_{h}h"]].rename(columns={f"_tmp_future_{h}h": f"wait_time_{h}h"})
        df = pd.merge_asof(
            df.sort_values(tcol),
            tmp.sort_values(tcol),
            on=tcol,
            direction="nearest",
            tolerance=pd.Timedelta("3H")
        )
    return df


def classify_congestion_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical congestion_level using current-row indicators only.
    Logic: compute a normalized score from available columns, then threshold.
    - score = z(congestion_index) + z(yard_utilization_ratio) + z(berth_wait_hrs)
    Thresholds (robust):
      Low: score <= 0
      Medium: 0 < score <= 1.0
      High: score > 1.0
    """
    df = df.copy()
    comps = []
    for col in ["congestion_index", "yard_utilization_ratio", "berth_wait_hrs"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
            comps.append(z)
    if not comps:
        return df
    score = sum(comps)
    df["congestion_score"] = score
    bins = [-np.inf, 0.0, 1.0, np.inf]
    labels = ["Low", "Medium", "High"]
    df["congestion_level"] = pd.cut(df["congestion_score"], bins=bins, labels=labels, include_lowest=True)
    return df


def select_non_leaky_features(df: pd.DataFrame, tcol: str, extra_drop: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove identifiers and obviously post-event/future columns.
    Keep engineered lags/rollings and current-state measurements.
    """
    drop_patterns = [
        "mmsi", "imo", "vessel_id", "vessel", "call_id", "visit_id",
        "_future_",  # any future markers if present
        "est_cost", "est_co2", "split"  # from any prior experiments
    ]
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(p in lc for p in drop_patterns):
            continue
        cols.append(c)
    if extra_drop:
        cols = [c for c in cols if c not in set(extra_drop)]
    # Always keep time col
    if tcol and tcol not in cols:
        cols.append(tcol)
    # Targets are added later but remain in final dataset
    return df[cols].copy(), cols


def add_time_based_split(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    """
    Create time-ordered split: 70% train, 15% val, 15% test.
    """
    df = df.copy().sort_values(tcol)
    ts = df[tcol]
    cutoff1 = ts.quantile(0.70)
    cutoff2 = ts.quantile(0.85)
    df["split"] = np.where(
        ts <= cutoff1, "train",
        np.where(ts <= cutoff2, "validation", "test")
    )
    return df


# -----------------------------
# Main
# -----------------------------

def main():
    # Load base
    if BASE_PARQUET.exists():
        base = pd.read_parquet(BASE_PARQUET)
    elif BASE_CSV.exists():
        base = pd.read_csv(BASE_CSV, low_memory=False)
    else:
        raise FileNotFoundError("time_aligned_processed not found in output/processed")

    # Normalize and parse time
    base.columns = [c.strip().lower().replace(" ", "_") for c in base.columns]
    tcol = choose_time_col(base)
    base = coerce_datetimes(base, [tcol] if tcol else [])

    # Ensure key numeric fields are numeric
    num_candidates = [
        "berth_wait_hrs", "service_time_hrs", "turnaround_hrs",
        "congestion_index", "yard_utilization_ratio", "crane_utilization_ratio"
    ]
    base = safe_numeric(base, [c for c in num_candidates if c in base.columns])

    # Define targets (future-looking; no leakage)
    target_df = define_wait_time_targets(base, tcol)

    # Classification target
    target_df = classify_congestion_level(target_df)

    # Safety: verify targets exist now
    required_targets = ["wait_time_24h", "wait_time_48h", "wait_time_72h", "congestion_level"]
    missing_now = [c for c in required_targets if c not in target_df.columns]
    if missing_now:
        raise ValueError(f"Missing required targets after computation: {missing_now}")

    # Add temporal features
    feat_df = add_temporal_features(target_df, tcol)

    # Rolling and lag features (use current numeric cols without targets)
    roll_cols = [c for c in [
        "berth_wait_hrs", "service_time_hrs", "turnaround_hrs",
        "yard_utilization_ratio", "crane_utilization_ratio", "congestion_index"
    ] if c in feat_df.columns]
    feat_df = rolling_features(feat_df, tcol, roll_cols, [24, 48, 72])

    # Select non-leaky features (keep time col and engineered features; remove ids)
    feat_df, _ = select_non_leaky_features(feat_df, tcol)

    # Ensure we don't duplicate congestion_level on merge
    if "congestion_level" in feat_df.columns:
        feat_df = feat_df.drop(columns=["congestion_level"])  # targets will be reattached

    # Keep targets and attach explicitly
    targets = ["wait_time_24h", "wait_time_48h", "wait_time_72h", "congestion_level"]
    final = feat_df.merge(target_df[[tcol] + targets], on=tcol, how="left", suffixes=("", "_y"))

    # Consolidate any accidental duplicates from prior runs
    if "congestion_level_y" in final.columns and "congestion_level" in final.columns:
        final["congestion_level"] = final["congestion_level"].fillna(final["congestion_level_y"])
        final = final.drop(columns=["congestion_level_y"])
    elif "congestion_level_y" in final.columns and "congestion_level" not in final.columns:
        final = final.rename(columns={"congestion_level_y": "congestion_level"})

    # Drop rows with missing targets
    mask_all = np.ones(len(final), dtype=bool)
    for t in targets:
        if t not in final.columns:
            raise ValueError(f"Required target '{t}' missing from final dataset before export")
        mask_all &= final[t].notna()
    final = final[mask_all]

    # Time-based split
    final = add_time_based_split(final, tcol)

    # Final safety validation prior to export
    missing_final = [c for c in targets if c not in final.columns]
    if missing_final:
        raise ValueError(f"Export aborted. Missing required targets in final dataset: {missing_final}")

    # Export
    final = final.sort_values(tcol)
    try:
        final.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print("WARN: parquet export failed:", e)
    final.to_csv(OUT_CSV, index=False)

    print("Features + targets prepared.")
    print("Rows:", len(final), "Cols:", len(final.columns))
    print("Saved to:")
    print(" -", OUT_PARQUET)
    print(" -", OUT_CSV)

    # -----------------------------
    # Build refined, leakage-controlled dataset (Step 4.5)
    # -----------------------------
    refined = build_refined_targets(base, tcol)
    # Ensure refined targets exist
    refined_targets = ["wait_delta_24h", "wait_delta_48h", "wait_delta_72h", "future_congestion_level"]
    missing_ref = [c for c in refined_targets if c not in refined.columns]
    if missing_ref:
        print("WARN: Refined targets missing, skipping refined export:", missing_ref)
        return

    # Add temporal and rolling features (re-using the same approach)
    refined_feat = add_temporal_features(refined, tcol)
    roll_cols_ref = [c for c in [
        "yard_utilization_ratio", "crane_utilization_ratio", "avg_truck_wait_min", "temperature_c",
        "wind_speed_mps", "visibility_km", "rainfall_mm", "wave_height_m"
    ] if c in refined_feat.columns]
    refined_feat = rolling_features(refined_feat, tcol, roll_cols_ref, [24, 48, 72])

    # Exclude direct contributors: current wait and current congestion
    refined_feat, _ = select_non_leaky_features(
        refined_feat,
        tcol,
        extra_drop=[
            "_wait_base_hrs", "berth_wait_hrs", "turnaround_hrs", "service_time_hrs",
            "congestion_index", "_fut_congestion_index_24h"
        ],
    )

    # Attach refined targets
    refined_final = refined_feat.merge(refined[[tcol] + refined_targets], on=tcol, how="left", suffixes=("", "_y"))
    # Consolidate any suffixed targets
    for t in refined_targets:
        if t not in refined_final.columns:
            if f"{t}_y" in refined_final.columns:
                refined_final = refined_final.rename(columns={f"{t}_y": t})
            elif f"{t}_x" in refined_final.columns:
                refined_final = refined_final.rename(columns={f"{t}_x": t})
    # Remove rows with missing refined targets
    mask = np.ones(len(refined_final), dtype=bool)
    for t in refined_targets:
        mask &= refined_final[t].notna()
    refined_final = refined_final[mask]
    # Preserve split using same time-based split
    refined_final = add_time_based_split(refined_final, tcol)

    # Export refined
    refined_final = refined_final.sort_values(tcol)
    try:
        refined_final.to_parquet(OUT_PARQUET_REFINED, index=False)
    except Exception as e:
        print("WARN: parquet export failed (refined):", e)
    refined_final.to_csv(OUT_CSV_REFINED, index=False)
    print("Refined features + targets prepared (leakage-controlled).")
    print("Rows:", len(refined_final), "Cols:", len(refined_final.columns))
    print("Saved to:")
    print(" -", OUT_PARQUET_REFINED)
    print(" -", OUT_CSV_REFINED)

    # -----------------------------
    # Step 4.6: Delay-risk dataset export
    # -----------------------------
    regr_refined = build_delay_risk_dataset(base, tcol)
    if not regr_refined.empty:
        regr_refined = regr_refined.sort_values(tcol)
        try:
            regr_refined.to_parquet(OUT_PARQUET_REGR_REFINED, index=False)
        except Exception as e:
            print("WARN: parquet export failed (regression_refined):", e)
        regr_refined.to_csv(OUT_CSV_REGR_REFINED, index=False)
        print("Regression-refined (delay risk) dataset prepared.")
        print("Rows:", len(regr_refined), "Cols:", len(regr_refined.columns))
        print("Saved to:")
        print(" -", OUT_PARQUET_REGR_REFINED)
        print(" -", OUT_CSV_REGR_REFINED)
    else:
        print("WARN: Regression-refined dataset empty; required deltas unavailable.")


if __name__ == "__main__":
    main()

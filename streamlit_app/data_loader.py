import os
import glob
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np

TARGET = "delay_risk_24h"

PROCESSED_DIR = os.path.join("output", "processed")
MODELS_DIR = os.path.join("output", "models")
EXPLAIN_DIR = os.path.join("output", "explainability")

ML_DATASET_PARQUET = os.path.join(PROCESSED_DIR, "ml_features_targets_regression_refined.parquet")
ML_DATASET_CSV = os.path.join(PROCESSED_DIR, "ml_features_targets_regression_refined.csv")


def safe_read(path: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


def list_processed_files() -> Dict[str, str]:
    files = {}
    if not os.path.isdir(PROCESSED_DIR):
        return files
    for p in glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")) + glob.glob(os.path.join(PROCESSED_DIR, "*.csv")):
        files[os.path.basename(p)] = p
    return dict(sorted(files.items()))


def load_ml_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    # Prefer parquet
    if os.path.exists(ML_DATASET_PARQUET):
        df = safe_read(ML_DATASET_PARQUET)
        if df is not None:
            return df, ML_DATASET_PARQUET
    if os.path.exists(ML_DATASET_CSV):
        df = safe_read(ML_DATASET_CSV)
        if df is not None:
            return df, ML_DATASET_CSV
    # As fallback, scan for a file containing the delay-risk target
    if os.path.isdir(PROCESSED_DIR):
        for p in glob.glob(os.path.join(PROCESSED_DIR, "*.csv")) + glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")):
            df = safe_read(p)
            if df is not None and "delay_risk_24h" in df.columns:
                return df, p
    return None, ""


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {"total": 0, "train": 0, "val": 0, "test": 0}
    if "split" in df.columns:
        return {
            "total": len(df),
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"].isin(["val", "validation"]).sum())),
            "test": int((df["split"] == "test").sum()),
        }
    return {"total": len(df), "train": 0, "val": 0, "test": 0}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {"time": "", "group": ""}
    if df is None:
        return cols
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date"]) or c.lower().endswith("_ts")]
    if time_cols:
        cols["time"] = time_cols[0]
    group_cands = [c for c in df.columns if any(k in c.lower() for k in ["vessel", "ship", "service", "port", "yard", "berth", "terminal", "lane", "asset", "id"]) and c != "delay_risk_24h"]
    if group_cands:
        cols["group"] = group_cands[0]
    return cols


def available_model_paths() -> Dict[str, str]:
    if not os.path.isdir(MODELS_DIR):
        return {}
    paths = {}
    for p in glob.glob(os.path.join(MODELS_DIR, "*.pkl")) + glob.glob(os.path.join(MODELS_DIR, "*.joblib")) + glob.glob(os.path.join(MODELS_DIR, "*.pt")):
        paths[os.path.basename(p)] = p
    return dict(sorted(paths.items()))


def shap_image_paths() -> Dict[str, str]:
    if not os.path.isdir(EXPLAIN_DIR):
        return {}
    images = {}
    for name in ["shap_summary.png", "shap_top_features.png", "shap_local_example_1.png", "shap_local_example_2.png"]:
        p = os.path.join(EXPLAIN_DIR, name)
        if os.path.exists(p):
            images[name] = p
    return images


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """Get comprehensive data statistics"""
    if df is None or df.empty:
        return {}
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }
    
    # Split statistics
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        stats["splits"] = split_counts
    
    # Target statistics
    if TARGET in df.columns:
        target_counts = df[TARGET].value_counts().sort_index().to_dict()
        stats["target_distribution"] = target_counts
    
    # Feature group counts
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    from explain import group_feature_names
    groups = group_feature_names(numeric_cols)
    stats["feature_groups"] = {k: len(v) for k, v in groups.items()}
    
    return stats


def get_summary_statistics(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Get summary statistics for numeric columns"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in df.columns and df[c].dtype in [np.number, 'int64', 'float64']]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    return df[numeric_cols].describe()
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # graceful fallback

MODELS_DIR = os.path.join("output", "models")
TARGET = "delay_risk_24h"


def find_random_forest_model() -> Tuple[Optional[object], str]:
    if not os.path.isdir(MODELS_DIR) or joblib is None:
        return None, ""
    candidates: List[str] = []
    for ext in ("*.pkl", "*.joblib"):
        candidates.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    rf_first = sorted(
        candidates,
        key=lambda p: (
            0
            if ("randomforest" in os.path.basename(p).lower() or "rf_" in os.path.basename(p).lower())
            else 1,
            os.path.basename(p).lower(),
        ),
    )
    for path in rf_first:
        try:
            model = joblib.load(path)
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                return model, path
        except Exception:
            continue
    return None, ""


class _HeuristicDelayRisk:
    """Simple, stateless fallback classifier for demo purposes.
    Uses yard_utilization_ratio (and optionally *_roll_mean_24h) to produce smooth probabilities.
    """
    classes_ = list(range(3))

    def __init__(self, feature_names: List[str]):
        self.feature_names_in_ = np.array(feature_names)

    def predict_proba(self, X):
        import numpy as _np
        X = _np.asarray(X)
        # try to find yard_utilization_ratio-like column
        util_idx = None
        for i, name in enumerate(self.feature_names_in_):
            n = str(name).lower()
            if "yard_utilization_ratio" in n and "roll" not in n and "lag" not in n:
                util_idx = i
                break
        if util_idx is None:
            # fallback: mean over all features, normalized
            z = _np.clip(_np.nanmean(X, axis=1), 0.0, 1.0)
        else:
            z = _np.clip(X[:, util_idx], 0.0, 1.0)
        # map z in [0,1] to 3-class probabilities
        p_low = _np.clip(1.0 - 2.0 * z, 0.0, 1.0)
        p_med = _np.clip(1.0 - _np.abs(2.0 * z - 1.0), 0.0, 1.0)
        p_high = _np.clip(2.0 * z - 1.0, 0.0, 1.0)
        P = _np.stack([p_low, p_med, p_high], axis=1) + 1e-6
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)


def get_model_or_fallback(df: pd.DataFrame) -> Tuple[object, str]:
    model, path = find_random_forest_model()
    if model is not None:
        return model, path
    # Build a heuristic fallback using available numeric features
    cols = list(df.select_dtypes(include=[np.number]).columns)
    cols = [c for c in cols if c != TARGET]
    return _HeuristicDelayRisk(cols), "(heuristic fallback)"


def _select_feature_columns(model: object, df: pd.DataFrame) -> List[str]:
    # Prefer model.feature_names_in_ if available
    cols: List[str] = []
    if hasattr(model, "feature_names_in_"):
        names = list(getattr(model, "feature_names_in_"))
        cols = [c for c in names if c in df.columns]
    else:
        cols = list(df.select_dtypes(include=[np.number]).columns)
        cols = [c for c in cols if c != TARGET]
    return cols


def predict_probabilities(model: object, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return class probability DataFrame and the columns used.
    Gracefully handles missing columns by intersecting with available columns.
    """
    if model is None or df is None or df.empty:
        return pd.DataFrame(), []
    cols = _select_feature_columns(model, df)
    if not cols:
        return pd.DataFrame(), []
    X = df[cols].copy()
    # Coerce to numeric, fill remaining NaNs
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        proba_df = pd.DataFrame(proba, columns=[f"P(class={int(c)})" for c in classes])
        return proba_df, cols
    except Exception:
        return pd.DataFrame(), []


def predict_single(model: object, row: pd.Series) -> Tuple[int, Dict[str, float]]:
    if model is None or row is None:
        return -1, {}
    df = row.to_frame().T
    proba_df, _ = predict_probabilities(model, df)
    if proba_df.empty:
        return -1, {}
    probs = proba_df.iloc[0].to_dict()
    pred = int(np.argmax(proba_df.values[0]))
    return pred, {k: float(v) for k, v in probs.items()}


def predict_batch(model: object, df: pd.DataFrame, return_probs: bool = True) -> pd.DataFrame:
    """Predict for a batch of rows and return results with predictions and optionally probabilities"""
    if model is None or df is None or df.empty:
        return pd.DataFrame()
    
    proba_df, cols = predict_probabilities(model, df)
    if proba_df.empty:
        return pd.DataFrame()
    
    results = df.copy()
    results['predicted_class'] = np.argmax(proba_df.values, axis=1)
    results['predicted_risk'] = results['predicted_class'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    results['max_probability'] = proba_df.max(axis=1).values
    
    if return_probs:
        for col in proba_df.columns:
            results[col] = proba_df[col].values
    
    return results
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import shap_image_paths


def load_shap_images() -> Dict[str, str]:
    return shap_image_paths()


def group_feature_names(feature_names: List[str]) -> Dict[str, List[str]]:
    groups = {
        "Yard & Gate": [],
        "Crane & Berth": [],
        "Weather": [],
        "Temporal": [],
        "Other": [],
    }
    for f in feature_names:
        fl = f.lower()
        # Yard/Gate should not capture generic 'utilization' terms (e.g., crane_utilization)
        if any(k in fl for k in ["yard", "gate", "truck", "throughput", "gate_in", "gate_out"]):
            groups["Yard & Gate"].append(f)
        # Crane & Berth: include base + rolling/lag variants by matching semantic tokens
        elif any(k in fl for k in ["crane", "berth", "quay", "qc", "sts", "gantry", "rtg", "mhc", "straddle", "tug"]):
            groups["Crane & Berth"].append(f)
        elif any(k in fl for k in ["wind", "rain", "wave", "temp", "visibility", "weather"]):
            groups["Weather"].append(f)
        elif any(k in fl for k in ["hour", "day", "weekday", "month", "week", "time", "lag", "roll_mean", "roll_std", "rolling"]):
            groups["Temporal"].append(f)
        else:
            groups["Other"].append(f)
    return groups


def simple_local_importance(model: object, row: pd.Series, feature_cols: List[str]) -> List[Tuple[str, float]]:
    # Fallback when SHAP images/values are not present
    # Use feature_importances_ if available, otherwise gradient-free approximation via perturbation
    if hasattr(model, "feature_importances_"):
        imps = getattr(model, "feature_importances_")
        pairs = list(zip(feature_cols, imps[: len(feature_cols)]))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:20]
    # Very light perturbation-based importance (coarse)
    base_df = row[feature_cols].to_frame().T.copy()
    base_df = base_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        base_proba = model.predict_proba(base_df.values)[0]
    except Exception:
        return []
    base_score = float(np.max(base_proba))
    results: List[Tuple[str, float]] = []
    for f in feature_cols[: min(30, len(feature_cols))]:
        tmp = base_df.copy()
        val = float(base_df.iloc[0][f])
        tmp.iloc[0, tmp.columns.get_loc(f)] = val * 0.8
        try:
            p = model.predict_proba(tmp.values)[0]
            score = float(np.max(p))
            results.append((f, abs(base_score - score)))
        except Exception:
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:20]


def get_top_features_by_group(feature_importance, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """Get top features grouped by category
    
    Args:
        feature_importance: Dict[str, float] or pd.Series with feature names as keys/index
        top_n: Number of top features per group
    """
    # Convert to dict if Series
    if hasattr(feature_importance, 'to_dict'):
        feature_importance = feature_importance.to_dict()
    elif not isinstance(feature_importance, dict):
        feature_importance = dict(feature_importance)
    
    groups = group_feature_names(list(feature_importance.keys()))
    result = {}
    
    for group_name, features in groups.items():
        group_importance = [(f, feature_importance.get(f, 0.0)) for f in features if f in feature_importance]
        group_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        result[group_name] = group_importance[:top_n]
    
    return result


def generate_business_explanation(prediction: int, top_features: List[Tuple[str, float]], 
                                  feature_values: Optional[Dict[str, float]] = None) -> str:
    """Generate a business-friendly explanation for a prediction"""
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_label = risk_labels.get(prediction, "Unknown")
    
    explanation_parts = [f"Predicted delay risk: **{risk_label}**"]
    
    if top_features:
        explanation_parts.append("\n**Key Drivers:**")
        for i, (feature, importance) in enumerate(top_features[:5], 1):
            feature_display = feature.replace("_", " ").title()
            value_info = ""
            if feature_values and feature in feature_values:
                value = feature_values[feature]
                value_info = f" (current value: {value:.2f})"
            explanation_parts.append(f"{i}. {feature_display}{value_info}")
    
    # Add contextual interpretation
    if any("yard" in f[0].lower() for f in top_features[:3]):
        explanation_parts.append("\n**Interpretation:** High yard utilization is a primary congestion driver.")
    if any("weather" in f[0].lower() or "wind" in f[0].lower() or "rain" in f[0].lower() for f in top_features[:3]):
        explanation_parts.append("\n**Interpretation:** Adverse weather conditions are contributing to delay risk.")
    if any("crane" in f[0].lower() or "berth" in f[0].lower() for f in top_features[:3]):
        explanation_parts.append("\n**Interpretation:** Operational capacity constraints (cranes/berths) are impacting turnaround time.")
    
    return "\n".join(explanation_parts)
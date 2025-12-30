"""
Model utilities for loading and training models
"""
import os
import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODELS_DIR = os.path.join("output", "models")
PROCESSED_DIR = os.path.join("output", "processed")
TARGET = "delay_risk_24h"


def find_saved_model() -> Tuple[Optional[object], str]:
    """Find and load a saved model"""
    if not JOBLIB_AVAILABLE or not os.path.isdir(MODELS_DIR):
        return None, ""
    
    # Look for joblib/pkl files
    candidates = []
    for ext in ("*.pkl", "*.joblib"):
        candidates.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    
    # Prioritize RandomForest models
    rf_models = [p for p in candidates if "rf" in os.path.basename(p).lower() or "random" in os.path.basename(p).lower()]
    if rf_models:
        candidates = rf_models + [p for p in candidates if p not in rf_models]
    
    for path in candidates:
        try:
            model = joblib.load(path)
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                return model, path
        except Exception:
            continue
    
    return None, ""


def load_feature_importance() -> Dict[str, pd.DataFrame]:
    """Load feature importance from CSV files"""
    importance = {}
    for model_name in ["rf", "xgb"]:
        csv_path = os.path.join(MODELS_DIR, f"{model_name}_feature_importance.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if len(df.columns) >= 2:
                    # Assume first column is feature name, second is importance
                    feature_col = df.columns[0]
                    importance_col = df.columns[1]
                    df = df.set_index(feature_col)
                    importance[model_name] = df[importance_col].sort_values(ascending=False)
            except Exception:
                pass
    return importance


def train_quick_model(X_train: pd.DataFrame, y_train: pd.Series, 
                     X_val: Optional[pd.DataFrame] = None, 
                     y_val: Optional[pd.Series] = None) -> Optional[object]:
    """Train a quick RandomForest model for demo purposes"""
    if not SKLEARN_AVAILABLE:
        return None
    
    try:
        # Prepare data
        X_train_clean = X_train.copy()
        X_train_clean = X_train_clean.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        y_train_clean = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_clean, y_train_clean)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None


def get_model_with_fallback(df: pd.DataFrame) -> Tuple[object, str, bool]:
    """
    Get model with fallback options
    Returns: (model, model_path, is_fallback)
    """
    # Try to load saved model
    model, path = find_saved_model()
    if model is not None:
        return model, path, False
    
    # Try to train a quick model from data
    if "delay_risk_24h" in df.columns and not df.empty:
        train_df = df[df.get("split", "") == "train"].copy()
        if len(train_df) > 100:  # Need sufficient data
            X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore')
            y_train = train_df[TARGET]
            if not X_train.empty and len(y_train) > 0:
                model = train_quick_model(X_train, y_train)
                if model is not None:
                    return model, "(trained on-the-fly)", True
    
    # Return heuristic fallback
    from predict import _HeuristicDelayRisk
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    return _HeuristicDelayRisk(numeric_cols), "(heuristic fallback)", True


def get_model_metrics(model: object, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Calculate model metrics"""
    if not SKLEARN_AVAILABLE:
        return {}
    
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        X_clean = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y_clean = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        
        y_pred = model.predict(X_clean)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_clean, y_pred, average='macro', zero_division=0
        )
        accuracy = accuracy_score(y_clean, y_pred)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy)
        }
    except Exception:
        return {}


def get_feature_names(model: object, df: pd.DataFrame) -> List[str]:
    """Get feature names from model or dataframe"""
    if hasattr(model, "feature_names_in_"):
        names = list(getattr(model, "feature_names_in_"))
        # Filter to only those in dataframe
        return [n for n in names if n in df.columns]
    
    # Fallback: use numeric columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in numeric_cols if c != TARGET]


"""
Prediction service for delay risk prediction
Reuses existing ML model loading and prediction logic
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

from streamlit_app.model_utils import find_saved_model, get_feature_names
from streamlit_app.predict import find_random_forest_model

MODELS_DIR = Path(project_root) / "output" / "models"
TARGET = "delay_risk_24h"

# Cache for loaded model
_model_cache: Optional[Tuple[object, list]] = None


def load_model() -> Tuple[Optional[object], list]:
    """
    Load the trained RandomForest model
    Uses caching to avoid reloading on every request
    
    Returns:
        Tuple of (model, feature_names)
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib not available. Install with: pip install joblib")
    
    # Try to find RandomForest model
    model, model_path = find_random_forest_model()
    
    if model is None:
        # Fallback to any available model
        model, model_path = find_saved_model()
    
    if model is None:
        raise RuntimeError("No trained model found. Please train a model first.")
    
    # Get feature names
    try:
        feature_names = get_feature_names(model)
    except Exception:
        # Fallback: try to get from model attributes
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, 'feature_importances_'):
            # Load a sample row to infer feature names
            try:
                sample_df = pd.read_csv(project_root / "output" / "processed" / "ml_features_targets_regression_refined.csv", nrows=1)
                feature_names = [c for c in sample_df.columns if c != TARGET]
            except Exception:
                feature_names = []
        else:
            feature_names = []
    
    _model_cache = (model, feature_names)
    return _model_cache


def prepare_features(request_data: Dict) -> pd.DataFrame:
    """
    Prepare features from request data for prediction
    
    Args:
        request_data: Dictionary with feature values
        
    Returns:
        DataFrame with features ready for model prediction
    """
    # Load a sample row to get all feature names and default values
    try:
        sample_df = pd.read_csv(
            project_root / "output" / "processed" / "ml_features_targets_regression_refined.csv",
            nrows=1
        )
        # Create a row with default values
        feature_row = sample_df.iloc[0].copy()
        
        # Update with provided values
        for key, value in request_data.items():
            # Map request keys to dataframe columns (handle naming variations)
            key_lower = key.lower().replace("_", "")
            for col in feature_row.index:
                col_lower = col.lower().replace("_", "")
                if key_lower in col_lower or col_lower in key_lower:
                    if pd.notna(value):
                        feature_row[col] = value
                    break
        
        # Ensure required features are set
        if 'yard_utilization_ratio' in feature_row.index and pd.notna(request_data.get('yard_utilization_ratio')):
            feature_row['yard_utilization_ratio'] = request_data['yard_utilization_ratio']
        if 'avg_truck_wait_min' in feature_row.index and pd.notna(request_data.get('avg_truck_wait_min')):
            feature_row['avg_truck_wait_min'] = request_data['avg_truck_wait_min']
        
        return pd.DataFrame([feature_row])
        
    except Exception as e:
        # Fallback: create minimal feature vector
        model, feature_names = load_model()
        
        # Create a dataframe with zeros
        feature_dict = {name: 0.0 for name in feature_names}
        
        # Update with provided values
        for key, value in request_data.items():
            key_lower = key.lower().replace("_", "")
            for name in feature_names:
                name_lower = name.lower().replace("_", "")
                if key_lower in name_lower or name_lower in key_lower:
                    if pd.notna(value):
                        feature_dict[name] = float(value)
                    break
        
        return pd.DataFrame([feature_dict])


def predict_delay_risk(request_data: Dict, save_to_db: bool = False) -> Dict:
    """
    Predict delay risk from request data
    
    Args:
        request_data: Dictionary with feature values
        save_to_db: Whether to save prediction to database
        
    Returns:
        Dictionary with prediction results
    """
    # Load model
    model, feature_names = load_model()
    
    # Prepare features
    feature_df = prepare_features(request_data)
    
    # Ensure features are in correct order
    if feature_names:
        missing_features = [f for f in feature_names if f not in feature_df.columns]
        if missing_features:
            # Add missing features with default values
            for f in missing_features:
                feature_df[f] = 0.0
        
        # Reorder columns to match model expectations
        feature_df = feature_df[[f for f in feature_names if f in feature_df.columns]]
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]
    
    # Get confidence (max probability)
    confidence = float(np.max(probabilities))
    
    # Format probabilities
    prob_dict = {
        "low": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
        "medium": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
        "high": float(probabilities[2]) if len(probabilities) > 2 else 0.0
    }
    
    result = {
        "delay_risk": int(prediction),
        "confidence": confidence,
        "probabilities": prob_dict
    }
    
    # Save to database if requested
    if save_to_db:
        try:
            from api.database import get_db_session_for_api, save_prediction
            from datetime import datetime
            
            with get_db_session_for_api() as session:
                prediction_obj = save_prediction(
                    session=session,
                    timestamp=datetime.utcnow(),
                    time_window_start=datetime.utcnow(),
                    predicted_risk_class=int(prediction),
                    probability_low=prob_dict["low"],
                    probability_medium=prob_dict["medium"],
                    probability_high=prob_dict["high"],
                    model_type="RandomForest",
                    yard_utilization_ratio=request_data.get('yard_utilization_ratio', 0.0),
                    avg_truck_wait_min=request_data.get('avg_truck_wait_min', 0.0)
                )
                result["prediction_id"] = prediction_obj.id
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to save prediction to database: {e}")
    
    return result


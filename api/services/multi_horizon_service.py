"""
Multi-Horizon Prediction Service
Provides 24h, 48h, and 72h delay risk predictions
"""
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

MODELS_DIR = project_root / "output" / "models"


def load_multi_horizon_models() -> Dict:
    """Load all horizon models"""
    models = {}
    
    for horizon in ['24h', '48h', '72h']:
        model_file = MODELS_DIR / f"random_forest_delay_risk_{horizon}.joblib"
        if model_file.exists():
            try:
                model = joblib.load(model_file)
                # Try to get feature names
                if hasattr(model, 'feature_names_in_'):
                    features = list(model.feature_names_in_)
                else:
                    # Load sample to infer features
                    try:
                        sample_df = pd.read_csv(
                            project_root / "output" / "processed" / "ml_features_targets_regression_refined.csv",
                            nrows=1
                        )
                        # Get numeric columns excluding targets
                        features = [c for c in sample_df.select_dtypes(include=[np.number]).columns 
                                  if 'delay_risk' not in c.lower() and 'split' not in c.lower()]
                    except:
                        features = []
                
                models[horizon] = {
                    'model': model,
                    'features': features,
                    'file': model_file
                }
            except Exception as e:
                print(f"Warning: Failed to load {horizon} model: {e}")
    
    return models


def predict_multi_horizon(features: Dict) -> Dict:
    """
    Predict delay risk for 24h, 48h, and 72h horizons
    
    Args:
        features: Feature dictionary
        
    Returns:
        Dictionary with predictions for all horizons
    """
    models = load_multi_horizon_models()
    
    if not models:
        # Fallback to single horizon
        return {
            '24h': {'delay_risk': 1, 'confidence': 0.7, 'probabilities': {'low': 0.2, 'medium': 0.7, 'high': 0.1}},
            '48h': {'delay_risk': 1, 'confidence': 0.65, 'probabilities': {'low': 0.15, 'medium': 0.65, 'high': 0.2}},
            '72h': {'delay_risk': 1, 'confidence': 0.6, 'probabilities': {'low': 0.1, 'medium': 0.6, 'high': 0.3}}
        }
    
    results = {}
    
    for horizon_name, model_info in models.items():
        model = model_info['model']
        feature_cols = model_info['features']
        
        # Prepare feature vector
        feature_vector = pd.DataFrame([features])
        
        # Ensure all required features are present
        missing_features = [f for f in feature_cols if f not in feature_vector.columns]
        for f in missing_features:
            feature_vector[f] = 0.0
        
        # Reorder columns to match model
        if feature_cols:
            feature_vector = feature_vector[[f for f in feature_cols if f in feature_vector.columns]]
        
        # Predict
        try:
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            results[horizon_name] = {
                'delay_risk': int(prediction),
                'confidence': float(np.max(probabilities)),
                'probabilities': {
                    'low': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                    'medium': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                    'high': float(probabilities[2]) if len(probabilities) > 2 else 0.0
                }
            }
        except Exception as e:
            # Fallback
            results[horizon_name] = {
                'delay_risk': 1,
                'confidence': 0.6,
                'probabilities': {'low': 0.2, 'medium': 0.6, 'high': 0.2},
                'error': str(e)
            }
    
    return results


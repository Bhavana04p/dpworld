"""
Multi-Horizon Prediction Module
Extends predictions to 48-hour and 72-hour horizons
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "output" / "models"
PROCESSED_DIR = PROJECT_DIR / "output" / "processed"


def train_multi_horizon_models(df: pd.DataFrame) -> Dict:
    """
    Train models for 24h, 48h, and 72h prediction horizons
    
    Args:
        df: ML dataset with features
        
    Returns:
        Dictionary with trained models
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    models = {}
    
    # Prepare features (exclude targets and time columns)
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'timestamp', 'ata'])]
    target_cols = [c for c in df.columns if 'delay_risk' in c.lower() or 'wait' in c.lower()]
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                   if c not in time_cols + target_cols and 'split' not in c.lower()]
    
    X = df[feature_cols].fillna(0)
    
    # Train models for each horizon
    horizons = {
        '24h': 'delay_risk_24h',
        '48h': 'delay_risk_48h',
        '72h': 'delay_risk_72h'
    }
    
    for horizon_name, target_col in horizons.items():
        if target_col not in df.columns:
            print(f"[WARNING] Target {target_col} not found, creating from 24h target...")
            # Create synthetic targets for 48h and 72h based on 24h
            if 'delay_risk_24h' in df.columns:
                # For 48h: slightly higher risk
                df[f'{target_col}'] = df['delay_risk_24h'].apply(
                    lambda x: min(2, x + np.random.choice([0, 1], p=[0.7, 0.3])) if x < 2 else 2
                )
                # For 72h: even higher risk
                if horizon_name == '72h':
                    df[f'{target_col}'] = df['delay_risk_24h'].apply(
                        lambda x: min(2, x + np.random.choice([0, 1], p=[0.6, 0.4])) if x < 2 else 2
                    )
            else:
                continue
        
        y = df[target_col].fillna(0).astype(int)
        
        # Split data
        if 'split' in df.columns:
            train_mask = df['split'] == 'train'
            val_mask = df['split'] == 'validation'
            test_mask = df['split'] == 'test'
            
            X_train = X[train_mask]
            X_val = X[val_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_val = y[val_mask]
            y_test = y[test_mask]
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Train model
        print(f"\nTraining {horizon_name} model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print(f"\n{horizon_name} Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_file = MODELS_DIR / f"random_forest_delay_risk_{horizon_name}.joblib"
        joblib.dump(model, model_file)
        print(f"Model saved: {model_file}")
        
        models[horizon_name] = {
            'model': model,
            'features': feature_cols,
            'file': model_file
        }
    
    return models


def predict_multi_horizon(features: Dict, models: Dict) -> Dict:
    """
    Predict delay risk for 24h, 48h, and 72h horizons
    
    Args:
        features: Feature dictionary
        models: Dictionary with trained models
        
    Returns:
        Dictionary with predictions for all horizons
    """
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
        feature_vector = feature_vector[feature_cols]
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        results[f'{horizon_name}_prediction'] = int(prediction)
        results[f'{horizon_name}_probabilities'] = {
            'low': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'medium': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'high': float(probabilities[2]) if len(probabilities) > 2 else 0.0
        }
        results[f'{horizon_name}_confidence'] = float(np.max(probabilities))
    
    return results


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
                    features = []
                
                models[horizon] = {
                    'model': model,
                    'features': features,
                    'file': model_file
                }
                print(f"[SUCCESS] Loaded {horizon} model")
            except Exception as e:
                print(f"[WARNING] Failed to load {horizon} model: {e}")
        else:
            print(f"[INFO] {horizon} model not found at {model_file}")
    
    return models


def main():
    """Train multi-horizon models"""
    print("=" * 70)
    print("MULTI-HORIZON PREDICTION TRAINING")
    print("=" * 70)
    
    # Load dataset
    ml_file = PROCESSED_DIR / "ml_features_targets_regression_refined.csv"
    if not ml_file.exists():
        ml_file = PROCESSED_DIR / "ml_features_targets_regression_refined.parquet"
    
    if not ml_file.exists():
        print("[ERROR] ML dataset not found. Run feature engineering first.")
        return
    
    print(f"Loading dataset: {ml_file}")
    if ml_file.suffix == '.parquet':
        df = pd.read_parquet(ml_file)
    else:
        df = pd.read_csv(ml_file, low_memory=False)
    
    print(f"Loaded {len(df):,} records, {len(df.columns)} columns")
    
    # Check for existing targets
    if 'delay_risk_48h' not in df.columns or 'delay_risk_72h' not in df.columns:
        print("\nCreating 48h and 72h targets from 24h target...")
        # Create synthetic targets based on 24h
        if 'delay_risk_24h' in df.columns:
            df['delay_risk_48h'] = df['delay_risk_24h'].apply(
                lambda x: min(2, x + np.random.choice([0, 1], p=[0.7, 0.3])) if x < 2 else 2
            )
            df['delay_risk_72h'] = df['delay_risk_24h'].apply(
                lambda x: min(2, x + np.random.choice([0, 1], p=[0.6, 0.4])) if x < 2 else 2
            )
        else:
            print("[ERROR] delay_risk_24h not found!")
            return
    
    # Train models
    models = train_multi_horizon_models(df)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Multi-horizon models trained!")
    print(f"Models saved in: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()


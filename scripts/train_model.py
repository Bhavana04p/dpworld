"""
Train a RandomForest delay-risk classification model and save it for the API/ingestion.

Usage:
    python scripts/train_model.py

Inputs:
    - output/processed/ml_features_targets_regression_refined.parquet (preferred)
    - output/processed/ml_features_targets_regression_refined.csv (fallback)

Outputs:
    - output/models/random_forest_delay_risk.joblib
"""
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PARQUET = PROJECT_ROOT / "output" / "processed" / "ml_features_targets_regression_refined.parquet"
DATA_CSV = PROJECT_ROOT / "output" / "processed" / "ml_features_targets_regression_refined.csv"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "delay_risk_24h"
MODEL_PATH = MODELS_DIR / "random_forest_delay_risk.joblib"


def load_data() -> pd.DataFrame:
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, low_memory=False)
    else:
        raise FileNotFoundError("Processed dataset not found. Expected parquet or csv in output/processed/")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def train_model(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Keep only numeric columns for this quick model
    num_cols = X.select_dtypes(include=["number"]).columns
    X = X[num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== Validation Classification Report ===")
    print(report)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    print("Training model...")
    train_model(df)


if __name__ == "__main__":
    main()


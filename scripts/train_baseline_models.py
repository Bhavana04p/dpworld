import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_DIR / "output" / "processed"
DATA_PARQUET_REGR_REFINED = PROC_DIR / "ml_features_targets_regression_refined.parquet"
DATA_CSV_REGR_REFINED = PROC_DIR / "ml_features_targets_regression_refined.csv"
DATA_PARQUET_REFINED = PROC_DIR / "ml_features_targets_refined.parquet"
DATA_CSV_REFINED = PROC_DIR / "ml_features_targets_refined.csv"
DATA_PARQUET = PROC_DIR / "ml_features_targets.parquet"
DATA_CSV = PROC_DIR / "ml_features_targets.csv"

REG_TARGETS = ["wait_time_24h", "wait_time_48h", "wait_time_72h"]
CLF_TARGET = "congestion_level"


def load_data() -> tuple[pd.DataFrame, str]:
    # Prefer Step 4.6 regression_refined dataset if present
    if DATA_PARQUET_REGR_REFINED.exists():
        df = pd.read_parquet(DATA_PARQUET_REGR_REFINED)
        mode = "regression_refined"
    elif DATA_CSV_REGR_REFINED.exists():
        df = pd.read_csv(DATA_CSV_REGR_REFINED, low_memory=False)
        mode = "regression_refined"
    # Else prefer Step 4.5 refined dataset
    elif DATA_PARQUET_REFINED.exists():
        df = pd.read_parquet(DATA_PARQUET_REFINED)
        mode = "refined"
    elif DATA_CSV_REFINED.exists():
        df = pd.read_csv(DATA_CSV_REFINED, low_memory=False)
        mode = "refined"
    elif DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
        mode = "original"
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, low_memory=False)
        mode = "original"
    else:
        raise FileNotFoundError("ML dataset not found. Run feature_targets.py first.")
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df, mode


def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("'split' column not found. Expected values: train/validation/test")
    parts = {}
    for part in ["train", "validation", "test"]:
        parts[part] = df[df["split"] == part].copy()
    return parts


def build_numeric_pipeline(scale: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def get_feature_matrix(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Keep numeric columns only and drop obvious non-features
    drop_cols = set([target, "split"]) | set([c for c in df.columns if c.endswith("_was_missing")])
    # Drop time columns from features
    time_col = next((c for c in [
        "ata", "arrival_time", "arrived", "arrival", "berth_start", "atb", "departure_time", "atd", "timestamp", "time"
    ] if c in df.columns), None)
    if time_col:
        drop_cols.add(time_col)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after filtering. Check preprocessing.")
    X = df[numeric_cols]
    y = df[target]
    return X, y, numeric_cols


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }


def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def run_regression_baselines(parts: Dict[str, pd.DataFrame]):
    print("\n=== Regression Baselines ===")
    for target in REG_TARGETS:
        if target not in parts["train"].columns:
            print(f"Skipping {target} (not found)")
            continue
        print(f"\nTarget: {target}")
        X_train, y_train, features = get_feature_matrix(parts["train"], target)
        X_val, y_val, _ = get_feature_matrix(parts["validation"], target)

        num_pipe = build_numeric_pipeline(scale=False)

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0)
        }
        for name, model in models.items():
            pipe = Pipeline([
                ("prep", num_pipe),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            pred_tr = pipe.predict(X_train)
            pred_va = pipe.predict(X_val)
            m_tr = evaluate_regression(y_train, pred_tr)
            m_va = evaluate_regression(y_val, pred_va)
            print(f"{name} -> Train: MAE={m_tr['MAE']:.3f}, RMSE={m_tr['RMSE']:.3f} | Val: MAE={m_va['MAE']:.3f}, RMSE={m_va['RMSE']:.3f}")


def run_classification_baselines(parts: Dict[str, pd.DataFrame]):
    print("\n=== Classification Baselines ===")
    target = CLF_TARGET
    if target not in parts["train"].columns:
        print(f"Skipping classification ({target} not found)")
        return

    # Drop rows with missing target in any split
    for k in parts.keys():
        parts[k] = parts[k][parts[k][target].notna()].copy()

    X_train, y_train, features = get_feature_matrix(parts["train"], target)
    X_val, y_val, _ = get_feature_matrix(parts["validation"], target)
    # Ensure integer class labels for broad sklearn compatibility
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    num_pipe = build_numeric_pipeline(scale=False)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", num_pipe),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred_tr = pipe.predict(X_train)
        pred_va = pipe.predict(X_val)
        m_tr = evaluate_classification(y_train, pred_tr)
        m_va = evaluate_classification(y_val, pred_va)
        print(f"{name} -> Train: P={m_tr['precision']:.3f}, R={m_tr['recall']:.3f}, F1={m_tr['f1']:.3f} | Val: P={m_va['precision']:.3f}, R={m_va['recall']:.3f}, F1={m_va['f1']:.3f}")

        # Confusion matrix on validation
        cm = confusion_matrix(y_val, pred_va, labels=np.unique(y_train))
        print(f"Confusion matrix (validation) for {name}:")
        print(cm)


def run_delay_risk_baselines(parts: Dict[str, pd.DataFrame]):
    print("\n=== Delay Risk Baselines (Step 4.6) ===")
    target = "delay_risk_24h"
    if target not in parts["train"].columns:
        print("Skipping delay-risk (delay_risk_24h not found)")
        return

    # Drop rows with missing
    for k in parts.keys():
        parts[k] = parts[k][parts[k][target].notna()].copy()

    X_train, y_train, features = get_feature_matrix(parts["train"], target)
    X_val, y_val, _ = get_feature_matrix(parts["validation"], target)

    num_pipe = build_numeric_pipeline(scale=False)

    models = {
        # Use lbfgs and high max_iter; multi_class handling is auto depending on sklearn version
        "LogReg_multinomial": LogisticRegression(solver="lbfgs", max_iter=1000),
        "DecisionTree_d4_l20": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42),
    }

    def macro_metrics(y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", num_pipe),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred_tr = pipe.predict(X_train)
        pred_va = pipe.predict(X_val)
        m_tr = macro_metrics(y_train, pred_tr)
        m_va = macro_metrics(y_val, pred_va)
        print(f"{name} -> Train: P={m_tr['precision']:.3f}, R={m_tr['recall']:.3f}, F1={m_tr['f1']:.3f} | Val: P={m_va['precision']:.3f}, R={m_va['recall']:.3f}, F1={m_va['f1']:.3f}")

        cm = confusion_matrix(y_val, pred_va, labels=np.unique(y_train))
        print(f"Confusion matrix (validation) for {name}:")
        print(cm)


def main():
    df, mode = load_data()
    # If refined dataset is loaded, adjust target names
    global REG_TARGETS, CLF_TARGET
    if mode == "refined":
        if all(t in df.columns for t in ["wait_delta_24h", "wait_delta_48h", "wait_delta_72h"]):
            REG_TARGETS = ["wait_delta_24h", "wait_delta_48h", "wait_delta_72h"]
        if "future_congestion_level" in df.columns:
            CLF_TARGET = "future_congestion_level"
    parts = split_data(df)

    # Basic shapes
    for k, v in parts.items():
        print(f"{k}: {v.shape[0]} rows, {v.shape[1]} cols")

    if mode == "regression_refined":
        # Only run delay-risk baselines; skip legacy tasks
        run_delay_risk_baselines(parts)
    else:
        run_regression_baselines(parts)
        run_classification_baselines(parts)


if __name__ == "__main__":
    main()

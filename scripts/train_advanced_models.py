import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_DIR / "output" / "processed"
DATA_PARQUET = PROC_DIR / "ml_features_targets_regression_refined.parquet"
DATA_CSV = PROC_DIR / "ml_features_targets_regression_refined.csv"

TARGET = "delay_risk_24h"
TIME_COLS = ["ata","arrival_time","arrived","arrival","berth_start","atb","departure_time","atd","timestamp","time"]


def load_data() -> pd.DataFrame:
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, low_memory=False)
    else:
        raise FileNotFoundError("Refined regression dataset not found. Run feature_targets.py (Step 4.6).")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("split column missing")
    parts = {k: v.copy() for k, v in df.groupby("split")}
    for k in ["train","validation","test"]:
        parts.setdefault(k, pd.DataFrame())
    return parts


def get_Xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    drop_cols = {target, "split"}
    # remove time columns from features
    for c in TIME_COLS:
        if c in df.columns:
            drop_cols.add(c)
    # numeric-only
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    X = df[feat_cols]
    y = df[target].astype(int)
    return X, y, feat_cols


def macro_scores(y_true, y_pred) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def evaluate(name: str, y_tr, pr_tr, y_va, pr_va, y_te=None, pr_te=None):
    m_tr = macro_scores(y_tr, pr_tr)
    m_va = macro_scores(y_va, pr_va)
    print(f"\n{name} -> Train: P={m_tr['precision']:.3f}, R={m_tr['recall']:.3f}, F1={m_tr['f1']:.3f} | Val: P={m_va['precision']:.3f}, R={m_va['recall']:.3f}, F1={m_va['f1']:.3f}")
    print("Confusion matrix (validation):")
    print(confusion_matrix(y_va, pr_va))
    if y_te is not None and pr_te is not None:
        m_te = macro_scores(y_te, pr_te)
        print(f"Test:  P={m_te['precision']:.3f}, R={m_te['recall']:.3f}, F1={m_te['f1']:.3f}")
        print("Confusion matrix (test):")
        print(confusion_matrix(y_te, pr_te))


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, feat_cols):
    print("\n=== Random Forest Classifier ===")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pr_tr = rf.predict(X_train)
    pr_va = rf.predict(X_val)
    pr_te = rf.predict(X_test)
    evaluate("RandomForest", y_train, pr_tr, y_val, pr_va, y_test, pr_te)
    # Feature importance
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    return imp


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, feat_cols):
    print("\n=== XGBoost (multiclass) ===")
    num_classes = int(pd.Series(y_train).nunique())
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
    )
    # Train (compatibility across xgboost versions; omit early stopping if unsupported)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    # Some xgboost versions return probabilities for predict; use predict_proba->argmax for safety
    pr_tr = np.argmax(xgb.predict_proba(X_train), axis=1)
    pr_va = np.argmax(xgb.predict_proba(X_val), axis=1)
    pr_te = np.argmax(xgb.predict_proba(X_test), axis=1)
    evaluate("XGBoost", y_train, pr_tr, y_val, pr_va, y_test, pr_te)
    # Importance (version-robust): use sklearn's feature_importances_
    imp = pd.Series(xgb.feature_importances_, index=feat_cols)
    return imp.sort_values(ascending=False)


def main():
    df = load_data()
    parts = split(df)
    # Prepare splits
    X_train, y_train, feat_cols = get_Xy(parts["train"], TARGET)
    X_val, y_val, _ = get_Xy(parts["validation"], TARGET)
    X_test, y_test, _ = get_Xy(parts["test"], TARGET)

    # Pipelines
    scaler = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    no_scale = Pipeline([("impute", SimpleImputer(strategy="median"))])

    # Impute/scale
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_ns = no_scale.fit_transform(X_train)
    X_val_ns = no_scale.transform(X_val)
    X_test_ns = no_scale.transform(X_test)

    # Models
    rf_imp = train_random_forest(X_train_ns, y_train, X_val_ns, y_val, X_test_ns, y_test, feat_cols)
    xgb_imp = train_xgboost(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, feat_cols)

    # Compare top features from best model (by validation macro-F1)
    # Simple heuristic: choose model with higher validation F1 reported above.
    # Save feature importances
    out_dir = PROJECT_DIR / "output" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    rf_imp.head(50).to_csv(out_dir / "rf_feature_importance.csv")
    xgb_imp.head(50).to_csv(out_dir / "xgb_feature_importance.csv")
    print("\nSaved feature importances to:", out_dir)

if __name__ == "__main__":
    main()

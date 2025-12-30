import os
import sys
import json
import math
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join("output", "models")
EXPLAIN_DIR = os.path.join("output", "explainability")
PROCESSED_DIR = os.path.join("output", "processed")

TARGET_COL = "delay_risk_24h"
DEFAULT_WINDOW = int(os.environ.get("LSTM_WINDOW", "24"))
BATCH_SIZE = int(os.environ.get("LSTM_BATCH", "64"))
EPOCHS = int(os.environ.get("LSTM_EPOCHS", "20"))
LR = float(os.environ.get("LSTM_LR", "1e-3"))
HIDDEN = int(os.environ.get("LSTM_HIDDEN", "64"))
LAYERS = int(os.environ.get("LSTM_LAYERS", "1"))
DROPOUT = float(os.environ.get("LSTM_DROPOUT", "0.1"))
SEED = int(os.environ.get("SEED", "42"))
OVERRIDE_TS = os.environ.get("LSTM_TS_COL", "").strip()
OVERRIDE_GROUP = os.environ.get("LSTM_GROUP_COL", "").strip()

np.random.seed(SEED)


def safe_try_import_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torch.utils.data as tud  # noqa: F401
        return True
    except Exception as e:
        sys.stdout.write(
            f"PyTorch is not installed. Skipping LSTM training. To enable, install torch. Reason: {type(e).__name__}: {e}\n"
        )
        return False


def find_delay_risk_dataset() -> str:
    if not os.path.isdir(PROCESSED_DIR):
        return ""
    candidates = []
    for name in os.listdir(PROCESSED_DIR):
        if not name.lower().endswith(".csv"):
            continue
        if "delay" in name.lower() or "risk" in name.lower():
            candidates.append(os.path.join(PROCESSED_DIR, name))
    # Fallback: consider all CSVs and inspect columns
    others = [os.path.join(PROCESSED_DIR, n) for n in os.listdir(PROCESSED_DIR) if n.lower().endswith(".csv")]
    for path in candidates + others:
        try:
            df_head = pd.read_csv(path, nrows=200)
            if TARGET_COL in df_head.columns:
                return path
        except Exception:
            continue
    return ""


def select_time_and_group_cols(df: pd.DataFrame) -> Tuple[str, str]:
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or c.lower().endswith("_ts")]
    ts_col = time_cols[0] if time_cols else ""
    group_cands = [
        c for c in df.columns
        if any(k in c.lower() for k in ["vessel", "ship", "service", "port", "yard", "berth", "terminal", "lane", "asset", "id"]) \
        and c != TARGET_COL
    ]
    group_col = group_cands[0] if group_cands else ""
    return ts_col, group_col


def build_sequences(df: pd.DataFrame, features: List[str], target: str, ts_col: str, group_col: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if ts_col and ts_col in df.columns:
        try:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values([ts_col] + ([group_col] if group_col else []))
        except Exception:
            df = df.sort_index()
    else:
        df = df.sort_index()

    X_list, y_list = [], []

    if group_col and group_col in df.columns:
        for _, g in df.groupby(group_col, sort=False):
            g = g.dropna(subset=[target])
            vals = g[features].values
            ys = g[target].astype(int).values
            if len(vals) <= window:
                continue
            for i in range(window, len(vals)):
                X_list.append(vals[i-window:i])
                y_list.append(ys[i])
        # Fallback: if no sequences found with grouping, try ungrouped
        if not X_list:
            g = df.dropna(subset=[target])
            vals = g[features].values
            ys = g[target].astype(int).values
            if len(vals) > window:
                for i in range(window, len(vals)):
                    X_list.append(vals[i-window:i])
                    y_list.append(ys[i])
    else:
        g = df.dropna(subset=[target])
        vals = g[features].values
        ys = g[target].astype(int).values
        if len(vals) > window:
            for i in range(window, len(vals)):
                X_list.append(vals[i-window:i])
                y_list.append(ys[i])

    if not X_list:
        return np.empty((0, window, len(features))), np.empty((0,), dtype=int)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    return X, y


def time_aware_split(X: np.ndarray, y: np.ndarray, val_ratio=0.15, test_ratio=0.15):
    n = len(X)
    if n == 0:
        return (X, y, X, y, X, y)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y: np.ndarray) -> List[float]:
    if len(y) == 0:
        return [1.0, 1.0, 1.0]
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    weights = counts.sum() / (len(classes) * counts)
    wmap = {c: float(w) for c, w in zip(classes, weights)}
    return [wmap.get(i, 1.0) for i in range(int(classes.max()) + 1)]


def train_lstm():
    if not safe_try_import_torch():
        return 0
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    data_path = find_delay_risk_dataset()
    if not data_path:
        sys.stdout.write("No dataset with delay_risk_24h found under output/processed. Skipping LSTM.\n")
        return 0

    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        sys.stdout.write("Target delay_risk_24h not found. Skipping LSTM.\n")
        return 0

    ts_col, group_col = select_time_and_group_cols(df)
    # Apply optional overrides if provided and valid
    if OVERRIDE_TS and OVERRIDE_TS in df.columns:
        ts_col = OVERRIDE_TS
    if OVERRIDE_GROUP and OVERRIDE_GROUP in df.columns:
        group_col = OVERRIDE_GROUP
    # Informational prints for transparency
    sys.stdout.write(
        f"Detected time column: '{ts_col or 'None'}', group column: '{group_col or 'None'}', window: {DEFAULT_WINDOW}\n"
    )

    exclude_cols = {TARGET_COL}
    exclude_cols.update([c for c in df.columns if "wait" in c.lower()])
    cat_like = df.select_dtypes(include=["object", "category"]).columns.tolist()
    exclude_cols.update(cat_like)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = list(pd.Index(feature_cols).intersection(df.select_dtypes(include=[np.number]).columns))

    if not feature_cols:
        sys.stdout.write("No numeric features available for LSTM. Skipping.\n")
        return 0

    X, y = build_sequences(df, feature_cols, TARGET_COL, ts_col, group_col, DEFAULT_WINDOW)
    if len(X) == 0:
        sys.stdout.write("Insufficient sequence data after windowing. Try reducing LSTM_WINDOW (e.g., 8 or 4). Skipping.\n")
        return 0

    X_train, y_train, X_val, y_val, X_test, y_test = time_aware_split(X, y)

    n_classes = int(np.max(y)) + 1
    class_weights = torch.tensor(compute_class_weights(y_train), dtype=torch.float32)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden, layers, dropout, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=(dropout if layers > 1 else 0.0))
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            logits = self.fc(out)
            return logits

    model = LSTMClassifier(input_dim=X.shape[-1], hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, num_classes=n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                ys.append(yb.cpu().numpy())
                ps.append(pred.cpu().numpy())
        if not ys:
            return 0.0, 0.0
        ytrue = np.concatenate(ys)
        ypred = np.concatenate(ps)
        acc = (ytrue == ypred).mean() if len(ytrue) else 0.0
        # macro F1 (simple, avoiding sklearn hard dep here)
        f1s = []
        for c in range(n_classes):
            tp = np.sum((ytrue == c) & (ypred == c))
            fp = np.sum((ytrue != c) & (ypred == c))
            fn = np.sum((ytrue == c) & (ypred != c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        return acc, macro_f1

    best_val = -1.0
    patience, best_epoch = 5, -1
    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc, val_f1 = eval_loader(val_loader)
        avg_loss = total_loss / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": X.shape[-1],
                "hidden": HIDDEN,
                "layers": LAYERS,
                "dropout": DROPOUT,
                "n_classes": n_classes,
                "window": DEFAULT_WINDOW,
                "features": feature_cols,
            }, os.path.join(OUTPUT_DIR, "lstm_delay_risk.pt"))
        if epoch - best_epoch >= patience:
            break

    # Final test
    # Reload best
    best_path = os.path.join(OUTPUT_DIR, "lstm_delay_risk.pt")
    if os.path.exists(best_path):
        chk = torch.load(best_path, map_location=device)
        model.load_state_dict(chk["model_state"])
    test_acc, test_f1 = eval_loader(test_loader)

    metrics = {
        "best_val_f1": round(float(best_val), 4),
        "test_acc": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "epochs_trained": len(history["epoch"]),
        "window": DEFAULT_WINDOW,
        "features": feature_cols,
        "dataset": os.path.relpath(data_path).replace("\\", "/"),
        "group_col": group_col,
        "time_col": ts_col,
    }
    with open(os.path.join(OUTPUT_DIR, "lstm_delay_risk_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Optional simple learning curve CSV
    try:
        pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "lstm_delay_risk_history.csv"), index=False)
    except Exception:
        pass

    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        exit_code = train_lstm()
        sys.exit(exit_code)
    except Exception as e:
        # Hard safety: never crash the pipeline
        sys.stdout.write(f"LSTM script encountered a non-fatal issue and exited gracefully: {type(e).__name__}: {e}\n")
        sys.exit(0)

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "output" / "processed"
EDA_DIR = PROJECT_DIR / "output" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

parquet_path = PROCESSED_DIR / "time_aligned_processed.parquet"
csv_path = PROCESSED_DIR / "time_aligned_processed.csv"

if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
elif csv_path.exists():
    df = pd.read_csv(csv_path, low_memory=False)
else:
    raise FileNotFoundError("time_aligned_processed not found in output/processed")

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

for c in [
    "ata","arrival_time","arrived","arrival","berth_start","atb","departure_time","atd","timestamp","time"
]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

time_col = next((c for c in [
    "ata","arrival_time","arrived","arrival","berth_start","atb","departure_time","atd"
] if c in df.columns), None)

missing_cnt = df.isna().sum()
missing_pct = df.isna().mean() * 100
missing = pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
missing.sort_values("missing_count", ascending=False).to_csv(EDA_DIR / "missing_values.csv")

dup_count = int(df.duplicated().sum())
(Path(EDA_DIR / "duplicates_count.txt")).write_text(str(dup_count))

checks = {}
for col in ["berth_wait_hrs","service_time_hrs","turnaround_hrs"]:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        checks[f"{col}_negative"] = int((s < 0).sum())

if "yard_utilization_ratio" in df.columns:
    y = pd.to_numeric(df["yard_utilization_ratio"], errors="coerce")
    checks["yard_utilization_gt1"] = int((y > 1.0).sum())
    checks["yard_utilization_lt0"] = int((y < 0.0).sum())

for c in [c for c in df.columns if "utilization_ratio" in c and c != "yard_utilization_ratio"]:
    s = pd.to_numeric(df[c], errors="coerce")
    checks[f"{c}_gt1"] = int((s > 1.0).sum())
    checks[f"{c}_lt0"] = int((s < 0.0).sum())

(Path(EDA_DIR / "unrealistic_checks.json")).write_text(json.dumps(checks, indent=2))

key_cols = [
    "berth_wait_hrs","service_time_hrs","turnaround_hrs","congestion_index","yard_utilization_ratio","crane_utilization_ratio"
]
present = [c for c in key_cols if c in df.columns]
if present:
    stats = df[present].describe(percentiles=[.05,.25,.5,.75,.95]).T
    stats.to_csv(EDA_DIR / "summary_stats.csv")

numeric_df = df.select_dtypes(include=[np.number]).copy()
if not numeric_df.empty:
    corr = numeric_df.corr(numeric_only=True)
    corr.to_csv(EDA_DIR / "correlations.csv")
    plt.figure(figsize=(12,9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=False)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

plot_cols = [c for c in present if c != "congestion_index"] + ["congestion_index"]
for c in plot_cols:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        plt.figure(figsize=(8,4))
        sns.histplot(s.dropna(), kde=True)
        plt.title(f"Histogram: {c}")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"hist_{c}.png", dpi=150)
        plt.close()
        plt.figure(figsize=(6,4))
        sns.boxplot(x=s)
        plt.title(f"Boxplot: {c}")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"box_{c}.png", dpi=150)
        plt.close()

if time_col:
    ts = df[[time_col]].dropna().sort_values(time_col).copy()
    if not ts.empty:
        ts["count"] = 1
        daily_arrivals = ts.set_index(time_col)["count"].resample("D").sum()
        plt.figure(figsize=(12,4))
        daily_arrivals.plot()
        plt.title("Daily vessel arrivals")
        plt.xlabel("date")
        plt.ylabel("arrivals")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "ts_daily_arrivals.png", dpi=150)
        plt.close()

if time_col and "berth_wait_hrs" in df.columns:
    s = df.set_index(time_col).sort_index()["berth_wait_hrs"].astype(float)
    daily = s.resample("D").mean()
    plt.figure(figsize=(12,4))
    daily.plot()
    plt.title("Daily mean berth_wait_hrs")
    plt.xlabel("date")
    plt.ylabel("hours")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "ts_daily_berth_wait.png", dpi=150)
    plt.close()

if time_col and "congestion_index" in df.columns:
    s = df.set_index(time_col).sort_index()["congestion_index"].astype(float)
    daily = s.resample("D").mean()
    plt.figure(figsize=(12,4))
    daily.plot()
    plt.title("Daily mean congestion_index")
    plt.xlabel("date")
    plt.ylabel("index")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "ts_daily_congestion_index.png", dpi=150)
    plt.close()

print("EDA complete. Outputs saved to:", EDA_DIR)

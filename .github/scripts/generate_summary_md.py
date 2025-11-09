#!/usr/bin/env python3
"""
Print a rich Pipeline Summary to $GITHUB_STEP_SUMMARY, matching the POC-Mobile layout.

It looks for artifacts in this order (first found wins):
  CSV:      .mlops/commitHistory.csv  → artifacts/commitHistory.csv → commitHistory.csv
  CHART:    .mlops/val_accuracy.png   → .mlops/val_accuracy.svg
            → artifacts/val_accuracy.png → artifacts/val_accuracy.svg
            → val_accuracy.png → val_accuracy.svg

It keeps headers stable across repos.
"""

import os
import sys
import io
from datetime import datetime
import pandas as pd

# ---------- Helpers ----------
def find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def ensure_toronto_timestamp(ts):
    # Accepts string or pd.Timestamp; prints as "YYYY-MM-DD HH:MM"
    try:
        if pd.isna(ts):
            return ""
        if isinstance(ts, pd.Timestamp):
            t = ts
        else:
            t = pd.to_datetime(ts, errors="coerce", utc=True)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        toronto = t.tz_convert("America/Toronto")
        return toronto.strftime("%Y-%m-%d %H:%M")
    except Exception:
        # Best effort
        try:
            return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts) if ts is not None else ""

def to_md_table(df):
    if df.empty:
        return "_No data._\n"
    buf = io.StringIO()
    # escape pipes in cells
    safe = df.copy()
    for c in safe.columns:
        safe[c] = safe[c].astype(str).str.replace("|", r"\|", regex=False)
    header = "| " + " | ".join(safe.columns) + " |"
    sep = "|" + "|".join(["---"] * len(safe.columns)) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in safe.astype(str).values.tolist()]
    buf.write(header + "\n")
    buf.write(sep + "\n")
    buf.write("\n".join(rows) + "\n")
    return buf.getvalue()

def format_params_row(row):
    # Collect common hyperparams if present, otherwise fallback to any param_* columns.
    keys_preferred = [
        "epochs","opt_name","opt_learning_rate","monitor",
        "patience","restore_best_weights","batch_size","lr","optimizer"
    ]
    parts = []
    for k in keys_preferred:
        if k in row and pd.notna(row[k]) and str(row[k]) != "":
            parts.append(f"{k}: {row[k]}")
    # Any param_* columns (param_foo) get included as foo: value
    for c in row.index:
        if c.startswith("param_") and pd.notna(row[c]) and str(row[c]) != "":
            parts.append(f"{c[6:]}: {row[c]}")
    if not parts and "parameters" in row and pd.notna(row["parameters"]):
        # raw JSON/YAML string if available
        return str(row["parameters"])
    return "\n".join(parts) if parts else "—"

# ---------- Locate inputs ----------
csv_path = find_first([
    ".mlops/commitHistory.csv", "artifacts/commitHistory.csv", "commitHistory.csv"
])
chart_path = find_first([
    ".mlops/val_accuracy.png", ".mlops/val_accuracy.svg",
    "artifacts/val_accuracy.png", "artifacts/val_accuracy.svg",
    "val_accuracy.png", "val_accuracy.svg"
])

# ---------- Read CSV (tolerant) ----------
cols_target_latest = [
    "Timestamp (Toronto)","Branch","Author","Cause",
    "val_accuracy","Δval_accuracy","model_version","run_id",
    "train_min","status","trained"
]
cols_fallback_map = {
    # map your lite CSV -> target columns when possible
    "commit_id": None, "branch": "Branch", "status": "status", "trained": "trained",
    "cause": "Cause", "model_ver": "model_version", "run_id": "run_id",
    "val_acc": "val_accuracy", "train_min": "train_min", "author": "Author",
    "timestamp": "Timestamp (Toronto)",
}

df = pd.DataFrame()
if csv_path and os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        pass

# Normalize columns
if not df.empty:
    # Create target columns
    for col in cols_target_latest:
        if col not in df.columns:
            # try to fill from fallback map
            source = [k for k,v in cols_fallback_map.items() if v == col]
            filled = False
            for s in source:
                if s in df.columns:
                    df[col] = df[s]
                    filled = True
                    break
            if not filled:
                df[col] = ""

    # Timestamp formatting
    # Prefer any 'Timestamp (Toronto)' already present; else derive from 'timestamp' or MLflow start_time
    if df["Timestamp (Toronto)"].eq("").all():
        maybe = None
        for cand in ["timestamp","start_time","end_time","datetime","time"]:
            if cand in df.columns:
                maybe = cand
                break
        if maybe:
            df["Timestamp (Toronto)"] = df[maybe].apply(ensure_toronto_timestamp)
        else:
            df["Timestamp (Toronto)"] = ""

    # Compute Δval_accuracy per branch
    try:
        tmp = df.copy()
        tmp["val_accuracy"] = pd.to_numeric(tmp["val_accuracy"], errors="coerce")
        tmp.sort_values(by=["Branch","Timestamp (Toronto)"], inplace=True)
        tmp["prev"] = tmp.groupby("Branch")["val_accuracy"].shift(1)
        tmp["Δval_accuracy"] = (tmp["val_accuracy"] - tmp["prev"]).round(3)
        # merge deltas back on index
        df["Δval_accuracy"] = tmp["Δval_accuracy"]
    except Exception:
        pass

# ---------- Section 1: Model from this workflow run ----------
# pick the latest successful row if available; else the latest row.
latest_row = None
if not df.empty:
    # sort by timestamp-like column if possible
    try:
        order = pd.to_datetime(df["Timestamp (Toronto)"], errors="coerce")
        df = df.loc[order.sort_values(ascending=False).index]
    except Exception:
        pass
    cand = df[(df["status"].astype(str).str.lower() == "success") & (df["trained"].astype(str).str.lower() == "true")]
    latest_row = cand.iloc[0] if not cand.empty else df.iloc[0]

# Build section 1 table
sec1_cols = ["Timestamp (Toronto)","model_version","Cause","Parameters","accuracy"]
sec1_df = pd.DataFrame(columns=sec1_cols)
if latest_row is not None:
    params = format_params_row(latest_row)
    accuracy = latest_row["val_accuracy"] if "val_accuracy" in latest_row.index else latest_row.get("accuracy","")
    sec1_df.loc[0] = [
        latest_row.get("Timestamp (Toronto)",""),
        latest_row.get("model_version",""),
        latest_row.get("Cause",""),
        params,
        accuracy if pd.notna(accuracy) else ""
    ]

# ---------- Section 2: Chart ----------
chart_md = "_No chart generated._"
if chart_path:
    chart_md = f"![val_accuracy]({chart_path})"

# ---------- Section 3: Latest 10 runs ----------
sec3_df = pd.DataFrame(columns=cols_target_latest)
if not df.empty:
    take = df[cols_target_latest].copy()
    # prettify TS
    take["Timestamp (Toronto)"] = take["Timestamp (Toronto)"].apply(ensure_toronto_timestamp)
    sec3_df = take.head(10)

# ---------- Emit Markdown ----------
out = io.StringIO()
out.write("# Pipeline Summary\n\n")
out.write("## 1) Model in APK: from this workflow run\n\n")
out.write(to_md_table(sec1_df))
out.write("\n")
out.write("## 2) Model Performance (val_accuracy)\n\n")
out.write(chart_md + "\n\n")
out.write("## 3) Latest 10 runs\n\n")
out.write(to_md_table(sec3_df))

sys.stdout.write(out.getvalue())

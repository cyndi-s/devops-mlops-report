#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs(".mlops", exist_ok=True)

# Find CSV
csv_candidates = [".mlops/commitHistory.csv", "artifacts/commitHistory.csv", "commitHistory.csv"]
csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
if not csv_path:
    print("No commit history found; skipping chart.")
    raise SystemExit(0)

df = pd.read_csv(csv_path)
if "val_acc" in df.columns and "val_accuracy" not in df.columns:
    df["val_accuracy"] = pd.to_numeric(df["val_acc"], errors="coerce")

if "val_accuracy" not in df.columns:
    print("No val_accuracy column; skipping chart.")
    raise SystemExit(0)

# Sort by time if available
time_col = None
for c in ["Timestamp (Toronto)","timestamp","start_time","datetime","time"]:
    if c in df.columns:
        time_col = c
        break
if time_col:
    try:
        df["_t"] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values("_t")
    except Exception:
        pass

y = pd.to_numeric(df["val_accuracy"], errors="coerce")
x = range(len(y))

# Compute 3-run MA
ma3 = y.rolling(3).mean()
median = y.median()

# Plot
plt.figure()
plt.plot(x, y, marker="o", label="val_accuracy")
plt.axhline(median, linestyle="--", label="Median")
plt.plot(x, ma3, linestyle=":", label="3-run MA")
plt.legend()
plt.xlabel("Run # (old → new)")
plt.ylabel("val_accuracy")
plt.title("Model Performance")

# Save both
svg_path = ".mlops/val_accuracy.svg"
png_path = ".mlops/val_accuracy.png"
plt.savefig(svg_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight")
print(f"Wrote {svg_path} and {png_path}")

#!/usr/bin/env python3
import os, csv, subprocess
from pathlib import Path

CSV = Path("commitHistory.csv")

def sh(cmd): return subprocess.check_output(cmd, text=True).strip()
def _f1(v): 
    try: return f"{float(v):.1f}"
    except: return v or ""

def _f3(v):
    try: return f"{float(v):.3f}"
    except: return v or ""

# ----- git info -----
commit_timestamp = sh(["git", "log", "-1", "--date=format:%Y-%m-%d %H:%M:%S", "--pretty=%ad"])
commit_id = (os.getenv("GITHUB_SHA", "")[:7] or sh(["git", "rev-parse", "--short", "HEAD"]))
commit_message = sh(["git", "log", "-1", "--pretty=%s"]).replace(",", ";")
author = os.getenv("GITHUB_ACTOR", "") or sh(["git", "log", "-1", "--pretty=%an"])
branch = os.getenv("GITHUB_REF_NAME", "") or sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])

# ----- pipeline / mlflow envs -----
status = os.getenv("STATUS", "success")
pipeline_duration = os.getenv("PIPE_DUR", "N/A")
trained_model = os.getenv("TRAINED", "false").lower()

experiment_name = os.getenv("EXP_NAME", "Default")
run_id = os.getenv("RUN_ID", "")

training_duration = (
    os.getenv("training_duration")
    or os.getenv("TRAIN_DUR")
    or os.getenv("MLFLOW_DUR", "")
)
accuracy = os.getenv("accuracy", os.getenv("ACCURACY", ""))
loss = os.getenv("loss", os.getenv("LOSS", ""))
val_accuracy = os.getenv("val_accuracy", os.getenv("VAL_ACCURACY", ""))
val_loss = os.getenv("val_loss", os.getenv("VAL_LOSS", ""))

epochs = os.getenv("epochs", os.getenv("EPOCHS", ""))
opt_name = os.getenv("opt_name", os.getenv("OPT_NAME", ""))
opt_learning_rate = os.getenv("opt_lr", os.getenv("OPT_LR", ""))
monitor = os.getenv("monitor", os.getenv("MONITOR", ""))
patience = os.getenv("patience", os.getenv("PATIENCE", ""))
restore_best_weights = os.getenv("restore_best_weights", os.getenv("RESTORE_BEST_WEIGHTS", ""))
model_version = os.getenv("model_version", os.getenv("MODEL_VERSION", ""))

# ----- ensure header -----
header = (
    "\"commit_timestamp\n(EDT)\",commit_id,commit_message,author,branch,status,"
    "pipeline_duration,trained_model,cause_mlops,"
    "model_version,run_id,\"training_duration\n(min)\",accuracy,loss,val_accuracy,val_loss,"
    "epochs,opt_name,opt_learning_rate,monitor,patience,restore_best_weights\n"
)
if not CSV.exists():
    CSV.write_text(header, encoding="utf-8")

# ----- duplicate-commit guard -----
with open(CSV, "r", encoding="utf-8") as f:
    for line in f:
        if f",{commit_id}," in line:
            print(f"commit_id {commit_id} already logged; skip append.")
            raise SystemExit(0)

# ----- append row -----
if trained_model not in {"true", "1", "yes", "y", "on"}:
    model_version = cause_mlops = ""
    # Optional: also blank run_id so the CSV clearly shows "no new run"
    run_id = "no new run"
    training_duration = ""
    
    accuracy = loss = val_accuracy = val_loss = ""
    epochs = opt_name = opt_learning_rate = monitor = patience = restore_best_weights = ""


row = [
    commit_timestamp, commit_id, commit_message, author, branch, status, pipeline_duration, trained_model,os.getenv("CAUSE_MLOPS", "None"),
    model_version, run_id, _f1(training_duration), _f3(accuracy), _f3(loss), _f3(val_accuracy), _f3(val_loss),
    epochs, opt_name, _f3(opt_learning_rate), monitor, patience, restore_best_weights
]

with open(CSV, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)

print(f"Appended (trained_model={trained_model}) → {CSV}")

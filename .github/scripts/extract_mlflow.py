#!/usr/bin/env python3
import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

out = os.environ["GITHUB_OUTPUT"]
run_id = os.environ.get("RUN_ID", "")

TRAINED_FLAG = (os.getenv("TRAINED_THIS_RUN") or os.getenv("TRAINED") or "").lower() == "true"

# Default (empty) outputs when no run is found
empty = {
    "run_id": run_id or "",
    "accuracy": "", "loss": "", "val_accuracy": "", "val_loss": "",
    "training_duration": "", "model_version": "",
    "epochs": "", "opt_name": "", "opt_lr": "", "monitor": "",
    "patience": "", "restore_best_weights": "",
     "trained_hint": "false",
}

def ms_to_seconds(ms):
    return "" if ms is None else f"{ms/60000:.3f}"

def write_outputs(d):
    with open(out, "a",encoding="utf-8") as f:
        for k, v in d.items():
            f.write(f"{k}={v}\n")

def get_default_experiment(client: MlflowClient):
    # Your experiment name is guaranteed to be "Default"
    exp = mlflow.get_experiment_by_name("Default")
    if exp is None:
        # fallback to the built-in default experiment id "0"
        try:
            exp = client.get_experiment("0")
        except Exception:
            exp = None
    return exp

def latest_run_id(client: MlflowClient):
    exp = get_default_experiment(client)
    if not exp:
        return ""
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0].info.run_id if runs else ""

client = MlflowClient()

# Resolve run_id if not provided
if not run_id:
    run_id = latest_run_id(client)



if not run_id:
    write_outputs(empty)
    print("No MLflow run_id available; wrote empty outputs.")
    raise SystemExit(0)

# Fetch run data
run = client.get_run(run_id)
params = run.data.params or {}
metrics = run.data.metrics or {}

start_ms = run.info.start_time
end_ms = run.info.end_time
train_secs = ms_to_seconds((end_ms or 0) - (start_ms or 0))

# Resolve model version (if registered)
version = ""
if TRAINED_FLAG and run_id:
    try:
        for mv in client.search_model_versions(f"run_id = '{run_id}'"):
            version = mv.version
            break
    except Exception:
        pass

mapped = {
    "run_id": run_id,
    "accuracy": str(metrics.get("accuracy", "")),
    "loss": str(metrics.get("loss", "")),
    "val_accuracy": str(metrics.get("val_accuracy", "")),
    "val_loss": str(metrics.get("val_loss", "")),
    "training_duration": train_secs,
     "model_version": version or "",
    "epochs": params.get("epochs", ""),
    "opt_name": params.get("opt_name", ""),
    "opt_lr": params.get("opt_learning_rate", ""),
    "monitor": params.get("monitor", ""),
    "patience": params.get("patience", ""),
    "restore_best_weights": params.get("restore_best_weights", ""),
    "trained_hint": "true" if TRAINED_FLAG else "false",
   
}

write_outputs(mapped)
print("Extracted MLflow params/metrics/version")

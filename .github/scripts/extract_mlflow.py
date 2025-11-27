#!/usr/bin/env python3
import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

GITHUB_OUTPUT = os.environ["GITHUB_OUTPUT"]

def write_outputs(values: dict) -> None:
    """Write key=value pairs to GITHUB_OUTPUT for later steps."""
    with open(GITHUB_OUTPUT, "a", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")


def get_latest_run_id(client: MlflowClient) -> str:
    """Return run_id from env if provided, else latest run across all experiments."""
    run_id_env = (os.getenv("RUN_ID") or "").strip()
    if run_id_env:
        return run_id_env

    # search latest run across all experiments
    try:
        df = mlflow.search_runs(
            search_all_experiments=True,
            max_results=1,
            order_by=["start_time DESC"],
        )
    except TypeError:
        # fallback for older mlflow
        df = mlflow.search_runs(
            experiment_ids=None,
            max_results=1,
            order_by=["start_time DESC"],
        )

    if df is None or df.empty:
        return ""
    return str(df.loc[0, "run_id"])


def main() -> None:
    trained_flag = (os.getenv("TRAINED_THIS_RUN") or os.getenv("TRAINED") or "").lower() == "true"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    run_id = get_latest_run_id(client)
    if not run_id:
        # no run found – emit empty values so downstream code shows nan
        write_outputs({
            "run_id": "",
            "accuracy": "",
            "loss": "",
            "val_accuracy": "",
            "val_loss": "",
            "training_duration": "",
            "model_version": "",
            "epochs": "",
            "opt_name": "",
            "opt_learning_rate": "",
            "monitor": "",
            "patience": "",
            "restore_best_weights": "",
            "trained_hint": "true" if trained_flag else "false",
        })
        print("No MLflow run found.")
        return

    run = client.get_run(run_id)
    metrics = run.data.metrics or {}
    params = run.data.params or {}
    info = run.info

    # ---- duration (min) from start/end times (ms since epoch) ----
    duration_min = ""
    if info.start_time and info.end_time:
        duration_sec = (info.end_time - info.start_time) / 1000.0
        duration_min = round(duration_sec / 60.0, 2)

    # ---- metrics (with simple fallbacks) ----
    accuracy = (
        metrics.get("val_accuracy")
        or metrics.get("accuracy")
        or metrics.get("acc")
        or ""
    )
    val_accuracy = (
        metrics.get("val_accuracy")
        or metrics.get("val_acc")
        or accuracy  # fallback: use accuracy if no val_ metric
    )
    loss = metrics.get("loss") or ""
    val_loss = metrics.get("val_loss") or ""

    # ---- params ----
    epochs = params.get("epochs", "")
    opt_name = params.get("opt_name", "")
    opt_lr = params.get("opt_learning_rate", "")
    monitor = params.get("monitor", "")
    patience = params.get("patience", "")
    restore_best_weights = params.get("restore_best_weights", "")

    # model_version: keep empty for now (can wire registry later)
    model_version = ""

    out = {
        "run_id": run_id,
        "accuracy": accuracy,
        "loss": loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "training_duration": duration_min,
        "model_version": model_version,
        "epochs": epochs,
        "opt_name": opt_name,
        "opt_learning_rate": opt_lr,
        "monitor": monitor,
        "patience": patience,
        "restore_best_weights": restore_best_weights,
        "trained_hint": "true" if trained_flag else "false",
    }

    write_outputs(out)
    print(f"Extracted MLflow metrics/params for run {run_id}")


if __name__ == "__main__":
    main()

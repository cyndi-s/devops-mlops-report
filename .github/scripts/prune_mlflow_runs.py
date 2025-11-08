#!/usr/bin/env python3
import os
from mlflow.tracking import MlflowClient

# Simple pruning policy:
# - Keep latest 20 runs
# - Keep all runs that produced a registered model version
# - Skip if token/URI not set

uri = os.getenv("MLFLOW_TRACKING_URI")
tok = os.getenv("MLFLOW_TOKEN") or os.getenv("DAGSHUB_TOKEN")
if not uri or not tok:
    print("No MLflow creds; skip prune.")
    raise SystemExit(0)

client = MlflowClient()

exp = client.get_experiment_by_name("Default") or client.get_experiment("0")
if not exp:
    print("No experiment; skip prune.")
    raise SystemExit(0)

runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=200)
keep_ids = set()

# Keep runs with model versions
try:
    mvs = client.search_model_versions("name LIKE '%'")
    for mv in mvs:
        keep_ids.add(mv.run_id)
except Exception:
    pass

# Keep latest 20
for r in runs[:20]:
    keep_ids.add(r.info.run_id)

# Mark others for deletion
to_delete = [r for r in runs if r.info.run_id not in keep_ids]

print(f"Pruning {len(to_delete)} old runs (keeping {len(keep_ids)}).")
for r in to_delete:
    try:
        client.delete_run(r.info.run_id)
    except Exception as e:
        print(f"Skip delete {r.info.run_id}: {e}")

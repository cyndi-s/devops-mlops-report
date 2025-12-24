import argparse
import os
from typing import List

import yaml


MAX_RUNS_PRIVATE = 90


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def try_import_mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
        return mlflow, MlflowClient
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="caller report-config.yml path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    vis = (cfg.get("repo_visibility") or "public").strip().lower()
    if vis != "private":
        print("Prune skipped (repo_visibility != private).")
        return 0

    tracking_uri = ((cfg.get("mlflow") or {}).get("tracking_uri") or "").strip()
    if not tracking_uri:
        print("Prune skipped (missing mlflow.tracking_uri in config).")
        return 0

    mlflow, MlflowClient = try_import_mlflow()
    if mlflow is None or MlflowClient is None:
        print("Prune skipped (mlflow not available).")
        return 0

    # auth comes from env:
    # MLFLOW_TRACKING_USERNAME (non-secret) + MLFLOW_TRACKING_PASSWORD (DAGSHUB_TOKEN)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # prune across all experiments, but only delete ONE run total to be safe
        exps = client.search_experiments()
        deleted = 0

        for exp in exps:
            if deleted >= 1:
                break

            exp_id = exp.experiment_id
            runs = client.search_runs(
                experiment_ids=[exp_id],
                order_by=["attributes.start_time ASC"],  # oldest first
                max_results=1000,
            )

            if len(runs) <= MAX_RUNS_PRIVATE:
                continue

            # delete the oldest run (index 0)
            oldest = runs[0]
            rid = oldest.info.run_id
            try:
                client.delete_run(rid)
                deleted += 1
                print(f"Pruned 1 oldest run in exp {exp_id}: run_id={rid} (count was {len(runs)})")
            except Exception as e:
                print(f"Failed to delete run {rid}: {e}")

        if deleted == 0:
            print("No pruning needed (all experiments <= 90 runs).")

        return 0

    except Exception as e:
        print(f"Prune failed: {e}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

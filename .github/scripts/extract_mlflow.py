import argparse
import json
import os
from typing import List, Dict, Any

import yaml


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
        return mlflow, MlflowClient
    except Exception:
        return None, None


def _load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="report-config.yml path (caller repo checkout)")
    ap.add_argument("--sha", required=True, help="GITHUB_SHA")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    # You should already have dagshub/mlflow fields (even if empty).
    # We'll look for these keys (adjust if your config uses different names):
    tracking_uri = (
        (cfg.get("mlflow") or {}).get("tracking_uri")
        or (cfg.get("dagshub") or {}).get("mlflow_tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI", "")
    )

    sha = (args.sha or "").strip()

    payload: Dict[str, Any] = {
        "is_trained": "No",
        "run_id": "",
        "reason": "",
        "tracking_uri": tracking_uri,
        "candidates": [],
    }

    mlflow, MlflowClient = _try_import_mlflow()
    if mlflow is None or MlflowClient is None:
        payload["reason"] = "mlflow package not available in runner environment"
    elif not tracking_uri:
        payload["reason"] = "missing tracking_uri (config.mlflow.tracking_uri or config.dagshub.mlflow_tracking_uri)"
    elif not sha:
        payload["reason"] = "missing sha"
    else:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Search across all experiments.
        exps = client.search_experiments(view_type=1)  # ACTIVE_ONLY

        # MLflow commonly stores git commit in this standard tag:
        # tags.mlflow.source.git.commit
        # We also try common custom tag/param names.
        filters = [
            f"tags.mlflow.source.git.commit = '{sha}'",
            f"tags.commit_id = '{sha}'",
            f"tags.commit = '{sha}'",
            f"params.commit_id = '{sha}'",
            f"params.commit = '{sha}'",
        ]

        run_ids = set()
        candidates: List[Dict[str, str]] = []

        for exp in exps:
            exp_id = exp.experiment_id
            for flt in filters:
                try:
                    df = mlflow.search_runs(
                        experiment_ids=[exp_id],
                        filter_string=flt,
                        output_format="pandas",
                        max_results=50,
                    )
                except Exception:
                    continue

                if df is None or df.empty:
                    continue

                # run_id column is standard in mlflow search_runs output
                for rid in df["run_id"].astype(str).tolist():
                    if rid and rid not in run_ids:
                        run_ids.add(rid)
                        candidates.append({"experiment_id": exp_id, "run_id": rid, "filter": flt})

        payload["candidates"] = candidates

        if len(run_ids) == 1:
            payload["is_trained"] = "Yes"
            payload["run_id"] = next(iter(run_ids))
            payload["reason"] = "unique run matched by commit sha"
        elif len(run_ids) == 0:
            payload["reason"] = "no run matched commit sha in tracking server"
        else:
            payload["reason"] = f"multiple runs matched commit sha ({len(run_ids)})"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

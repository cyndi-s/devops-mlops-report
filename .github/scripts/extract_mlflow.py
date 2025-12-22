import argparse
import json
import os
from typing import Dict, Any

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


def _get_tracking_uri(cfg: dict) -> str:
    # Prefer config, fallback to env
    return (
        (cfg.get("mlflow") or {}).get("tracking_uri")
        or (cfg.get("dagshub") or {}).get("mlflow_tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI", "")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="report-config.yml path (caller repo checkout)")
    ap.add_argument("--sha", required=True, help="GITHUB_SHA (full 40-char)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    tracking_uri = _get_tracking_uri(cfg)
    sha = (args.sha or "").strip()

    payload: Dict[str, Any] = {
        "is_trained": "No",
        "run_id": "",
        "reason": "",
        "tracking_uri": tracking_uri,
        "filter": "",
        "matches": [],
    }

    mlflow, MlflowClient = _try_import_mlflow()
    if mlflow is None or MlflowClient is None:
        payload["reason"] = "mlflow package not available in runner environment"
    elif not tracking_uri:
        payload["reason"] = "missing tracking_uri (config.mlflow.tracking_uri or env MLFLOW_TRACKING_URI)"
    elif not sha:
        payload["reason"] = "missing sha"
    else:
        mlflow.set_tracking_uri(tracking_uri)

        # Standard tag we agreed on:
        # training MUST log: mlflow.set_tag("mlflow.source.git.commit", GITHUB_SHA)
        flt = f"tags.mlflow.source.git.commit = '{sha}'"
        payload["filter"] = flt

        try:
            # Search across all experiments: MLflow supports experiment_ids=None
            # but behavior differs by version; we do client search_experiments for compatibility.
            client = MlflowClient()
            exps = client.search_experiments(view_type=1)  # ACTIVE_ONLY

            run_ids: list[str] = []
            for exp in exps:
                exp_id = exp.experiment_id
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

                # collect run_id
                for rid in df["run_id"].astype(str).tolist():
                    rid = rid.strip()
                    if rid and rid not in run_ids:
                        run_ids.append(rid)

            payload["matches"] = run_ids

            if len(run_ids) == 1:
                payload["is_trained"] = "Yes"
                payload["run_id"] = run_ids[0]
                payload["reason"] = "unique run matched by tags.mlflow.source.git.commit"
            elif len(run_ids) == 0:
                payload["reason"] = "no run matched commit sha (missing tag or training not logged to backend)"
            else:
                payload["reason"] = f"multiple runs matched commit sha ({len(run_ids)})"

        except Exception as e:
            payload["reason"] = f"mlflow query failed: {type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

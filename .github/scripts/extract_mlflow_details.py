import argparse
import json
import os
import sys
from typing import Dict, Any

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def kv_string(d: Dict[str, Any]) -> str:
    # stable, readable "k=v; k2=v2"
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        items.append(f"{k}={v}")
    return "; ".join(items)

def format_duration_ms(ms: int | None) -> str:
    if not ms or ms <= 0:
        return ""
    total_sec = int(ms // 1000)
    if total_sec < 60:
        return f"{total_sec}s"
    m, s = divmod(total_sec, 60)
    return f"{m}m {s}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="report-config.yml path")
    ap.add_argument("--run-id", required=True, help="MLflow run_id")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # tracking uri from config or env
    tracking_uri = (
        (cfg.get("mlflow") or {}).get("tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI", "")
    )

    payload: Dict[str, Any] = {
        "run_id": args.run_id,
        "experiment_id": "",
        "params": {},
        "metrics": {},
        "params_kv": "",
        "metrics_kv": "",
        "duration": "",
        "reason": "",
    }

    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        run = client.get_run(args.run_id)
        info = run.info

        duration_ms = None
        if info.start_time and info.end_time:
            duration_ms = info.end_time - info.start_time
        duration_str = format_duration_ms(duration_ms)


        payload["experiment_id"] = run.info.experiment_id
        payload["params"] = dict(run.data.params or {})
        payload["metrics"] = dict(run.data.metrics or {})

        payload["params_kv"] = kv_string(payload["params"])
        payload["metrics_kv"] = kv_string(payload["metrics"])
        payload["duration"] = duration_str 
        payload["reason"] = "ok"

    except Exception as e:
        payload["reason"] = f"failed: {type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

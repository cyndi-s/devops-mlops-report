import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="report-config.yml path (caller repo)")
    ap.add_argument("--caller-root", required=True, help="caller repo absolute/relative path")
    ap.add_argument("--cause", default="", help="Data/Script/Both/''")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    caller_root = os.path.abspath(args.caller_root)
    mlproject_path = os.path.join(caller_root, "MLproject")

    cause = (args.cause or "").strip()
    payload: Dict[str, Any] = {
        "should_train": False,
        "run_id": "",
        "reason": "",
    }

    if not os.path.exists(mlproject_path):
        payload["reason"] = "MLproject not found"
    elif not cause:
        payload["reason"] = "cause is empty (DevOps-only)"
    else:
        payload["should_train"] = True

        # MLflow must be installed in the runner environment
        import mlflow  # type: ignore

        # Optional: set tracking URI from config/env (prefer env set by workflow)
        # cfg = load_cfg(args.config)  # not required here if env already has MLFLOW_TRACKING_URI
        # If you store tracking_uri in config and want to enforce:
        # tracking_uri = (cfg.get("mlflow") or {}).get("tracking_uri", "")
        # if tracking_uri:
        #     mlflow.set_tracking_uri(tracking_uri)

        # Trigger MLflow Project entry point in caller repo.
        # This honors MLproject + conda_env if your runner supports conda env manager.
        submitted = mlflow.projects.run(
            uri=caller_root,
            entry_point="main",
            parameters={},            # keep defaults from MLproject
            env_manager="local",      # uses local python env; if you rely on conda_env, switch to "conda"
            synchronous=True,
        )

        run_id = getattr(submitted, "run_id", "") or ""
        payload["run_id"] = run_id
        if not run_id:
            payload["reason"] = "training finished but run_id not returned by MLflow Projects"
        else:
            payload["reason"] = "training triggered and run_id captured"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

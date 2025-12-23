import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

def ensure_repo_name_symlink(caller_root: str) -> str:
    """
    Some MLproject templates use paths like ../<repo_name>/src/train.py.
    In our reusable workflow, caller is checked out to <workspace>/caller.
    Create <workspace>/<repo_name> -> <workspace>/caller symlink so those paths resolve.
    Returns a compat root path to use as the MLflow project URI.
    """
    repo_full = os.environ.get("GITHUB_REPOSITORY", "")
    repo_name = repo_full.split("/")[-1] if repo_full else "repo"

    workspace = os.path.dirname(os.path.abspath(caller_root))  # parent of .../caller
    compat_root = os.path.join(workspace, repo_name)

    target = os.path.abspath(caller_root)

    if os.path.islink(compat_root):
        # ensure it points to caller_root
        try:
            if os.readlink(compat_root) != target:
                os.unlink(compat_root)
                os.symlink(target, compat_root)
        except OSError:
            pass
    elif os.path.exists(compat_root):
        # exists but is not a symlink (folder/file). leave it; fallback to caller_root
        return target
    else:
        try:
            os.symlink(target, compat_root)
        except FileExistsError:
            pass

    return compat_root

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
        compat_root = ensure_repo_name_symlink(caller_root)

        try:
            submitted = mlflow.projects.run(
                uri=compat_root,
                entry_point="main",
                parameters={},
                env_manager="local",
                synchronous=True,
            )
            run_id = getattr(submitted, "run_id", "") or ""
            payload["run_id"] = run_id
            payload["reason"] = "training triggered and run_id captured" if run_id else \
                "training finished but run_id not returned by MLflow Projects"
        except Exception as e:
            payload["should_train"] = True
            payload["run_id"] = ""
            payload["reason"] = f"training failed: {type(e).__name__}: {e}"
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

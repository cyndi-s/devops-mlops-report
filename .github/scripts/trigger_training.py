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

    cause = (args.cause or "").strip()
    payload: Dict[str, Any] = {
        "should_train": False,
        "trained": False,  
        "run_id": "",
        "reason": "",
    }

    # v2: best-effort training. No hard requirement for MLproject.
    if not cause:
        payload["reason"] = "cause is empty (DevOps-only)"
        print("[trigger_training] skip: cause is empty (DevOps-only)")
    else:
        cfg = load_cfg(args.config)
        project = cfg.get("project") or {}
        train_script = (project.get("train_script") or "").strip()
        data_paths = project.get("data_paths") or []
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        data_paths = [str(x).strip() for x in data_paths if str(x).strip()]

        # If user didn't configure a script, we can't train (but we don't fail).
        if not train_script:
            payload["reason"] = "train_script not set in config (skip training)"
            print("[trigger_training] skip: train_script not set in config")
        else:
            payload["should_train"] = True

            # Best-effort: try MLflow Projects first; if unavailable, fall back to running the script directly.
            tracking_uri = ((cfg.get("mlflow") or {}).get("tracking_uri") or "").strip()
            mlproject_path = os.path.join(caller_root, "MLproject")

            try:
                import mlflow  # type: ignore

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                # If MLproject exists, we can try projects.run; otherwise skip straight to fallback.
                if os.path.exists(mlproject_path):
                    compat_root = ensure_repo_name_symlink(caller_root)
                    submitted = mlflow.projects.run(
                        uri=compat_root,
                        entry_point="main",
                        parameters={},          # v2: do not force param schema
                        env_manager="local",
                        synchronous=True,
                    )
                    run_id = getattr(submitted, "run_id", "") or ""
                    payload["run_id"] = run_id
                    payload["trained"] = True
                    payload["reason"] = "mlflow.projects.run ok" if run_id else "mlflow.projects.run ok (no run_id)"
                    print(f"[trigger_training] mlflow.projects.run ok, run_id={run_id or 'NA'}")
                else:
                    raise RuntimeError("MLproject not present; fallback to direct script execution")


            except Exception as e:
                # Fallback: run the user script directly (no forced conda/requirements).
                try:
                    import subprocess

                    script_abs = os.path.join(caller_root, train_script)
                    cmd = ["python", script_abs]
                    # minimal, optional: pass first data path as --data if user provided any
                    if data_paths:
                        cmd += ["--data", os.path.join(caller_root, data_paths[0])]

                    print(f"[trigger_training] fallback: running script: {' '.join(cmd)}")
                    subprocess.check_call(cmd, cwd=caller_root)

                    payload["trained"] = True
                    payload["run_id"] = ""
                    payload["reason"] = f"fallback script run ok (no mlflow run_id); primary error: {type(e).__name__}"
                    print("[trigger_training] fallback ok (no mlflow run_id)")
                except Exception as e2:
                    payload["trained"] = False
                    payload["run_id"] = ""
                    payload["reason"] = f"training failed (projects+fallback): {type(e).__name__}: {e} | {type(e2).__name__}: {e2}"
                    print(f"[trigger_training] failed: {payload['reason']}")


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

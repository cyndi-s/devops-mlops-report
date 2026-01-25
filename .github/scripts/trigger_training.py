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

def ensure_tmp_mlproject(caller_root: str, train_script: str, train_args: list[str]) -> str:
    """
    v2: If caller repo has no MLproject but user provided paths, create a temporary MLproject
    in the CI workspace so the rest of the pipeline always uses mlflow.projects.run().
    Returns the MLproject path (in caller workspace).
    """
    mlproject_path = os.path.join(caller_root, "MLproject")
    if os.path.exists(mlproject_path):
        return mlproject_path

    # Build command (do not guess flags; only use user-provided args)
    cmd = "python " + train_script
    if train_args:
        cmd += " " + " ".join(train_args)

    content = {
        "name": "tmp-mlproject-from-report-config",
        "entry_points": {
            "main": {
                "command": cmd
            }
        }
    }

    with open(mlproject_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)

    print(f"[trigger_training] created tmp MLproject at {mlproject_path}")
    return mlproject_path

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

         # train_script can be str or list[str]
        ts = project.get("train_script") or []
        if isinstance(ts, str):
            train_scripts = [ts]
        elif isinstance(ts, list):
            train_scripts = ts
        else:
            train_scripts = []
        train_scripts = [str(x).strip() for x in train_scripts if str(x).strip()]

        # convention: first script is the execution entry
        train_script = train_scripts[0] if train_scripts else ""
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

            # v2: Always run training via MLflow Projects.
            # If MLproject is missing but user provided paths, create a temporary MLproject in the workspace.
            tracking_uri = ((cfg.get("mlflow") or {}).get("tracking_uri") or "").strip()

            # Minimal path validation (existence only)
            script_abs = os.path.join(caller_root, train_script)
            if not os.path.isfile(script_abs):
                payload["should_train"] = False
                payload["trained"] = False
                payload["run_id"] = ""
                payload["reason"] = f"invalid config: train_script not found: {train_script}"
                print(f"[trigger_training] skip: {payload['reason']}")
            else:
                # normalize train_args
                train_args = project.get("train_args") or []
                if isinstance(train_args, str):
                    train_args = [train_args]
                train_args = [str(x).strip() for x in train_args if str(x).strip()]

                # data paths are validated lightly (existence), but not required to include in MLproject
                for dp in data_paths:
                    dp_abs = os.path.join(caller_root, dp)
                    if not os.path.exists(dp_abs):
                        print(f"[trigger_training] warn: data_path not found: {dp}")

                try:
                    import mlflow  # type: ignore

                    if tracking_uri:
                        mlflow.set_tracking_uri(tracking_uri)

                    # ensure MLproject exists (real or tmp)
                    ensure_tmp_mlproject(caller_root, train_script, train_args)

                    # compat symlink helps GoMLOps templates using ../<repo_name>/...
                    compat_root = ensure_repo_name_symlink(caller_root)

                    submitted = mlflow.projects.run(
                        uri=compat_root,
                        entry_point="main",
                        parameters={},   # v2: do not force param schema
                        env_manager="local",
                        synchronous=True,
                    )
                    run_id = getattr(submitted, "run_id", "") or ""
                    payload["run_id"] = run_id
                    payload["trained"] = True
                    payload["reason"] = "mlflow.projects.run ok" if run_id else "mlflow.projects.run ok (no run_id)"
                    print(f"[trigger_training] mlflow.projects.run ok, run_id={run_id or 'NA'}")

                except Exception as e:
                    payload["trained"] = False
                    payload["run_id"] = ""
                    payload["reason"] = f"training failed (mlflow.projects.run): {type(e).__name__}: {e}"
                    print(f"[trigger_training] failed: {payload['reason']}")


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

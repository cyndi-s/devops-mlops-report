import argparse
import json
import os
import sys
from typing import Any, Dict
import yaml

def find_run_by_short_sha(tracking_uri: str, short_sha: str) -> tuple[str, str]:
    """
    Find the MLflow run whose git commit tag starts with short_sha (first 7 of GITHUB_SHA).
    Prefer the run that has metrics/params (real training run).
    Returns (run_id, experiment_id) or ("","") if not found.
    """
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()

        exps = client.search_experiments()
        exp_ids = [e.experiment_id for e in exps] or ["0"]

        # mlflow tag key is usually: mlflow.source.git.commit
        # Filter for prefix match using LIKE.
        filt = f"tags.`mlflow.source.git.commit` LIKE '{short_sha}%'"

        best = None
        best_score = -1

        for exp_id in exp_ids:
            runs = client.search_runs(
                experiment_ids=[exp_id],
                filter_string=filt,
                order_by=["attributes.start_time DESC"],
                max_results=50,
            )
            for r in runs:
                metrics_n = len(getattr(r.data, "metrics", {}) or {})
                params_n = len(getattr(r.data, "params", {}) or {})
                score = metrics_n * 100 + params_n * 10  # metrics > params

                if score > best_score:
                    best = r
                    best_score = score

        if best:
            return best.info.run_id, best.info.experiment_id

        return "", ""
    except Exception:
        return "", ""

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

def ensure_tmp_mlproject(project_root: str, script_relpath: str, train_args: list[str]) -> str:
    """
     v2: Create a temporary MLproject in the *execution root* so user code that relies on
    relative paths (e.g., 'data/', 'models/') works as expected.

    - project_root: the directory we will run MLflow Projects from (cwd)
    - script_relpath: path to the training script *relative to project_root*
    """
    mlproject_path = os.path.join(project_root, "MLproject")
    if os.path.exists(mlproject_path):
        return mlproject_path

    cmd = "python " + script_relpath
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

                    # compat symlink helps GoMLOps templates using ../<repo_name>/...
                    compat_root = ensure_repo_name_symlink(caller_root)
                    workdir = (project.get("workdir") or "").strip().strip("/")
                    script_dir_rel = os.path.dirname(train_script).strip("/")
                    run_root_rel = script_dir_rel
                    run_root_abs = os.path.join(caller_root, run_root_rel) if run_root_rel else caller_root
                    # If workdir is provided and valid, prefer it as the execution root
                    if workdir:
                        candidate_abs = os.path.join(caller_root, workdir)
                        if os.path.isdir(candidate_abs):
                            # Only use workdir if train_script is inside it (so we can compute a clean relative path)
                            script_abs = os.path.join(caller_root, train_script)
                            try:
                                rel_to_workdir = os.path.relpath(script_abs, candidate_abs)
                            except ValueError:
                                rel_to_workdir = ""

                            if rel_to_workdir and not rel_to_workdir.startswith(".."):
                                run_root_rel = workdir
                                run_root_abs = candidate_abs
                            else:
                                print(f"[trigger_training] warn: workdir '{workdir}' does not contain train_script '{train_script}'. Ignoring workdir.")
                        else:
                            print(f"[trigger_training] warn: workdir not found: {workdir}. Ignoring workdir.")

                    # Build MLflow project URI + local path using the chosen execution root
                    uri = os.path.join(compat_root, run_root_rel) if run_root_rel else compat_root
                    project_root = run_root_abs

                    # Compute script path relative to execution root (what goes into MLproject command)
                    script_abs = os.path.join(caller_root, train_script)
                    script_rel_for_cmd = os.path.relpath(script_abs, project_root)

                    # Ensure MLproject exists in the execution root (real or tmp)
                    ensure_tmp_mlproject(project_root, script_rel_for_cmd, train_args)

                    from subprocess import check_call
                    cmd = ["python", script_rel_for_cmd] + train_args
                    check_call(cmd, cwd=project_root)
                    # Match MLflow run by commit (first 7 chars of GITHUB_SHA)
                    full_sha = (os.environ.get("GITHUB_SHA") or "").strip()
                    short_sha = full_sha[:7] if len(full_sha) >= 7 else full_sha
                    rid, exp_id = find_run_by_short_sha(tracking_uri, short_sha)

                    payload["run_id"] = rid
                    payload["trained"] = True if rid else False
                    payload["reason"] = f"training executed; mlflow run selected by commit tag prefix: {short_sha}"
                    print(f"[trigger_training] training ok, run_id={rid or '(missing)'}")


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

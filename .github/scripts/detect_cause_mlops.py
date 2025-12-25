import argparse
import json
import os
import re
import subprocess

import yaml


def path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def run_git(cwd: str, args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore").strip()


def get_changed_files_single_commit(repo_dir: str, sha: str) -> list[str]:
    try:
        txt = run_git(repo_dir, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
        return [f.strip() for f in txt.splitlines() if f.strip()]
    except Exception:
        return []


def parse_mlproject_for_script(mlproject_path: str) -> str | None:
    try:
        with open(mlproject_path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
    except Exception:
        return None

    entry_points = obj.get("entry_points") or {}
    for _, ep in entry_points.items():
        cmd = ep.get("command") if isinstance(ep, dict) else None
        if not cmd or not isinstance(cmd, str):
            continue
        m = re.search(r"\bpython(?:3)?\s+([^\s]+\.py)\b", cmd)
        if m:
            return m.group(1)
    return None


def find_strings_ending_with_py(obj) -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            hits.extend(find_strings_ending_with_py(v))
    elif isinstance(obj, list):
        for v in obj:
            hits.extend(find_strings_ending_with_py(v))
    elif isinstance(obj, str):
        if obj.strip().endswith(".py"):
            hits.append(obj.strip())
    return hits


def parse_pipeline_for_script_and_data(pipeline_path: str) -> tuple[str | None, list[str]]:
    try:
        with open(pipeline_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, []

    py_candidates = find_strings_ending_with_py(obj)
    script = py_candidates[0] if py_candidates else None

    data_paths: set[str] = set()

    def normalize_path(p: str) -> str:
        return p.replace("\\", "/").lstrip("./")

    def looks_like_data_path(key_lower: str, val: str) -> bool:
        v = val.replace("\\", "/").strip()
        if not v:
            return False
        if v.startswith(("data/", "dataset/", "datasets/")):
            return True
        if any(v.endswith(ext) for ext in (".csv", ".parquet", ".json", ".txt")) and ("/" in v):
            return True
        if "data" in key_lower and ("/" in v or v.startswith(".")):
            return True
        return False

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = str(k).lower()
                if isinstance(v, str) and looks_like_data_path(lk, v):
                    data_paths.add(normalize_path(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            if looks_like_data_path("", o):
                data_paths.add(normalize_path(o))

    walk(obj)
    return script, sorted(data_paths)

def normalize_repo_rel_path(p: str) -> str:
    """
    Normalize a path so it can be compared with git-changed paths (repo-relative).
    Handles GoMLOps patterns like '../<repo>/src/train.py' by stripping leading '../<something>/'.
    """
    p = (p or "").replace("\\", "/").strip()
    p = p.lstrip("./")

    # If GoMLOps emits '../<repo>/path', strip the first two segments: '..' + '<repo>'
    if p.startswith("../"):
        # ../<repo>/src/train.py -> src/train.py
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 3 and parts[0] == "..":
            p = "/".join(parts[2:])

    return p


def classify_cause(changed_files: list[str], script_path: str | None, data_paths: list[str]) -> tuple[str, dict]:
    changed = [normalize_repo_rel_path(f) for f in changed_files]
    script_hit = False
    data_hit = False

    script_prefixes: list[str] = []
    if script_path:
        sp = normalize_repo_rel_path(script_path)
        script_prefixes.append(sp)
        if "/" in sp:
            script_prefixes.append(sp.rsplit("/", 1)[0] + "/")

    # normalize data paths once
    norm_data_paths = [normalize_repo_rel_path(dp).rstrip("/") for dp in (data_paths or []) if dp]

    for f in changed:
        nf = normalize_repo_rel_path(f)

        if any(nf == p or nf.startswith(p) for p in script_prefixes):
            script_hit = True

        if any(nf == dp or nf.startswith(dp + "/") for dp in norm_data_paths if dp):
            data_hit = True


    if script_hit and data_hit:
        cause = "Both"
    elif script_hit:
        cause = "Script"
    elif data_hit:
        cause = "Data"
    else:
        cause = ""

    dbg = {
        "changed_files_count": str(len(changed)),
        "script_path": script_path or "",
        "data_paths": ";".join(data_paths),
        "script_hit": str(script_hit),
        "data_hit": str(data_hit),
    }
    return cause, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caller-root", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    caller_root = args.caller_root
    sha = args.sha

    mlproject = os.path.join(caller_root, "MLproject")
    arg2pipeline_dir = os.path.join(caller_root, "arg2pipeline")
    pipeline_json = os.path.join(arg2pipeline_dir, "pipeline.json")

    mlflow_project_detected = path_exists(mlproject) and os.path.isdir(arg2pipeline_dir)

    missing_reason = ""
    if not path_exists(mlproject):
        missing_reason = "MLproject file not found at repo root."
    elif not os.path.isdir(arg2pipeline_dir):
        missing_reason = "arg2pipeline/ directory not found."
    elif not path_exists(pipeline_json):
        missing_reason = "arg2pipeline/ found, but pipeline.json is missing (limited parsing)."

    changed_files = get_changed_files_single_commit(caller_root, sha) if sha else []

    cause = ""
    dbg = {
        "changed_files_count": str(len(changed_files)),
        "script_path": "",
        "data_paths": "",
        "script_hit": "",
        "data_hit": "",
    }

    if mlflow_project_detected:
        script_from_mlproject = parse_mlproject_for_script(mlproject)

        script_from_pipeline, data_paths = (None, [])
        if path_exists(pipeline_json):
            script_from_pipeline, data_paths = parse_pipeline_for_script_and_data(pipeline_json)

        script_path = script_from_mlproject or script_from_pipeline
        cause, dbg = classify_cause(changed_files, script_path, data_paths)

    payload = {
        "mlflow_project_detected": mlflow_project_detected,
        "missing_reason": missing_reason,
        "changed_files": changed_files,  # kept for debugging / future use
        "cause": cause,
        "dbg": dbg,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

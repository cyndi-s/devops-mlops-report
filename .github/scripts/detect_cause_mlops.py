import argparse
import json
import os
import re
import subprocess

import yaml

def _is_ignored_path(p: str) -> bool:
    """
    Tool-level default ignore for filesystem noise.
    These files should never trigger ML retraining.
    """
    name = p.rsplit("/", 1)[-1]
    return name in {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    } or "/__MACOSX/" in p

def path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def run_git(cwd: str, args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore").strip()


def get_changed_files_single_commit(repo_dir: str, sha: str) -> list[str]:
     if not sha:
        return []
     try:
         txt = run_git(repo_dir, ["show", "--name-only", "--pretty=format:", sha])
         files = [f.strip() for f in txt.splitlines() if f.strip()]
         return files
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

def normalize_repo_rel_path(p: str, repo_hints: set[str] | None = None) -> str:
    """
    Normalize to repo-relative paths (like git changed files).
    Handles:
      - ../<repo>/path  -> path
      - <repo>/path     -> path   (GoMLOps sometimes emits this)
    """
    p = (p or "").replace("\\", "/").strip()
    p = p.lstrip("./")

    # ../<repo>/src/train.py -> src/train.py
    if p.startswith("../"):
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 3 and parts[0] == "..":
            p = "/".join(parts[2:])

    # <repo>/src/train.py -> src/train.py (only if <repo> matches known hints)
    if repo_hints:
        for rh in repo_hints:
            rh = (rh or "").strip().strip("/")
            if rh and p.startswith(rh + "/"):
                p = p[len(rh) + 1 :]
                break

    return p


def classify_cause(changed_files: list[str], script_paths: list[str], data_paths: list[str]) -> tuple[str, dict]:
    changed = [normalize_repo_rel_path(f, REPO_HINTS) for f in changed_files]
    script_hit = False
    data_hit = False

    # Script triggers: exact file match only (no directory semantics)
    script_targets = [
        normalize_repo_rel_path(sp, REPO_HINTS).rstrip("/")
        for sp in (script_paths or [])
        if sp
    ]

    # Data triggers: file OR directory semantics
    norm_data_paths = [
        normalize_repo_rel_path(dp, REPO_HINTS).rstrip("/") 
        for dp in (data_paths or []) 
        if dp]

    for f in changed:
        if any(f == p for p in script_targets):
            script_hit = True

        if any(f == dp or f.startswith(dp + "/") for dp in norm_data_paths if dp):
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
        "script_paths": ";".join(script_targets),
        "data_paths": ";".join(data_paths or []),
        "script_hit": str(script_hit),
        "data_hit": str(data_hit),
    }
    return cause, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to report-config.yml in caller repo")
    ap.add_argument("--caller-root", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    caller_root = args.caller_root
    sha = args.sha


    # v2: We may use a repo MLproject, or generate a temporary one later from user config.
    mlproject = os.path.join(caller_root, "MLproject")
    repo_mlproject_exists = path_exists(mlproject)

    # v2: keep report clean (no warnings)
    missing_reason = ""


    changed_files = get_changed_files_single_commit(caller_root, sha) if sha else []
    
    cause = ""
    dbg = {
        "changed_files_count": str(len(changed_files)),
        "script_path": "",
        "data_paths": "",
        "script_hit": "",
        "data_hit": "",
        "workdir": workdir,
    }

    global REPO_HINTS
    REPO_HINTS = set()

    # hint from GitHub context (caller repo name)
    repo_full = os.getenv("GITHUB_REPOSITORY", "")  # like "cyndi-s/mlops-ci-demo"
    if repo_full and "/" in repo_full:
        REPO_HINTS.add(repo_full.split("/")[-1])

   
    # v2: user-defined paths are primary
    cfg = {}
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    project = cfg.get("project") or {}
    workdir = str(project.get("workdir") or "").strip()
     # train_script can be str or list[str]
    ts = project.get("train_script") or []
    if isinstance(ts, str):
        user_scripts = [ts]
    elif isinstance(ts, list):
        user_scripts = ts
    else:
        user_scripts = []
    user_scripts = [str(x).strip() for x in user_scripts if str(x).strip()]

    # convention: first train_script is the execution entry
    user_entry_script = user_scripts[0] if user_scripts else ""

    user_data = project.get("data_paths") or []
    if isinstance(user_data, str):
        user_data = [user_data]
    user_data = [str(x).strip() for x in user_data if str(x).strip()]

    # v2: user-defined paths are primary; otherwise best-effort parse repo MLproject (if present)
        # v2: user-defined scripts are primary; otherwise best-effort parse repo MLproject (if present)
    parsed_script = parse_mlproject_for_script(mlproject) if repo_mlproject_exists else None

    script_paths = user_scripts if user_scripts else ([parsed_script] if parsed_script else [])
    entry_script = user_entry_script or (parsed_script or "")

    data_paths = user_data

    # v2: minimal validation (existence only) — validate entry script only
    config_valid = True
    invalid_reason = ""

    if entry_script:
        if not path_exists(os.path.join(caller_root, entry_script)):
            config_valid = False
            invalid_reason = f"train_script not found: {entry_script}"

    # data paths are not strictly required for cause detection; warn-only behavior is handled elsewhere
    # (we don't invalidate training here just because a data path is missing)

    # v2: treat MLproject as effectively available if repo has it OR user provided a script path
    # (tmp MLproject will be created later if needed)
    mlflow_project_detected = bool(repo_mlproject_exists or (entry_script and config_valid))

    if config_valid:
        # Filter out tool-level ignored files (filesystem noise)
        filtered_changed_files = [
            f for f in changed_files
            if not _is_ignored_path(f)
        ]
        cause, dbg = classify_cause(filtered_changed_files, script_paths, data_paths)
    else:
        cause = ""
        dbg["script_paths"] = ";".join(user_scripts)
        dbg["data_paths"] = ";".join(user_data)
        dbg["script_hit"] = "False"
        dbg["data_hit"] = "False"
        dbg["invalid_reason"] = invalid_reason



    payload = {
        "mlflow_project_detected": mlflow_project_detected,
        "repo_mlproject_exists": repo_mlproject_exists,
        "config_valid": config_valid if "config_valid" in locals() else True,
        "invalid_reason": invalid_reason if "invalid_reason" in locals() else "",
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

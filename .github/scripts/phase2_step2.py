import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import subprocess
from io import StringIO
from urllib import request, error
from zoneinfo import ZoneInfo

import yaml

CSV_NAME = "commitHistory.csv"
FIXED_MLFLOW_MISSING_MSG_MD = """
**MLflow project not detected**

No `MLproject` file was found in this repository.

The `devops-mlops-report` expects an MLflow Project with MLflow runs
(metrics and params logged to an MLflow tracking server).

You can generate an MLproject file using the [GoMLOps] (https://github.com/yorku-ease/GoMLOps) tool.

After adding `MLproject` file and `arg2pipeline/` folder, re-run this workflow.
""".strip()


# ---------- GitHub Gist helpers ----------
def extract_gist_id(gist_url: str) -> str:
    m = re.search(r"gist\.github\.com/(?:[^/]+/)?([a-f0-9]+)", gist_url.strip())
    if not m:
        raise ValueError(f"Invalid gist_url format: {gist_url}")
    return m.group(1)


def gh_api_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "devops-mlops-report",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API error {e.code} {e.reason}: {detail}") from e


def get_gist_file_content(gist_id: str, token: str, filename: str) -> str | None:
    gist = gh_api_request("GET", f"https://api.github.com/gists/{gist_id}", token)
    files = gist.get("files", {})
    if filename not in files:
        return None
    return files[filename].get("content")


def update_gist_file(gist_id: str, token: str, filename: str, content: str) -> None:
    payload = {"files": {filename: {"content": content}}}
    gh_api_request("PATCH", f"https://api.github.com/gists/{gist_id}", token, payload)


def append_csv_row(existing_csv: str | None, row: dict, fieldnames: list[str]) -> str:
    """
    Stable behavior:
    - Always rewrites header to current 'fieldnames'
    - Preserves any existing rows by mapping known keys; fills missing with ""
    """
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    if existing_csv and existing_csv.strip():
        reader = csv.DictReader(StringIO(existing_csv))
        for r in reader:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()


# ---------- MLflow Project detection + cause attribution ----------
def path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def run_git(cwd: str, args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore").strip()


def get_changed_files_single_commit(repo_dir: str, sha: str) -> list[str]:
    # Works with fetch-depth=1 as long as this commit exists locally.
    # Lists files changed in THIS commit.
    try:
        txt = run_git(repo_dir, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
        files = [f.strip() for f in txt.splitlines() if f.strip()]
        return files
    except Exception:
        return []


def parse_mlproject_for_script(mlproject_path: str) -> str | None:
    """
    Heuristic: find 'python <something>.py' in entry point command.
    MLproject is YAML.
    """
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
    hits = []
    if isinstance(obj, dict):
        for v in obj.values():
            hits.extend(find_strings_ending_with_py(v))
    elif isinstance(obj, list):
        for v in obj:
            hits.extend(find_strings_ending_with_py(v))
    elif isinstance(obj, str):
        if obj.endswith(".py") and ("/" in obj or "\\" in obj or obj.count(".") >= 1):
            hits.append(obj)
    return hits


def parse_pipeline_for_script_and_data(pipeline_path: str) -> tuple[str | None, list[str]]:
    """
    Heuristics:
    - script: first string ending with .py found anywhere
    - data paths: any string value that looks like data/... or dataset/... or ends with .csv/.parquet/.json
      OR dict keys containing 'data' and values that are paths.
    """
    try:
        with open(pipeline_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, []

    py_candidates = find_strings_ending_with_py(obj)
    script = py_candidates[0] if py_candidates else None

    data_paths: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = str(k).lower()
                if isinstance(v, str):
                    sv = v.strip()
                    if looks_like_data_path(lk, sv):
                        data_paths.add(normalize_path(sv))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            sv = o.strip()
            if looks_like_data_path("", sv):
                data_paths.add(normalize_path(sv))

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

    walk(obj)
    return script, sorted(data_paths)


def classify_cause(changed_files: list[str], script_path: str | None, data_paths: list[str]) -> tuple[str, dict]:
    """
    Returns: (cause, debug_info)
      cause in {"Data","Script","Both",""}  (empty => no attribution)
    """
    changed = [f.replace("\\", "/").lstrip("./") for f in changed_files]
    script_hit = False
    data_hit = False

    script_prefixes = []
    if script_path:
        sp = script_path.replace("\\", "/").lstrip("./")
        script_prefixes.append(sp)
        # also treat script directory as script-scope
        if "/" in sp:
            script_prefixes.append(sp.rsplit("/", 1)[0] + "/")

    for f in changed:
        if any(f == p or f.startswith(p) for p in script_prefixes):
            script_hit = True
        if any(f == dp or f.startswith(dp.rstrip("/") + "/") or f.startswith(dp.rstrip("/") + "/") for dp in data_paths):
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
        "changed_files_count": len(changed),
        "script_path": script_path or "",
        "data_paths": ";".join(data_paths),
        "script_hit": str(script_hit),
        "data_hit": str(data_hit),
    }
    return cause, dbg


# ---------- Summary ----------
def write_summary(
    gist_url: str,
    mlflow_project_detected: bool,
    missing_reason: str,
    fixed_mlflow_missing_msg: str,
    cause: str,
    branch: str,
    author: str,
    commit_url: str,
    commit_sha: str,
    commit_msg: str,
    model_source_in_apk: str,
    workflow_status: str,
    finished_at: str,
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("# devops-mlops-report\n\n")

        # -------- Section 1 --------
        f.write("## Section 1 — Latest Model\n\n")
        if not mlflow_project_detected:
            # Your REQUIRED fixed message (2.2)
            f.write(fixed_mlflow_missing_msg.strip() + "\n\n")
            f.write("_Latest model is unchanged because MLOps processing is disabled._\n\n")
        else:
            f.write("Placeholder (Step 2: MLflow run extraction not implemented yet)\n\n")
            f.write(f"- Cause attribution (preview): **{cause or 'N/A'}**\n\n")

        # -------- Section 2 --------
        f.write("## Section 2 — Model Performance Trend\n\n")
        f.write("Placeholder (Step 2: trend SVG not implemented yet)\n\n")

        # -------- Section 3 (MUST be DevOps context) --------
        f.write("## Section 3 — Code\n\n")
        f.write(f"- Branch: `{branch}`\n")
        f.write(f"- Author: `{author}`\n")
        if commit_url:
            f.write(f"- Commit: [{commit_sha[:7]}]({commit_url}) — {commit_msg}\n")
        else:
            f.write(f"- Commit: {commit_sha[:7]} — {commit_msg}\n")
        f.write(f"- Model source in APK: {model_source_in_apk}\n")
        f.write(f"- Status: {workflow_status}\n")
        f.write(f"- Job finished at: {finished_at}\n\n")

        # -------- Section 4 --------
        f.write("## Section 4 — Artifacts\n\n")
        f.write("Placeholder (Step 2)\n\n")

        # -------- Section 5 --------
        f.write("## Section 5 — Commit History\n\n")
        f.write(f"- **Commit History:** [commitHistory.csv]({gist_url})\n\n")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to report-config.yml in caller repo")
    ap.add_argument("--caller-root", default="../caller", help="Caller repo checkout path (relative to app/)")
    args = ap.parse_args()
    

    token = os.environ.get("GIST_TOKEN")
    if not token:
        print("Missing env GIST_TOKEN", file=sys.stderr)
        sys.exit(2)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gist_url = cfg["storage"]["gist_url"]
    gist_id = extract_gist_id(gist_url)

    caller_root = os.path.abspath(os.path.join(os.getcwd(), args.caller_root))
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
        # optional extra detail (you can keep or remove)
        missing_reason = "arg2pipeline/ found, but pipeline.json is missing (limited parsing)."


    sha = os.environ.get("GITHUB_SHA", "")
    branch = os.environ.get("GITHUB_REF_NAME", "")
    actor = os.environ.get("GITHUB_ACTOR", "")

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    commit_url = f"{server_url}/{repo}/commit/{sha}" if repo and sha else ""

    commit_msg = ""
    try:
        commit_msg = run_git(caller_root, ["log", "-1", "--pretty=%s", sha]) if sha else ""
    except Exception:
        pass

    job_status = os.environ.get("WORKFLOW_STATUS", "success")  # pass from YAML
    finished_at = dt.datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S %Z")


    changed_files = get_changed_files_single_commit(caller_root, sha) if sha else []

    script_path = None
    data_paths: list[str] = []
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

        # Prefer MLproject script if found; else pipeline script.
        script_path = script_from_mlproject or script_from_pipeline

        cause, dbg = classify_cause(changed_files, script_path, data_paths)


    # Step 2 still DevOps-only row (no MLflow run association yet)
    is_trained = "No"

    now = dt.datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S")

    fieldnames = [
        "timestamp_toronto",
        "branch",
        "author",
        "commit_sha",
        "commit_message",
        "workflow_status",
        "mlflow_project_detected",
        "is_trained",
        "cause",
        "changed_files_count",
        "script_path",
        "data_paths",
        "script_hit",
        "data_hit",
    ]

    row = {
        "timestamp_toronto": now,
        "branch": branch,
        "author": actor,
        "commit_sha": sha,
        "commit_message": "",  # keep empty for now
        "workflow_status": "success",
        "mlflow_project_detected": "Yes" if mlflow_project_detected else "No",
        "is_trained": is_trained,
        "cause": cause,
        **dbg,
    }

    existing = get_gist_file_content(gist_id, token, CSV_NAME)
    updated = append_csv_row(existing, row, fieldnames)
    update_gist_file(gist_id, token, CSV_NAME, updated)

    write_summary(
        gist_url=gist_url,
        mlflow_project_detected=mlflow_project_detected,
        missing_reason=missing_reason,
        fixed_mlflow_missing_msg=FIXED_MLFLOW_MISSING_MSG_MD,
        cause=cause,
        branch=branch,       
        author=actor,
        commit_url=commit_url,
        commit_sha=sha,
        commit_msg=commit_msg,
        model_source_in_apk="Unknown (not implemented yet)",
        workflow_status=job_status,
        finished_at=finished_at,
    )


    print(f"Updated {CSV_NAME} in gist: {gist_url}")


if __name__ == "__main__":
    main()

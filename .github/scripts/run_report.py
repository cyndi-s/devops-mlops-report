import argparse
import datetime as dt
import json
import os
import subprocess
import time
from zoneinfo import ZoneInfo
from urllib import request

import yaml

def get_tz(cfg: dict) -> ZoneInfo | None:
    name = str(cfg.get("timezone") or "").strip()
    if not name:
        return None
    if name.upper() == "UTC":
        name = "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def sh(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def git_log_subject(repo_dir: str, sha: str) -> str:
    if not sha:
        return ""
    try:
        out = subprocess.check_output(["git", "log", "-1", "--pretty=%s", sha], cwd=repo_dir)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def list_run_artifacts(repo: str, run_id: str, token: str) -> dict:
    """
    Observer-only: lists GitHub Actions artifacts uploaded for the *current run*.
    Returns:
      {
        "run_url": "https://github.com/<repo>/actions/runs/<run_id>",
        "items": [{"name": "...", "size_in_bytes": 123, "expired": false}, ...]
      }
    """
    if not (repo and run_id and token):
        return {"run_url": "", "items": []}

    api_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "devops-mlops-report",
    }

    req = request.Request(api_url, headers=headers, method="GET")
    with request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8") or "{}")

    items = []
    for a in (data.get("artifacts") or []):
        items.append({
            "name": (a.get("name") or "").strip(),
            "size_in_bytes": int(a.get("size_in_bytes") or 0),
            "expired": bool(a.get("expired")),
        })

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server_url}/{repo}/actions/runs/{run_id}"
    return {"run_url": run_url, "items": items}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to report-config.yml in caller repo")
    ap.add_argument("--caller-root", default="../caller", help="Caller repo checkout path (relative to app/)")
    args = ap.parse_args()

    # Load report config (from caller checkout)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gist_url = cfg["storage"]["gist_url"]

    # Context
    caller_root = os.path.abspath(os.path.join(os.getcwd(), args.caller_root))
    sha    = os.environ.get("CALLER_SHA", "") or os.environ.get("GITHUB_SHA", "")
    branch = os.environ.get("CALLER_REF_NAME", "") or os.environ.get("GITHUB_REF_NAME", "")
    actor  = os.environ.get("CALLER_ACTOR", "") or os.environ.get("GITHUB_ACTOR", "")

    repo = os.environ.get("CALLER_REPO", "") or os.environ.get("GITHUB_REPOSITORY", "")
    server_url = os.environ.get("CALLER_SERVER_URL", "") or os.environ.get("GITHUB_SERVER_URL", "https://github.com")

    commit_url = f"{server_url}/{repo}/commit/{sha}" if repo and sha else ""
    commit_msg = git_log_subject(caller_root, sha)

    status = os.environ.get("WORKFLOW_STATUS", "success")
    tz = get_tz(cfg)
    now_dt = dt.datetime.now(tz) if tz else dt.datetime.now().astimezone()
    finished_at = now_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    # for Section 3 (POC-Mobile style)
    job_end_epoch = int(now_dt.timestamp())
    os.environ["JOB_END"] = str(job_end_epoch)

    # Work dir for step-to-step files
    workdir = os.path.abspath(".github/scripts/.work")
    os.makedirs(workdir, exist_ok=True)

    detect_json = os.path.join(workdir, "detect.json")
    mlflow_json = os.path.join(workdir, "mlflow.json")
    devops_json = os.path.join(workdir, "devops.json")
    row_json = os.path.join(workdir, "row.json")
    train_json = os.path.join(workdir, "train.json")
    ml_details_json = os.path.join(workdir, "ml_details.json")
    svg_json = os.path.join(workdir, "svg.json")
    model_json = os.path.join(workdir, "model.json")
    artifacts_json = os.path.join(workdir, "artifacts.json")

    # 1) Detect MLflow project + cause attribution (tested)
    sh([
    "python", ".github/scripts/detect_cause_mlops.py",
    "--caller-root", caller_root,
    "--sha", sha,
    "--out", detect_json
    ])

    with open(detect_json, "r", encoding="utf-8") as f:
        det = json.load(f)
        
    # Decide whether to trigger training (toggle)
    cause = (det.get("cause") or "").strip()
    mlflow_project_detected = bool(det.get("mlflow_project_detected"))

    # 1.5) Trigger training (Phase 2.3) and capture run_id (NO commit-tag query)
    t0 = time.time()
    sh([
        "python", ".github/scripts/trigger_training.py",
        "--config", args.config,
        "--caller-root", caller_root,
        "--cause", (cause if mlflow_project_detected else ""),
        "--out", train_json,
    ])

    with open(train_json, "r", encoding="utf-8") as f:
        tr = json.load(f)

    run_id = (tr.get("run_id") or "").strip()
    trained = bool(tr.get("trained"))  # NEW
    is_trained = "true" if trained else "false"
    
    # Ground truth: training happened in THIS workflow run (even if run_id missing)
    os.environ["TRAINED_THIS_RUN"] = "true" if trained else "false"

    
    # 2) Prepare DevOps payload for summary (tested format)
    devops_payload = {
        "branch": branch,
        "author": actor,
        "commit_sha": sha,
        "commit_msg": commit_msg,
        "commit_url": commit_url,
        "model_source_in_apk": "Unknown (not implemented yet)",
        "status": status,
        "finished_at": finished_at,
    }
    with open(devops_json, "w", encoding="utf-8") as f:
        json.dump(devops_payload, f, indent=2)

    # 3) Phase 2.2 row -> commitHistory.csv (tested gist append)
    now = (dt.datetime.now(tz) if tz else dt.datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M:%S")

    fieldnames = [
        "timestamp_local",
        "branch",
        "author",
        "commit_sha",
        "commit_message",
        "workflow_status",
        "mlflow_project_detected",
        "is_trained",
        "mlflow_run_id",
        "duration",
        "mlflow_params_kv",
        "mlflow_metrics_kv",
        "cause",
    ]

    # Phase 2.4: Extract metrics/params only when trained (key=value strings)
    params_kv = ""
    metrics_kv = ""
    duration = ""

    if is_trained == "true":
        sh([
            "python", ".github/scripts/extract_mlflow_details.py",
            "--config", args.config,
            "--run-id", run_id,
            "--out", ml_details_json,
        ])
        with open(ml_details_json, "r", encoding="utf-8") as f:
            md = json.load(f)

        params_kv = (md.get("params_kv") or "").strip()
        metrics_kv = (md.get("metrics_kv") or "").strip()
        duration = (md.get("duration") or "").strip()

    # Keep a small mlflow_json for summary compatibility (generate_summary_md.py expects it)
    ml = {
        "is_trained": is_trained,
        "run_id": run_id,
        "reason": tr.get("reason", ""),
        "duration": (duration if trained else ""),
    }
    with open(mlflow_json, "w", encoding="utf-8") as f:
        json.dump(ml, f, indent=2)


    row = {
        "timestamp_local": now,
        "branch": branch,
        "author": actor,
        "commit_sha": sha,
        "commit_message": commit_msg,
        "workflow_status": status,
        "mlflow_project_detected": "Yes" if mlflow_project_detected else "No",
        "is_trained": is_trained,
        "mlflow_run_id": run_id,
        "duration": (duration if run_id else ""),
        "mlflow_params_kv": params_kv,
        "mlflow_metrics_kv": metrics_kv,
        "cause": cause,
    }

    with open(row_json, "w", encoding="utf-8") as f:
        json.dump({"fieldnames": fieldnames, "row": row}, f, indent=2)

    sh(["python", ".github/scripts/append_commitHistory.py",
        "--gist-url", gist_url,
        "--row-json", row_json])

   # 4) Render trend SVG (CSV -> SVG -> gist)
    sh(["python", ".github/scripts/render_svg.py",
        "--config", args.config,
        "--gist-url", gist_url,
        "--out", svg_json])

    # 5) Fetch registry model version (best-effort)
    sh(["python", ".github/scripts/fetch_registry_model.py",
        "--config", args.config,
        "--gist-url", gist_url,
        "--out", model_json])

    # 5.5) List GitHub Actions artifacts for THIS run (metadata only)
    run_id_env = os.environ.get("GITHUB_RUN_ID", "").strip()
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()

    artifacts_payload = list_run_artifacts(repo, run_id_env, gh_token)
    with open(artifacts_json, "w", encoding="utf-8") as f:
        json.dump(artifacts_payload, f, indent=2)

    # 6) Summary markdown (CSV-only for Sections 1 & 2)
    sh(["python", ".github/scripts/generate_summary_md.py",
        "--config", args.config,
        "--gist-url", gist_url,
        "--svg-json", svg_json,
        "--model-json", model_json,
        "--devops-json", devops_json,
        "--artifacts-json", artifacts_json])

    # 7) Optional prune (fixed policy: private -> max 90)
    repo_visibility = str(cfg.get("repo_visibility") or "").strip().lower()
    if repo_visibility == "private":
        sh(["python", ".github/scripts/prune_mlflow_runs.py",
            "--config", args.config])


if __name__ == "__main__":
    main()

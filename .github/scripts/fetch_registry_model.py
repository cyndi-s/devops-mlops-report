import argparse
import csv
import json
import os
import re
from io import StringIO
from typing import Dict, List, Optional
from urllib import request, error

import yaml


CSV_NAME_DEFAULT = "commitHistory.csv"


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def is_true(v: str) -> bool:
    return (v or "").strip().lower() in ("yes", "true", "1")


def try_import_mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
        return mlflow, MlflowClient
    except Exception:
        return None, None


def best_model_version_for_run(client, run_id: str) -> str:
    # returns highest numeric version among matches
    try:
        mvs = client.search_model_versions(f"run_id = '{run_id}'")
    except Exception:
        return ""

    best: Optional[int] = None
    for mv in mvs:
        try:
            v = int(getattr(mv, "version", "") or "")
            best = v if best is None else max(best, v)
        except Exception:
            continue
    return str(best) if best is not None else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="caller report-config.yml path")
    ap.add_argument("--gist-url", required=True, help="gist url")
    ap.add_argument("--csv-name", default=CSV_NAME_DEFAULT)
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    token = (os.environ.get("GIST_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing env GIST_TOKEN")

    cfg = load_cfg(args.config)
    tracking_uri = ((cfg.get("mlflow") or {}).get("tracking_uri") or "").strip()

    gist_id = extract_gist_id(args.gist_url)
    csv_text = get_gist_file_content(gist_id, token, args.csv_name) or ""

    rows: List[Dict[str, str]] = []
    if csv_text.strip():
        rdr = csv.DictReader(StringIO(csv_text))
        for r in rdr:
            rows.append({k: (v or "").strip() for k, v in r.items()})

    # find latest trained run_id
    rows.sort(key=lambda r: r.get("timestamp_local", ""))
    run_id = ""
    for r in reversed(rows):
        if is_true(r.get("is_trained", "")):
            run_id = (r.get("mlflow_run_id") or "").strip()
            if run_id:
                break

    model_version = ""

    mlflow, MlflowClient = try_import_mlflow()
    if mlflow and MlflowClient and tracking_uri and run_id:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            model_version = best_model_version_for_run(client, run_id)
        except Exception:
            model_version = ""

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "model_version": model_version,
                "tracking_uri_set": bool(tracking_uri),
            },
            f,
            indent=2,
        )

    print(f"Registry lookup: run_id={run_id or 'NA'} model_version={model_version or 'NA'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

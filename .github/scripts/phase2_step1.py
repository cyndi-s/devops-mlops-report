import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from urllib import request, error
from zoneinfo import ZoneInfo

import yaml


CSV_NAME = "commitHistory.csv"


def extract_gist_id(gist_url: str) -> str:
    # Supports:
    # - https://gist.github.com/user/<id>
    # - https://gist.github.com/<id>
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
    file_obj = files[filename]
    # GitHub includes 'content' for small files; use it directly.
    return file_obj.get("content")


def update_gist_file(gist_id: str, token: str, filename: str, content: str) -> None:
    payload = {"files": {filename: {"content": content}}}
    gh_api_request("PATCH", f"https://api.github.com/gists/{gist_id}", token, payload)


def append_csv_row(existing_csv: str | None, row: dict, fieldnames: list[str]) -> str:
    # Ensure header exists
    output_lines: list[str] = []
    if existing_csv and existing_csv.strip():
        output_lines = existing_csv.splitlines()
        # If header mismatches, we still append using our defined fieldnames
        # (Phase 2 Step 1 keeps schema minimal and stable)
        existing = "\n".join(output_lines).strip("\n")
        existing_rows = existing.splitlines()
        # If file exists but has no header, overwrite with our header + existing content
        if not existing_rows or "," not in existing_rows[0]:
            existing_csv = None

    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    if existing_csv and existing_csv.strip():
        # Re-read existing, but only keep rows that match our schema keys.
        reader = csv.DictReader(StringIO(existing_csv))
        for r in reader:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()


def write_summary(gist_url: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    sections = [
        ("Section 1 — Latest Model", "Placeholder (Step 1: DevOps skeleton only)"),
        ("Section 2 — Model Performance Trend", "Placeholder (Step 1: DevOps skeleton only)"),
        ("Section 3 — Code (DevOps)", "Placeholder (Step 1: DevOps skeleton only)"),
        ("Section 4 — Artifacts", "Placeholder (Step 1: DevOps skeleton only)"),
        ("Section 5 — Commit History", f"Gist link: {gist_url}"),
    ]

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("# devops-mlops-report\n\n")
        for title, body in sections:
            f.write(f"## {title}\n\n{body}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to report-config.yml in caller repo")
    args = ap.parse_args()

    token = os.environ.get("GIST_TOKEN")
    if not token:
        print("Missing env GIST_TOKEN", file=sys.stderr)
        sys.exit(2)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gist_url = cfg["storage"]["gist_url"]
    gist_id = extract_gist_id(gist_url)

    # Minimal schema for Step 1 (DevOps-only record)
    fieldnames = [
        "timestamp_toronto",
        "branch",
        "author",
        "commit_sha",
        "commit_message",
        "workflow_status",
        "is_trained",
        "cause",
    ]

    # Collect run context from GitHub Actions env
    branch = os.environ.get("GITHUB_REF_NAME", "")
    sha = os.environ.get("GITHUB_SHA", "")
    actor = os.environ.get("GITHUB_ACTOR", "")
    # Commit message isn't always available without git log; keep empty for Step 1
    commit_message = ""

    # In Step 1 we don’t detect training yet
    is_trained = "No"
    cause = ""

    now = dt.datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp_toronto": now,
        "branch": branch,
        "author": actor,
        "commit_sha": sha,
        "commit_message": commit_message,
        "workflow_status": "success",  # Step 1 assumes this step runs to completion
        "is_trained": is_trained,
        "cause": cause,
    }

    existing = get_gist_file_content(gist_id, token, CSV_NAME)
    updated = append_csv_row(existing, row, fieldnames)
    update_gist_file(gist_id, token, CSV_NAME, updated)

    write_summary(gist_url)
    print(f"Updated {CSV_NAME} in gist: {gist_url}")


if __name__ == "__main__":
    main()

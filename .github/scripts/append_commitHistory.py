import argparse
import csv
import json
import os
import re
from io import StringIO
from urllib import request, error

CSV_NAME = "commitHistory.csv"


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
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    if existing_csv and existing_csv.strip():
        reader = csv.DictReader(StringIO(existing_csv))
        for r in reader:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gist-url", required=True)
    ap.add_argument("--row-json", required=True)
    args = ap.parse_args()

    token = os.environ.get("GIST_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing env GIST_TOKEN")

    gist_id = extract_gist_id(args.gist_url)

    with open(args.row_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fieldnames = payload["fieldnames"]
    row = payload["row"]

    existing = get_gist_file_content(gist_id, token, CSV_NAME)
    updated = append_csv_row(existing, row, fieldnames)
    update_gist_file(gist_id, token, CSV_NAME, updated)

    print(f"Updated {CSV_NAME} in gist: {args.gist_url}")


if __name__ == "__main__":
    main()

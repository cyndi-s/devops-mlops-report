import argparse
import csv
import html
import json
import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional
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


def parse_kv_string(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def is_true(v: str) -> bool:
    return (v or "").strip().lower() in ("yes", "true", "1")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "NA"):
            return None
        return float(x)
    except Exception:
        return None


def fmt_val(x: Any) -> str:
    try:
        fx = float(x)
        return f"{fx:.3f}" if abs(fx) >= 1e-3 or fx == 0 else f"{fx:.1e}"
    except Exception:
        return "" if x in (None, "") else str(x)


def arrow_delta(prev: Any, cur: Any) -> str:
    pa = safe_float(prev)
    ca = safe_float(cur)
    if pa is None or ca is None:
        return ""
    d = ca - pa
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
    mag = abs(d)
    mag_s = f"{mag:.3f}" if mag >= 1e-3 or mag == 0 else f"{mag:.1e}"
    return f"{arrow} {mag_s}"


def write_summary(md: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gist-url", required=True)
    ap.add_argument("--csv-name", default=CSV_NAME_DEFAULT)
    ap.add_argument("--svg-json", required=True)
    ap.add_argument("--model-json", required=True)
    args = ap.parse_args()

    token = (os.environ.get("GIST_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing env GIST_TOKEN")

    cfg = load_cfg(args.config)
    report_cfg = (cfg.get("report") or {})
    metric_key = str(report_cfg.get("highlight_metric") or "val_accuracy").strip()
    sharp_delta = float(report_cfg.get("sharp_delta") or 0.15)
    trend_window = int(report_cfg.get("trend_window") or 10)

    gist_id = extract_gist_id(args.gist_url)
    csv_text = get_gist_file_content(gist_id, token, args.csv_name) or ""

    rows: List[Dict[str, str]] = []
    if csv_text.strip():
        rdr = csv.DictReader(StringIO(csv_text))
        for r in rdr:
            rows.append({k: (v or "").strip() for k, v in r.items()})

    rows.sort(key=lambda r: r.get("timestamp_toronto", ""))

    trained_rows = [r for r in rows if is_true(r.get("is_trained", ""))]
    latest_model = trained_rows[-1] if trained_rows else None
    prev_model = trained_rows[-2] if len(trained_rows) >= 2 else None

    # determine if THIS workflow trained (by matching commit sha to last row)
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    trained_this_run = False
    for r in reversed(rows):
        if (r.get("commit_sha") or "").strip() == sha:
            trained_this_run = is_true(r.get("is_trained", ""))
            break

    # read svg + model json
    with open(args.svg_json, "r", encoding="utf-8") as f:
        svgj = json.load(f) or {}
    svg_url = (svgj.get("svg_url") or "").strip()

    with open(args.model_json, "r", encoding="utf-8") as f:
        mj = json.load(f) or {}
    model_version = (mj.get("model_version") or "").strip()

    # helpers to read metrics
    def metric_from_row(r: Dict[str, str], key: str) -> Optional[float]:
        m = parse_kv_string(r.get("mlflow_metrics_kv", ""))
        return safe_float(m.get(key))

    def params_str(r: Dict[str, str]) -> str:
        return (r.get("mlflow_params_kv") or "").strip()

    # Section 1 fields
    status_text = "from this workflow run" if trained_this_run else "from a previous run"
    badge = f"<strong><ins><code>{html.escape(status_text)}</code></ins></strong>"

    s1_ts = (latest_model.get("timestamp_toronto") if latest_model else "") or ""
    s1_cause = (latest_model.get("cause") if latest_model else "") or ""
    s1_params = params_str(latest_model) if latest_model else ""

    s1_acc = metric_from_row(latest_model, "accuracy") if latest_model else None
    s1_vacc = metric_from_row(latest_model, metric_key) if latest_model else None
    s1_loss = metric_from_row(latest_model, "loss") if latest_model else None
    s1_vloss = metric_from_row(latest_model, "val_loss") if latest_model else None

    p_vacc = metric_from_row(prev_model, metric_key) if prev_model else None
    p_vloss = metric_from_row(prev_model, "val_loss") if prev_model else None

    # Section 2: last N trained rows, oldest->newest
    trend_base = trained_rows[-trend_window:] if trained_rows else []
    # build deltas sequentially using the selected metric
    pts = []
    prev_v: Optional[float] = None
    for r in trend_base:
        v = metric_from_row(r, metric_key)
        if v is None:
            continue
        d = 0.0 if prev_v is None else (v - prev_v)
        sharp = (abs(d) >= sharp_delta) if prev_v is not None else False
        prev_v = v
        pts.append({
            "ts": (r.get("timestamp_toronto") or ""),
            "branch": (r.get("branch") or ""),
            "author": (r.get("author") or ""),
            "cause": (r.get("cause") or "None"),
            "val": v,
            "delta": d,
            "sharp": sharp,
            "sha": (r.get("commit_sha") or ""),
        })

    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    server = (os.environ.get("GITHUB_SERVER_URL") or "https://github.com").strip()

    def commit_cell(sha_full: str) -> str:
        short = (sha_full or "")[:7]
        if repo and sha_full:
            url = f"{server}/{repo}/commit/{sha_full}"
            return f'<a href="{html.escape(url)}"><code>{html.escape(short)}</code></a>'
        return f"<code>{html.escape(short)}</code>"

    md: List[str] = []
    md.append("# Pipeline Summary\n\n")

    # ---- Section 1 ----
    md.append(f"## 1) Model in APK: {badge}\n\n")
    if not latest_model:
        md.append("_No trained model found in commitHistory.csv yet._\n\n")
    else:
        md.append("<table style='width:100%; text-align:center;'>")
        md.append("<thead><tr>"
                  "<th>Timestamp<br>(Toronto)</th>"
                  "<th>model_version</th>"
                  "<th>Cause</th>"
                  "<th>Parameters</th>"
                  "<th>accuracy</th>"
                  f"<th>{html.escape(metric_key)}</th>"
                  f"<th>Δ{html.escape(metric_key)}</th>"
                  "<th>loss</th>"
                  "<th>val_loss</th>"
                  "<th>Δval_loss</th>"
                  "</tr></thead>")
        md.append("<tbody><tr>")

        cells = [
            html.escape(s1_ts),
            html.escape(model_version or ""),
            f"<code>{html.escape(s1_cause)}</code>",
            f"<code>{html.escape(s1_params)}</code>",
            html.escape(fmt_val(s1_acc) if s1_acc is not None else ""),
            html.escape(fmt_val(s1_vacc) if s1_vacc is not None else ""),
            html.escape(arrow_delta(p_vacc, s1_vacc)),
            html.escape(fmt_val(s1_loss) if s1_loss is not None else ""),
            html.escape(fmt_val(s1_vloss) if s1_vloss is not None else ""),
            html.escape(arrow_delta(p_vloss, s1_vloss)),
        ]
        for c in cells:
            md.append(f"<td style='text-align:center; vertical-align:middle; word-break:break-word; max-width:100%'>{c}</td>")
        md.append("</tr></tbody></table>\n\n")

    # ---- Section 2 ----
    md.append(f"## 2) Model Performance ({html.escape(metric_key)})\n\n")
    if svg_url:
        md.append(f"<img alt='{html.escape(metric_key)} trend' src='{html.escape(svg_url)}' style='width:100%; height:auto; display:block;'/>\n\n")
    else:
        md.append("<em>SVG not available.</em>\n\n")

    md.append("<table style='width:100%; table-layout:auto; border-collapse:collapse;'>")
    md.append("<thead><tr>"
              "<th>Timestamp<br>(Toronto)</th>"
              "<th>Branch</th>"
              "<th>Author</th>"
              "<th>Cause</th>"
              f"<th>{html.escape(metric_key)}</th>"
              f"<th>Δ{html.escape(metric_key)}</th>"
              "<th>Commit</th>"
              "</tr></thead><tbody>")

    # Build table rows oldest -> newest; highlight sharp changes only
    for i, p in enumerate(pts):
        ts_cell = html.escape(p["ts"])
        br = f"<code>{html.escape(p['branch'])}</code>"
        au = f"<code>{html.escape(p['author'])}</code>"
        cause = f"<code>{html.escape(p['cause'] or 'None')}</code>"

        val_txt = fmt_val(p["val"])
        d_txt = "→ 0.000" if i == 0 else arrow_delta(p["val"] - p["delta"], p["val"])
        val_cell = f"<strong><ins>{html.escape(val_txt)}</ins></strong>" if p["sharp"] else html.escape(val_txt)

        md.append("<tr>")
        for c in [ts_cell, br, au, cause, val_cell, html.escape(d_txt), commit_cell(p["sha"])]:
            md.append(f"<td style='text-align:center; vertical-align:middle; word-break:break-word; max-width:100%'>{c}</td>")
        md.append("</tr>")

    md.append("</tbody></table>\n\n")

    # ---- Section 3/4/5 (keep minimal for now) ----
    md.append("## 3) Code\n\n")
    branch = (os.environ.get("GITHUB_REF_NAME") or "").strip()
    actor = (os.environ.get("GITHUB_ACTOR") or "").strip()
    md.append(f"- Branch: `{html.escape(branch)}`\n")
    md.append(f"- Author: `{html.escape(actor)}`\n")
    if sha and repo:
        commit_url = f"{server}/{repo}/commit/{sha}"
        md.append(f"- Commit: [{html.escape(sha[:7])}]({html.escape(commit_url)})\n")
    md.append("\n")

    md.append("## 4) Artifacts\n\nPlaceholder (Phase 2.6)\n\n")

    md.append("## 5) Commit History\n\n")
    md.append(f"- **Commit History:** [commitHistory.csv](https://gist.github.com/{html.escape(gist_id)})\n\n")

    write_summary("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

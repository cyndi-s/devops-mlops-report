import argparse
import csv
import html
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional
from urllib import request, error
from zoneinfo import ZoneInfo

import yaml

CSV_NAME_DEFAULT = "commitHistory.csv"

def get_tz(cfg: dict):
    name = str(cfg.get("timezone") or "").strip()
    if not name:
        return None, ""
    if name.upper() == "UTC":
        name = "UTC"
    try:
        return ZoneInfo(name), ""
    except Exception:
        return None, f"Invalid timezone '{name}'. Falling back to runner timezone."


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


def parse_kv(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = (s or "").strip()
    if not s:
        return out
    for p in [x.strip() for x in s.split(";") if x.strip()]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


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

def fmt_kv_3dp(s: str) -> str:
    """
    Format a kv string like 'a=1.23456; b=0.9; c=abc' into 3dp for numeric values.
    Preserves original key order as much as possible by processing segments in order.
    """
    s = (s or "").strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.split(";") if p.strip()]
    out_parts: List[str] = []
    for p in parts:
        if "=" not in p:
            out_parts.append(p)
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        out_parts.append(f"{k}={fmt_val(v)}")
    return "; ".join(out_parts)

def fmt_bytes(n: int) -> str:
    try:
        n = int(n or 0)
    except Exception:
        return ""
    units = ["B", "KB", "MB", "GB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.1f} {units[i]}" if i > 0 else f"{int(v)} {units[i]}"


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


def sh(args: List[str]) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ""


def write_summary(md: str) -> None:
    out = os.environ.get("GITHUB_STEP_SUMMARY") or ""
    if not out:
        return
    with open(out, "a", encoding="utf-8") as f:
        f.write(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gist-url", required=True)
    ap.add_argument("--csv-name", default=CSV_NAME_DEFAULT)
    ap.add_argument("--svg-json", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--devops-json", required=True)
    ap.add_argument("--artifacts-json", required=True)
    args = ap.parse_args()

    token = (os.environ.get("GIST_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing env GIST_TOKEN")

    cfg = load_cfg(args.config)
    tz, tz_warn = get_tz(cfg)
    report = cfg.get("report") or {}
    metric = str(report.get("highlight_metric") or "").strip()
    sharp_delta = float(report.get("sharp_delta") or 0.15)
    window = int(report.get("trend_window") or 10)

    if not metric:
        raise SystemExit("Missing report.highlight_metric in config")

    gist_id = extract_gist_id(args.gist_url)
    csv_text = get_gist_file_content(gist_id, token, args.csv_name) or ""
    rows: List[Dict[str, str]] = []
    if csv_text.strip():
        rdr = csv.DictReader(StringIO(csv_text))
        for r in rdr:
            rows.append({k: (v or "").strip() for k, v in (r or {}).items()})

    rows.sort(key=lambda r: r.get("timestamp_local", ""))

    trained_rows_all = [r for r in rows if is_true(r.get("is_trained", ""))]

    # DEDUPE by commit_sha: keep latest row per commit
    seen = set()
    trained_rows = []
    for r in reversed(trained_rows_all):
        sha_i = (r.get("commit_sha") or "").strip()
        if sha_i and sha_i in seen:
            continue
        if sha_i:
            seen.add(sha_i)
        trained_rows.append(r)
    trained_rows.reverse()

    latest = trained_rows[-1] if trained_rows else None
    prev = trained_rows[-2] if len(trained_rows) >= 2 else None

    # ---- Ground truth: trained_this_run (Phase 2.4 rule) ----
    # Prefer an explicit env flag if run_report.py sets it; fallback to CSV row with this commit and a non-empty run_id.
    trained_this_run = (os.environ.get("TRAINED_THIS_RUN") or "").strip().lower() == "true"
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    if not trained_this_run and sha:
        for r in reversed(rows):
            if (r.get("commit_sha") or "").strip() == sha:
                trained_this_run = is_true(r.get("is_trained", ""))
                break
    
    # ---- Does CSV contain ANY model signal? ----
    # (Used to decide whether to show Section 1 badge + Section 3 Model source)
    def row_has_model_signal(r: Dict[str, str]) -> bool:
        return bool(
            (r.get("mlflow_run_id") or "").strip()
            or (r.get("model_version") or "").strip()
        )

    has_any_model_info = trained_this_run or any(row_has_model_signal(r) for r in rows)


    # ---- SVG/model json ----
    with open(args.svg_json, "r", encoding="utf-8") as f:
        svgj = json.load(f) or {}
    svg_url = (svgj.get("svg_url") or "").strip()

    with open(args.model_json, "r", encoding="utf-8") as f:
        mj = json.load(f) or {}
    model_version = (mj.get("model_version") or "").strip()

    with open(args.devops_json, "r", encoding="utf-8") as f:
        dev = json.load(f) or {}
    mlflow_project_detected = str(dev.get("mlflow_project_detected", "")).lower() in ("yes", "true", "1")
    # NEW: failure details captured from workflow (run_report.py)
    training_attempted = bool(dev.get("training_attempted"))
    training_failed = bool(dev.get("training_failed"))
    training_run_id = str(dev.get("training_run_id") or "").strip()
    training_failure_reason = str(dev.get("training_failure_reason") or "").strip()
    if training_failure_reason and not training_failure_reason.lower().startswith("cause:"):
        training_failure_reason = f"Cause: {training_failure_reason}"
        
    # status from devops_json (fallback to env, then "Unknown")
    workflow_status = (dev.get("status") or dev.get("workflow_status") or "").strip()
    if not workflow_status:
        workflow_status = (os.environ.get("WORKFLOW_STATUS") or "").strip()
    if not workflow_status:
        workflow_status = "Unknown"

    with open(args.artifacts_json, "r", encoding="utf-8") as f:
        aj = json.load(f) or {}
    run_url = (aj.get("run_url") or "").strip()
    artifact_items = aj.get("items") or []

    # ---- helpers ----
    def metric_from_row(r: Dict[str, str], key: str) -> Optional[float]:
        m = parse_kv(r.get("mlflow_metrics_kv", ""))
        return safe_float(m.get(key))

    def duration_str(r: Dict[str, str]) -> str:
        # preferred: dedicated string column from MLflow ("12s", "3m 12s")
        s = (r.get("duration") or "").strip()
        if s:
            return s

        # fallback: old numeric minutes column if it exists
        dm = safe_float(r.get("duration_min"))
        if dm is None:
            m = parse_kv(r.get("mlflow_metrics_kv", ""))
            dm = safe_float(m.get("duration_min"))

        if dm is None:
            return ""

        total_sec = int(round(dm * 60))
        if total_sec < 60:
            return f"{total_sec}s"
        m_, s_ = divmod(total_sec, 60)
        return f"{m_}m {s_}s"


    # ---- Section 1 data ----
    badge_txt = "from this workflow run" if trained_this_run else "from a previous run"
    badge = f"<strong><ins><code>{html.escape(badge_txt)}</code></ins></strong>"

    s1_ts = (latest.get("timestamp_local") if latest else "") or ""
    s1_cause = (latest.get("cause") if latest else "") or ""

    # training_attempted should be about THIS commit, not the latest successful trained model
    this_row = None
    sha_env = (os.environ.get("GITHUB_SHA") or "").strip()
    if sha_env:
        for r in reversed(rows):
            if (r.get("commit_sha") or "").strip() == sha_env:
                this_row = r
                break

    this_cause = (this_row.get("cause") if this_row else "") or ""
    this_run_id = (this_row.get("mlflow_run_id") if this_row else "") or ""
    training_attempted = bool(this_cause or this_run_id)
    s1_params = (latest.get("mlflow_params_kv") if latest else "") or ""
    s1_metrics = (latest.get("mlflow_metrics_kv") if latest else "") or ""
    s1_metrics = fmt_kv_3dp(s1_metrics)
    s1_dur = duration_str(latest) if latest else ""

    cur_h = metric_from_row(latest, metric) if latest else None
    prev_h = metric_from_row(prev, metric) if prev else None

    mv_cell = model_version if model_version else "Not Registered"

    # ---- Section 2 points (last N trained) ----
    base = trained_rows[-window:] if trained_rows else []
    pts = []
    prev_val: Optional[float] = None
    for r in base:
        v = metric_from_row(r, metric)
        if v is None:
            continue
        d = 0.0 if prev_val is None else (v - prev_val)
        sharp = (abs(d) >= sharp_delta) if prev_val is not None else False
        prev_val = v
        pts.append({
            "ts": r.get("timestamp_local", ""),
            "branch": r.get("branch", ""),
            "author": r.get("author", ""),
            "cause": r.get("cause", "None") or "None",
            "val": v,
            "delta": d,
            "sharp": sharp,
            "dur": duration_str(r),
            "sha": r.get("commit_sha", ""),
        })

    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    server = (os.environ.get("GITHUB_SERVER_URL") or "https://github.com").strip()

    def commit_link(sha_full: str) -> str:
        short = (sha_full or "")[:7]
        if repo and sha_full:
            url = f"{server}/{repo}/commit/{sha_full}"
            return f'<a href="{html.escape(url)}"><code>{html.escape(short)}</code></a>'
        return f"<code>{html.escape(short)}</code>"

    # ---- Section 3 (use caller devops_json to avoid callee mixups) ----
    branch = (dev.get("branch") or "").strip()
    actor = (dev.get("author") or "").strip()
    sha = (dev.get("commit_sha") or "").strip()
    commit_msg = (dev.get("commit_msg") or "").strip()
    commit_url = (dev.get("commit_url") or "").strip()
    finished_at = (dev.get("finished_at") or "").strip()


    model_source = ""
    if has_any_model_info:
        model_source = "Trained in this run" if trained_this_run else "From previous run"


    # ---- Write summary ----
    md: List[str] = []
    md.append("# Pipeline Summary\n\n")

    # Section 1
    if True:
        # Show badge; add FAILED badge when this run attempted training but failed (from devops_json)
        fail_badge = ""
        if mlflow_project_detected and training_failed:
            fail_badge = " <strong><code>Training attempted: FAILED this run</code></strong>"

        if has_any_model_info:
            md.append(f"## 1) Latest Trained Model: {badge}{fail_badge}\n\n")
        else:
            md.append(f"## 1) Latest Trained Model{fail_badge}\n\n")

        if not latest:
            md.append("_No trained model found in commitHistory.csv yet._\n\n")
            # Show failure details even when there is no successful trained model yet
            if mlflow_project_detected and training_failed:
                md.append("**Training failure details (this run):**\n\n")
                md.append("- **What failed:** `Training attempted: FAILED this run`\n")
                md.append(f"- **Why it failed:** `{training_failure_reason or 'Cause: unknown'}`\n")
                md.append(f"- **Where it failed:** `Run ID: {training_run_id or 'NA'}`\n\n")
        else:
            md.append("<table style='width:100%; text-align:center;'>")
            md.append("<thead><tr>"
                    "<th>Timestamp<br>(Toronto)</th>"
                    "<th>model_version</th>"
                    "<th>Cause</th>"
                    "<th>Parameters</th>"
                    "<th>Metrics</th>"
                    f"<th>Δ{html.escape(metric)}</th>"
                    "<th>Duration</th>"
                    "</tr></thead><tbody><tr>")

            cells = [
                html.escape(s1_ts),
                f"<code>{html.escape(mv_cell)}</code>",
                f"<code>{html.escape(s1_cause)}</code>",
                f"<code>{html.escape(s1_params)}</code>",
                f"<code>{html.escape(s1_metrics)}</code>",
                f"<code>{html.escape(arrow_delta(prev_h, cur_h))}</code>",
                html.escape(s1_dur or ""),
            ]
            for c in cells:
                md.append(f"<td style='text-align:center; vertical-align:middle; word-break:break-word; max-width:100%'>{c}</td>")
            md.append("</tr></tbody></table>\n\n")

            if mlflow_project_detected and training_failed:
                md.append("**Training failure details (this run):**\n\n")
                md.append("- **What failed:** `Training attempted: FAILED this run`\n")
                md.append(f"- **Why it failed:** `{training_failure_reason or 'Cause: unknown'}`\n")
                md.append(f"- **Where it failed:** `Run ID: {training_run_id or 'NA'}`\n\n")


    # Section 2
    md.append(f"## 2) Model Performance ({html.escape(metric)})\n\n")
    if svg_url:
        md.append(f"<img alt='{html.escape(metric)} trend' src='{html.escape(svg_url)}' style='width:100%; height:auto; display:block;'/>\n\n")
    else:
        md.append("<em>SVG not available (no metric points found).</em>\n\n")

    md.append("<table style='width:100%; table-layout:auto; border-collapse:collapse;'>")
    md.append("<thead><tr>"
          "<th>#</th>"
          "<th>Timestamp</th>"
          "<th>Branch</th>"
          "<th>Author</th>"
          "<th>Cause</th>"
          f"<th>{html.escape(metric)}</th>"
          f"<th>Δ{html.escape(metric)}</th>"
          "<th>Duration</th>"
          "<th>Commit</th>"
          "</tr></thead><tbody>")

    # Use computed per-step delta; highlight sharp rows only (>= sharp_delta)
    for i, p in enumerate(pts):
        idx = i + 1  # 1..N (matches SVG x-axis)
        val_txt = fmt_val(p["val"])

        # delta display: first point → 0
        if i == 0:
            d_txt = "→ 0.000"
            is_sharp = False
        else:
            prev_val_for_display = p["val"] - p["delta"]
            d_txt = arrow_delta(prev_val_for_display, p["val"])
            is_sharp = bool(p["sharp"])

        row_cells = [
            f"<code>{idx}</code>",
            html.escape(p["ts"]),
            f"<code>{html.escape(p['branch'])}</code>",
            f"<code>{html.escape(p['author'])}</code>",
            f"<code>{html.escape(p['cause'])}</code>",
            html.escape(val_txt),
            html.escape(d_txt),
            html.escape(p["dur"] or ""),
            commit_link(p["sha"]),
        ]

        if is_sharp:
            row_cells = [f"<strong><ins>{c}</ins></strong>" for c in row_cells]

        md.append("<tr>")
        for c in row_cells:
            md.append(f"<td style='text-align:center; vertical-align:middle; word-break:break-word; max-width:100%'>{c}</td>")
        md.append("</tr>")

    md.append("</tbody></table>\n\n")

    # Section 3 (your 6 items)
    md.append("## 3) Code\n\n")
    if branch:
        md.append(f"- **Branch:** <code>{html.escape(branch)}</code>\n")
    if actor:
        md.append(f"- **Author:** <code>{html.escape(actor)}</code>\n")
    if sha and repo:
        commit_url = f"{server}/{repo}/commit/{sha}"
        md.append(f'- **Commit:** <a href="{html.escape(commit_url)}"><code>{html.escape(sha[:7])}</code> — {html.escape(commit_msg)}</a>\n')
    if model_source:
        md.append(f"- **Model source:** {html.escape(model_source)}\n")
    md.append(f"- **Status:** <code>{html.escape(workflow_status)}</code>\n")
    if finished_at:
        md.append(f"- **Job finished at:** {html.escape(finished_at)}\n")


    md.append("## 4) Commit History\n\n")
    md.append(f"- **Commit History:** [commitHistory.csv](https://gist.github.com/{html.escape(gist_id)})\n\n")


    # ---- Section 5: MLflow Artifacts ----
    md.append("## 5) MLflow Artifacts (tracking UI)\n\n")
    tracking_uri = str((cfg.get("mlflow") or {}).get("tracking_uri") or "").strip()
    rid = (latest.get("mlflow_run_id") or "").strip() if latest else ""
    src_tag = "this workflow run" if trained_this_run else "previous run"

    if tracking_uri and rid:
        exp_id = (latest.get("experiment_id") or "").strip() if latest else ""
        if not exp_id:
            exp_id = "0"
        run_link = f"{tracking_uri}/#/experiments/{exp_id}/runs/{rid}"
        md.append(
            f'- MLflow run (tracking UI, {html.escape(src_tag)}): '
            f'<a href="{html.escape(run_link)}"><code>{html.escape(rid)}</code></a>\n'
        )
    else:
        md.append("- MLflow run (tracking UI): Not available (missing tracking_uri or run_id)\n")

    write_summary("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

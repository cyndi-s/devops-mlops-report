import argparse
import csv
import json
import math
import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
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


def update_gist_file(gist_id: str, token: str, filename: str, content: str) -> None:
    payload = {"files": {filename: {"content": content}}}
    gh_api_request("PATCH", f"https://api.github.com/gists/{gist_id}", token, payload)


def get_gist_raw_url(gist_id: str, token: str, filename: str) -> str:
    gist = gh_api_request("GET", f"https://api.github.com/gists/{gist_id}", token)
    files = gist.get("files", {})
    if filename not in files:
        return ""
    return (files[filename].get("raw_url") or "").strip()


def parse_kv_string(s: str) -> Dict[str, str]:
    # "k=v; k2=v2" -> dict
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


def is_trained_row(r: Dict[str, str]) -> bool:
    return (r.get("is_trained") or "").strip().lower() in ("yes", "true", "1")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "NA"):
            return None
        return float(x)
    except Exception:
        return None


def median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def build_points_from_csv(
    rows: List[Dict[str, str]],
    metric_key: str,
    window: int,
) -> List[Dict[str, Any]]:
    # rows expected sorted by timestamp ascending (oldest -> newest)
    trained = [r for r in rows if is_trained_row(r)]
    if not trained:
        return []

    # keep last window trained rows
    seq = trained[-window:]

    pts: List[Dict[str, Any]] = []
    prev_val: Optional[float] = None
    for r in seq:
        m = parse_kv_string(r.get("mlflow_metrics_kv", ""))
        v = safe_float(m.get(metric_key))
        if v is None:
            # skip missing metric rows
            continue
        d = 0.0 if prev_val is None else (v - prev_val)
        prev_val = v
        pts.append({
            "ts": (r.get("timestamp_toronto") or ""),
            "val": v,
            "delta": d,
        })
    return pts


def moving_average(vals: List[float], k: int = 3) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(vals)):
        w = vals[max(0, i - (k - 1)): i + 1]
        out.append(sum(w) / len(w) if w else None)
    return out


def build_svg(
    pts: List[Dict[str, Any]],
    metric_key: str,
    sharp_delta: float,
    width: int = 980,
    height: int = 360,
    pad: int = 44,
) -> str:
    if not pts:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"

    vals = [p["val"] for p in pts if p.get("val") is not None]
    if not vals:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"

    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    def sx(i: int) -> float:
        if len(pts) == 1:
            return pad + (width - 2 * pad) / 2
        return pad + (width - 2 * pad) * (i / (len(pts) - 1))

    def sy(v: float) -> float:
        return pad + (height - 2 * pad) * (1 - (v - vmin) / (vmax - vmin))

    med = median(vals)
    med_y = sy(med)

    ma = moving_average(vals, 3)

    # labels: show every 2 points + last
    labels: List[Tuple[float, float, str]] = []
    for i, p in enumerate(pts):
        if i % 2 == 0 or i == len(pts) - 1:
            t = (p["ts"] or "")[5:16]  # MM-DD HH:MM (from YYYY-MM-DD HH:MM:SS)
            labels.append((sx(i), height - 12, t))

    # polylines
    main_poly = " ".join([f"{sx(i):.1f},{sy(p['val']):.1f}" for i, p in enumerate(pts)])

    ma_pts = []
    for i, mv in enumerate(ma):
        if mv is None:
            continue
        ma_pts.append(f"{sx(i):.1f},{sy(mv):.1f}")
    ma_poly = " ".join(ma_pts)

    out: List[str] = []
    out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    out.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")

    # axes
    out.append(f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#000' stroke-width='1'/>")
    out.append(f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#000' stroke-width='1'/>")

    # median dashed line
    out.append(f"<line x1='{pad}' y1='{med_y:.1f}' x2='{width-pad}' y2='{med_y:.1f}' stroke='red' stroke-dasharray='6,4' stroke-width='2'/>")

    # MA line
    if ma_poly:
        out.append(f"<polyline points='{ma_poly}' fill='none' stroke='#f4a261' stroke-width='2' opacity='0.9'/>")

    # main line
    out.append(f"<polyline points='{main_poly}' fill='none' stroke='#1f77b4' stroke-width='3'/>")

    # points + sharp markers + value labels
    prev_val: Optional[float] = None
    for i, p in enumerate(pts):
        v = p["val"]
        d = 0.0 if prev_val is None else (v - prev_val)
        sharp = (abs(d) >= sharp_delta) if prev_val is not None else False
        prev_val = v

        x, y = sx(i), sy(v)
        out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#1f77b4'/>")
        if sharp:
            out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='red' opacity='0.9'/>")
        out.append(f"<text x='{x:.1f}' y='{(y-10):.1f}' font-size='12' text-anchor='middle'>{v:.3f}</text>")

    # x labels
    for x, y, t in labels:
        out.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='11' text-anchor='middle'>{t}</text>")

    # title + legend
    out.append(f"<text x='{width/2:.1f}' y='20' font-size='16' text-anchor='middle'>Model Performance ({metric_key})</text>")
    out.append(f"<text x='{pad}' y='{pad-10}' font-size='12'>red dot: abs(Δ) ≥ {sharp_delta:.2f}</text>")

    out.append("</svg>")
    return "\n".join(out)


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

    # sort by timestamp_toronto asc
    rows.sort(key=lambda r: r.get("timestamp_toronto", ""))

    pts = build_points_from_csv(rows, metric_key, trend_window)
    svg_name = f"trend_{metric_key}.svg"
    svg = build_svg(pts, metric_key, sharp_delta)

    update_gist_file(gist_id, token, svg_name, svg)
    raw_url = get_gist_raw_url(gist_id, token, svg_name)

    # cache-bust
    run_id = (os.environ.get("GITHUB_RUN_ID") or "").strip()
    svg_url = f"{raw_url}?ts={run_id}" if (raw_url and run_id) else raw_url

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": metric_key,
                "sharp_delta": sharp_delta,
                "trend_window": trend_window,
                "svg_name": svg_name,
                "svg_raw_url": raw_url,
                "svg_url": svg_url,
                "points_used": len(pts),
            },
            f,
            indent=2,
        )

    print(f"Uploaded {svg_name} to gist; points_used={len(pts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

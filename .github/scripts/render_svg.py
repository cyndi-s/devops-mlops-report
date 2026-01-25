import argparse
import csv
import json
import os
import re
import time
from io import StringIO
from typing import Any, Dict, List, Optional
from urllib import request, error

import yaml
import matplotlib.pyplot as plt



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
    f = (gist.get("files", {}) or {}).get(filename, {}) or {}
    return (f.get("raw_url") or "").strip()

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


def median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def moving_average(vals: List[float], k: int = 3) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(vals)):
        w = vals[max(0, i - (k - 1)) : i + 1]
        out.append(sum(w) / len(w) if w else None)
    return out


def build_svg(timestamps: List[str], vals: List[float], sharp_flags: List[bool], metric: str, sharp_delta: float) -> str:
    # x-axis is Run # (1..N), matching the Trend table numbering
    if not vals:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"

    xs = list(range(1, len(vals) + 1))

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    # Make axes L-shaped (left + bottom only)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


    # main line
    ax.plot(xs, vals, marker="o", label=metric)

    # sharp change points
    sharp_x = [xs[i] for i, f in enumerate(sharp_flags) if f]
    sharp_y = [vals[i] for i, f in enumerate(sharp_flags) if f]
    # Always show sharp-change threshold in legend (even if no sharp changes)
    ax.scatter([], [], c="red", label=f"sharp change (≥ {sharp_delta})")

    if sharp_x:
        ax.scatter(sharp_x, sharp_y, c="red", zorder=5)

    # median line
    med = median(vals)
    ax.axhline(med, linestyle="--", label=f"Median Performance ({med:.3f})")

    # 3-run moving average
    ma = moving_average(vals, 3)
    ma_x = [xs[i] for i, v in enumerate(ma) if v is not None]
    ma_y = [v for v in ma if v is not None]
    if ma_x:
        ax.plot(ma_x, ma_y, label="3-run MA", linewidth=2, alpha=0.7)
    
    ax.legend(loc="best")

    ax.set_title(f"Model Performance ({metric})")
    ax.set_xlabel("Run #")
    ax.set_ylabel(f"Performance ({metric})")

    # grid like your example
    ax.grid(True, linestyle="--", alpha=0.4)

    # show all ticks 1..N (N <= trend_window, usually 10)
    ax.set_xticks(xs)

    
    fig.tight_layout()

    from io import StringIO as _StringIO
    buf = _StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gist-url", required=True)
    ap.add_argument("--csv-name", default=CSV_NAME_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    token = (os.environ.get("GIST_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing env GIST_TOKEN")

    cfg = load_cfg(args.config)
    report = cfg.get("report") or {}
    metric = str(report.get("highlight_metric") or "").strip()
    if not metric:
        raise SystemExit("Missing report.highlight_metric in config")

    sharp_delta = float(report.get("sharp_delta") or 0.15)
    window = int(report.get("trend_window") or 10)

    gist_id = extract_gist_id(args.gist_url)
    csv_text = get_gist_file_content(gist_id, token, args.csv_name) or ""
    rows: List[Dict[str, str]] = []
    if csv_text.strip():
        rdr = csv.DictReader(StringIO(csv_text))
        for r in rdr:
            rows.append({k: (v or "").strip() for k, v in (r or {}).items()})

    rows.sort(key=lambda r: r.get("timestamp_local", ""))

    trained_all = [r for r in rows if is_true(r.get("is_trained", ""))]

    # DEDUPE by commit_sha: keep latest row per commit
    seen = set()
    trained = []
    for r in reversed(trained_all):
        sha = (r.get("commit_sha") or "").strip()
        if sha and sha in seen:
            continue
        if sha:
            seen.add(sha)
        trained.append(r)
    trained.reverse()

    # apply window AFTER dedupe
    trained = trained[-window:]


    timestamps: List[str] = []
    vals: List[float] = []
    sharp_flags: List[bool] = []

    prev: Optional[float] = None
    for r in trained:
        m = parse_kv(r.get("mlflow_metrics_kv", ""))
        v = safe_float(m.get(metric))
        if v is None:
            continue
        d = 0.0 if prev is None else (v - prev)
        sharp = (abs(d) >= sharp_delta) if prev is not None else False
        prev = v

        timestamps.append(r.get("timestamp_local", ""))
        vals.append(v)
        sharp_flags.append(sharp)

    svg_name = f"trend_{metric}.svg"
    svg = build_svg(timestamps, vals, sharp_flags, metric, sharp_delta)

    last_err = None
    for attempt in range(6):  # ~1+2+4+8+16 sec total wait
        try:
            update_gist_file(gist_id, token, svg_name, svg)
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            # GitHub sometimes returns 409 right after another PATCH to the same gist
            if " 409 " in msg or "409 Conflict" in msg:
                time.sleep(2 ** attempt)
                continue
            raise

    if last_err is not None:
        raise last_err

    raw_url = get_gist_raw_url(gist_id, token, svg_name)
    run_id = (os.environ.get("GITHUB_RUN_ID") or "").strip()
    svg_url = f"{raw_url}?ts={run_id}" if (raw_url and run_id) else raw_url

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": metric,
                "sharp_delta": sharp_delta,
                "trend_window": window,
                "svg_name": svg_name,
                "svg_raw_url": raw_url,
                "svg_url": svg_url,
                "points_used": len(vals),
            },
            f,
            indent=2,
        )

    print(f"SVG uploaded: {svg_name}, points_used={len(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
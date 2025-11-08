#!/usr/bin/env python3
import os, html, csv, io, urllib.request
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient
from zoneinfo import ZoneInfo
import subprocess

LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Toronto")
LOCAL_ZONE = ZoneInfo(LOCAL_TZ)

SUMMARY = os.getenv("GITHUB_STEP_SUMMARY", "/dev/stdout")
REPO = os.getenv("GITHUB_REPOSITORY", "")
SHA  = os.getenv("GITHUB_SHA", "")
BRANCH = os.getenv("GITHUB_REF_NAME", "")
ACTOR = os.getenv("GITHUB_ACTOR", "")
COMMIT_MSG = os.getenv("GITHUB_EVENT_HEAD_COMMIT_MESSAGE", "")
commit_url = f"https://github.com/{REPO}/commit/{SHA}" if (REPO and SHA) else ""
CSV_URL = os.getenv("COMMIT_CSV_RAW_URL", "")
client = MlflowClient()

# --- stable params first; extras appended (flexible) ---
STABLE_PARAM_KEYS = [
    "epochs", "opt_name", "opt_learning_rate",
    "monitor", "patience", "restore_best_weights"
]

def _load_cause_maps(url: str):
    cause_by_sha, cause_by_short, cause_by_version = {}, {}, {}
    if not url:
        return cause_by_sha, cause_by_short, cause_by_version
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        r = csv.DictReader(io.StringIO(data))
        for row in r:
            sha_full = (row.get("commit_id") or row.get("commit") or "").strip()
            short    = (sha_full[:7] if sha_full else (row.get("commit_short") or "")).strip()
            cause    = (row.get("cause_mlops") or "Script").strip() or "Script"  # no "None"
            ver      = (row.get("model_version") or "").strip()
            if sha_full: cause_by_sha[sha_full] = cause
            if short:    cause_by_short[short]  = cause
            if ver:      cause_by_version[ver]  = cause
    except Exception as e:
        print("WARN: could not load commitHistory.csv:", e)
    return cause_by_sha, cause_by_short, cause_by_version

CAUSE_BY_SHA, CAUSE_BY_SHORT, CAUSE_BY_VER = _load_cause_maps(CSV_URL)

def sh(args):
    try: return subprocess.check_output(args, text=True).strip()
    except Exception: return ""
    
def ts(ms):
    if not ms:
        return ""
    dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(ZoneInfo(LOCAL_TZ))
    return dt.strftime("%Y-%m-%d %H:%M")

def ts_from_epoch(s):
    try:
        dt = datetime.fromtimestamp(int(s), tz=timezone.utc).astimezone(LOCAL_ZONE)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ""

def fmt_val(x):
    """3 decimals; scientific if very small."""
    try:
        x = float(x)
        return f"{x:.3f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"
    except Exception:
        return "" if x in ("", None) else str(x)

def fmt_arrow_delta(a, b):
    """Arrow + absolute delta with 3 decimals (or scientific)."""
    try:
        if a in ("", None) or b in ("", None):
            return ""
        d = float(b) - float(a)
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        mag = abs(d)
        mag_s = f"{mag:.3f}" if mag >= 1e-3 or mag == 0 else f"{mag:.1e}"
        return f"{arrow} {mag_s}"
    except Exception:
        return ""
    
def fmt_arrow_minutes(d):
    """Arrow + absolute delta in minutes (1 decimals)."""
    try:
        if d in ("", None): 
            return ""
        d = float(d)
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        return f"{arrow} {abs(d):.1f}"
    except Exception:
        return ""    
    
def fmt_arrow_from_value(d):
    try:
        d = float(d)
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        mag = abs(d)
        return f"{arrow} {(f'{mag:.3f}' if mag >= 1e-3 or mag == 0 else f'{mag:.1e}')}"
    except Exception:
        return ""
    
def fmt_dur(m):
    return "" if m in (None, "") else f"{float(m):.1f}"

def get_default_exp():
    return mlflow.get_experiment_by_name("Default") or client.get_experiment("0")

def get_runs(exp_id, n):
    return client.search_runs(
        experiment_ids=[exp_id],
        max_results=n,
        order_by=["attributes.start_time DESC"],
    )

def model_version_for(run_id):
    try:
        for mv in client.search_model_versions(f"run_id = '{run_id}'"):
            return mv.version
    except Exception:
        pass
    return ""

def m(run, key): return "" if not run else run.data.metrics.get(key, "")
def p(run, key): return "" if not run else run.data.params.get(key, "")
def t(run, key): return "" if not run else run.data.tags.get(key, "")

exp = get_default_exp()
runs = get_runs(exp.experiment_id, 11) if exp else []
latest = runs[0] if runs else None
prev   = runs[1] if len(runs)>1 else None

# latest row fields
ts_end = ts(latest.info.end_time) if (latest and latest.info.end_time) else ""
run_id = latest.info.run_id if latest else ""
vacc   = m(latest,"val_accuracy"); pvacc = m(prev,"val_accuracy")
vlos   = m(latest,"val_loss");     plos  = m(prev,"val_loss")
acc    = m(latest,"accuracy");     los   = m(latest,"loss")
dur_ms = (latest.info.end_time - latest.info.start_time) if (latest and latest.info.end_time and latest.info.start_time) else None
prev_dur_ms = (
    (prev.info.end_time - prev.info.start_time)
    if (prev and prev.info.end_time and prev.info.start_time) else None
    )
dur_tx = f"{round(dur_ms/60000,1)}" if dur_ms else ""
ddur_tx  = ((dur_ms - prev_dur_ms)/60000.0) if (dur_ms is not None and prev_dur_ms is not None) else None
vers   = model_version_for(run_id)

# --- Flexible Parameters: stable first, extras after ---
params_all = dict(latest.data.params) if latest else {}
stable_pairs = []
for k in STABLE_PARAM_KEYS:
    if k in params_all:
        v = params_all.pop(k)
        # keep learning-rate pretty
        v = fmt_val(v) if k == "opt_learning_rate" else v
        stable_pairs.append(f"{k}={v}")

# extras compact (sorted for determinism)
extra_pairs = [f"{k}={params_all[k]}" for k in sorted(params_all.keys())]
params_str = ", ".join([*stable_pairs, *extra_pairs]) if (stable_pairs or extra_pairs) else ""
params_html = f"<code>{html.escape(params_str)}</code>" if params_str else "<code>-</code>"

# Code section
SHARP_DELTA = float(os.getenv("SHARP_DELTA", "0.15"))

def build_val_accuracy_points(runs, repo, window=10):
    """Return points oldest→newest with Δval_accuracy computed against the *true* previous run."""
    prev_seed = None
    if len(runs) > window:
        seed_va = runs[window].data.metrics.get("val_accuracy")
        try:
            prev_seed = float(seed_va) if seed_va is not None else None
        except Exception:
            prev_seed = None

    seq = list(reversed(runs[:window]))  # oldest→newest
    points, prev_va = [], prev_seed

    for r in seq:
        va_raw = r.data.metrics.get("val_accuracy")
        if va_raw is None:
            continue

        end = r.info.end_time
        start = r.info.start_time
        dur_min = ((end - start)/60000.0) if (end and start) else None

        end_or_start = end or start
        ts_str = datetime.fromtimestamp(end_or_start/1000, tz=timezone.utc) \
                   .astimezone(LOCAL_ZONE).strftime("%Y-%m-%d %H:%M")

        commit = r.data.tags.get("mlflow.source.git.commit") or r.data.tags.get("git.commit") or ""
        short  = commit[:7] if commit else ""
        url    = f"https://github.com/{repo}/commit/{commit}" if (repo and commit) else ""

        va = float(va_raw)
        delta = (va - prev_va) if (prev_va is not None) else 0.0

        points.append({
            "timestamp": ts_str,
            "val_accuracy": va,
            "delta": delta,
            "short": short,
            "url": url,
            "branch": BRANCH,
            "author": ACTOR,
            "dur_min": dur_min,
        })
        prev_va = va

    return points


# Write summary
lines = []
lines.append("# Pipeline Summary\n")

# --- Section 1: MLflow Summary Table ---
trained_flag = (os.getenv("TRAINED_THIS_RUN", "").lower() == "true")
# Fallback: check timestamp window if flag missing
if not trained_flag:
    try:
        js = int(os.getenv("JOB_START", "0")) * 1000
        je = int(os.getenv("JOB_END", "0")) * 1000
        if latest and latest.info and latest.info.end_time and js and je:
            trained_flag = (js <= latest.info.end_time <= je)
    except Exception:
        pass

status_text = "from this workflow run" if trained_flag else "from a previous run"
badge = f"<strong><ins><code>{html.escape(status_text)}</code></ins></strong>"
lines.append(f"## 1) Model in APK: {badge}\n")

# Cause: never "None"; default Script
cause_current = os.getenv("CAUSE_MLOPS") or "Script"
_cbv = globals().get("CAUSE_BY_VER")   or {}
_cbs = globals().get("CAUSE_BY_SHA")   or {}
_cb7 = globals().get("CAUSE_BY_SHORT") or {}

vers_key = str(vers) if vers is not None else ""
sha_full = SHA if 'SHA' in globals() else ""
sha_short = (SHA[:7] if 'SHA' in globals() and SHA else "")

cause_prev = _cbv.get(vers_key) or _cbs.get(sha_full) or _cb7.get(sha_short)
cause_s1 = cause_current if trained_flag else (cause_prev or "Script")
cause_cell = f"<code>{html.escape(cause_s1)}</code>"

lines.append('<table style="width:100%; text-align:center;">')
lines.append('<thead><tr>'
             f'<th>Timestamp<br>({LOCAL_TZ.split("/")[-1]})</th>'
             '<th>model_version</th><th>Cause</th><th>Parameters</th>'
             '<th>accuracy</th><th>val_accuracy</th><th>Δval_accuracy</th>'
             '<th>loss</th><th>val_loss</th><th>Δval_loss</th>'
             '<th>Duration<br>(min)</th><th>ΔDuration<br>(min)</th>'
             '</tr></thead>')
lines.append('<tbody><tr>')
for cell in [
    ts_end, vers or "", cause_cell, params_html,
    fmt_val(acc), fmt_val(vacc), fmt_arrow_delta(pvacc, vacc),
    fmt_val(los), fmt_val(vlos), fmt_arrow_delta(plos, vlos),
    dur_tx, fmt_arrow_minutes(ddur_tx)
]:
    lines.append(
        f"<td style='text-align:center; vertical-align:middle; "
        f"word-break:break-word; max-width:100%'>{cell}</td>"
    )
lines.append("</tr></tbody></table>")          
             
lines.append("\n")

# --- Section 2: Model Performance (val_accuracy, last 10) ---
lines.append("## 2) Model Performance (val_accuracy)\n")
points = build_val_accuracy_points(runs, REPO)   # oldest → newest
lines.append("<div style='display:flex; flex-wrap:wrap; align-items:flex-start; gap:16px;'>")
# LEFT: SVG (62%)
lines.append("<div style='flex:1 1 620px; min-width:320px;'>")
val_svg = os.getenv("VAL_SVG_URL", "")
if val_svg:
    cache_bust = os.getenv("GITHUB_RUN_ID", "")
    lines.append(f"<img alt='val_accuracy trend' src='{val_svg}?ts={cache_bust}' style='width:100%; height:auto; display:block;'/>")
else:
    lines.append("<em>SVG not available this run.</em>")
lines.append("</div>")

# RIGHT: table 
lines.append("<div style='flex:1 1 360px; min-width:320px;'>")
lines.append("<table style='width:100%; table-layout:auto; border-collapse:collapse;'>")
lines.append("<thead><tr>")
for col in [
    f"Timestamp<br>({LOCAL_TZ.split('/')[-1]})",
    "Branch", "Author", "Cause",
    "val_accuracy", "Δval_accuracy",
    "Duration<br>(min)", "Commit"
]:
    lines.append(f"<th style='text-align:center'>{col}</th>")
lines.append("</tr></thead><tbody>")

for pt in points:
    ts_cell = pt["timestamp"]
    br      = pt["branch"] or ""
    au      = pt["author"] or ""
    va      = pt["val_accuracy"]
    dval    = pt["delta"]
    d_txt   = fmt_dur(pt.get("dur_min"))
    short   = pt.get("short") or ""
    url     = pt.get("url") or ""
    sha_full_row = pt.get("sha") or (url.split("/commit/")[-1] if "/commit/" in url else "")
    cause_row = CAUSE_BY_SHA.get(sha_full_row) or CAUSE_BY_SHORT.get(short) or "Script"  # no "None"
    commit_cell_html = (
        f'<a href="{url}"><code>{html.escape(short)}</code></a>' if url
        else f'<code>{html.escape(short)}</code>'
    )

    cells = [
        ts_cell,
        f"<code>{html.escape(br)}</code>",
        f"<code>{html.escape(au)}</code>",
        f"<code>{html.escape(cause_row)}</code>",
        fmt_val(va),
        fmt_arrow_from_value(dval),
        d_txt or "",
        commit_cell_html
    ]

    try:
        dv = float(str(dval).replace('%','').strip()) if dval is not None else None
    except (TypeError, ValueError):
        dv = None

    is_sharp = (abs(dv) >= SHARP_DELTA) if dv is not None else False
    if is_sharp:
        cells = [f"<strong><ins>{c}</ins></strong>" for c in cells]
    lines.append("<tr>")
    for cell in cells:
        lines.append(
            f"<td style='text-align:center; vertical-align:middle; "
            f"word-break:break-word; max-width:100%'>{cell}</td>"
        )
    lines.append("</tr>")

lines.append("</tbody></table>")
lines.append("</div>")
lines.append("</div>")
#################################################

# --- Section 3: Code ---
lines.append("\n## 3) Code\n")
if BRANCH: lines.append(f"- **Branch:** `{BRANCH}`")
if ACTOR:  lines.append(f"- **Author:** `{ACTOR}`")
if SHA and REPO:
    msg = (COMMIT_MSG or sh(["git", "log", "-1", "--pretty=%s"])) or ""
    lines.append(f'- **Commit:** <a href="https://github.com/{REPO}/commit/{SHA}"><code>{SHA[:7]}</code> — {msg}</a>')

src = os.getenv("MODEL_SOURCE", "")
mname = os.getenv("MODEL_NAME", "")
mver = os.getenv("MODEL_VERSION", "")
mstage = os.getenv("MODEL_STAGE", "")
if src == "Registry":
    extra = f" (stage: {mstage}, v{mver})" if mver else ""
    lines.append(f"- **Model source in APK:** Registry — {html.escape(mname)}{extra}")
elif src == "Trained":
    lines.append("- **Model source in APK:** Trained in this run")

lines.append("- **Status:** Success")

job_end_str = ts_from_epoch(os.getenv("JOB_END", "")) if "ts_from_epoch" in globals() else ""
if job_end_str:
    lines.append(f"- **Job finished at:** {job_end_str}")

# --- Section 4: Artifacts ---
lines.append("\n## 4) Artifacts\n")
lines.append("| Artifact | Size |")
lines.append("|---|---:|")

artifacts_dir = "POC-Image-Classification/artifacts"
if os.path.isdir(artifacts_dir):
    arts = sorted(os.listdir(artifacts_dir))
    if arts:
        for a in arts:
            apath = os.path.join(artifacts_dir, a)
            sz = os.path.getsize(apath) if os.path.isfile(apath) else 0
            sz_kb = f"{sz/1024:.1f} KB" if sz else ""
            rel = os.path.relpath(apath, os.getcwd())
            lines.append(f"| [{a}]({rel}) | {sz_kb} |")
    else:
        lines.append("| _None_ | |")
else:
    lines.append("| _None_ | |")

# --- Section 5: Commit History (exact gist link, same as POC-Mobile) ---
gist_url = os.getenv("GIST_URL",
    "https://gist.github.com/cyndi-s/a0ace91cc2de1aeb3340a3dc29ba9c9f")
if gist_url:
    lines.append("\n## 5) Commit History\n")
    lines.append(f"- **Commit History:** [commitHistory.csv]({gist_url})")

with open(SUMMARY, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("Summary generated.")

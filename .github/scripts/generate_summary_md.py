#!/usr/bin/env python3
import os
import csv
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# ---- Environment / constants ----
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Toronto")
LOCAL_ZONE = ZoneInfo(LOCAL_TZ)

SUMMARY = os.getenv("GITHUB_STEP_SUMMARY", "/dev/stdout")
REPO = os.getenv("GITHUB_REPOSITORY", "")
SHA = os.getenv("GITHUB_SHA", "")
BRANCH = os.getenv("GITHUB_REF_NAME", "")
ACTOR = os.getenv("GITHUB_ACTOR", "")
COMMIT_MSG = os.getenv("GITHUB_EVENT_HEAD_COMMIT_MESSAGE", "")
SHARP_DELTA = float(os.getenv("SHARP_DELTA", "0.15"))

def ts_from_epoch(s: str) -> str:
    try:
        dt = datetime.fromtimestamp(int(s), tz=ZoneInfo("UTC")).astimezone(LOCAL_ZONE)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ""

# ---------- Formatting helpers (copied from concept script) ----------

def diff(a, b):
    try:
        if a in ("", None) or b in ("", None):
            return ""
        return f"{float(b) - float(a):+.3f}"
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


# ---------- Data loading ----------

def load_commit_history():
    candidates = [
        ".mlops/commitHistory.csv",
        "commitHistory.csv",
        "artifacts/commitHistory.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return path, rows
    return None, []


def parse_float(v):
    try:
        return float(v)
    except Exception:
        return None


def main():
    csv_path, rows = load_commit_history()

    lines = []
    lines.append("# Pipeline Summary\n")

    if not rows:
        lines.append("No `commitHistory.csv` found. The mlops-plugin did not log any runs yet.\n")
        with open(SUMMARY, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    # assume rows are oldest -> newest; protect if not
    # try to sort by timestamp column if it exists
    ts_field = None
    for k in rows[0].keys():
        lk = k.lower()
        if "timestamp" in lk:
            ts_field = k
            break
    if ts_field:
        rows = sorted(
            rows,
            key=lambda r: r.get(ts_field, ""),
        )

    latest = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None

    # ---- Section 1: latest run summary ----
    # timestamp
    ts_end = latest.get(ts_field, "")

    # metrics
    acc = latest.get("accuracy", "")
    vacc = latest.get("val_accuracy", "")
    los = latest.get("loss", "")
    vlos = latest.get("val_loss", "")
    pvacc = prev.get("val_accuracy", "") if prev else ""
    plos = prev.get("val_loss", "") if prev else ""

    # duration
    dur = latest.get("train_min", latest.get("training_duration", ""))
    prev_dur = prev.get("train_min", prev.get("training_duration", "")) if prev else ""
    dur_tx = fmt_dur(dur)
    d_dur = None
    if dur not in ("", None) and prev_dur not in ("", None):
        try:
            d_dur = float(dur) - float(prev_dur)
        except Exception:
            d_dur = None

    # params from columns
    pe = latest.get("epochs", "")
    po = latest.get("opt_name", "")
    plr = latest.get("opt_learning_rate", "")
    pm = latest.get("monitor", "")
    pp = latest.get("patience", "")
    prbw = latest.get("restore_best_weights", "")

    params_str = (
        f"epochs={pe}, opt_name={po}, opt_learning_rate={fmt_val(plr)}, "
        f"monitor={pm}, patience={pp}, restore_best_weights={prbw}"
    )
    params_html = f"<code>{html.escape(params_str)}</code>"

    model_version = latest.get("model_version", "")
    cause = latest.get("cause_mlops", latest.get("Cause", "None")) or "None"

    # trained flag from CSV (or env)
    trained_csv = str(latest.get("trained_model", latest.get("TRAINED", ""))).lower()
    trained_flag = trained_csv == "true"
    if not trained_flag:
        trained_env = (os.getenv("TRAINED_THIS_RUN", "").lower() == "true") or (
            os.getenv("TRAINED", "").lower() == "true"
        )
        trained_flag = trained_env

    status_text = "from this workflow run" if trained_flag else "from a previous run"
    badge = f"<strong><ins><code>{html.escape(status_text)}</code></ins></strong>"

    # cause cell
    cause_cell = f"<code>{html.escape(cause)}</code>"

    lines.append(f"## 1) Model in APK: {badge}\n")

    lines.append('<table style="width:100%; text-align:center;">')
    lines.append(
        "<thead><tr>"
        f"<th>Timestamp<br>({LOCAL_TZ.split('/')[-1]})</th>"
        "<th>model_version</th><th>Cause</th><th>Parameters</th>"
        "<th>accuracy</th><th>val_accuracy</th><th>Δval_accuracy</th>"
        "<th>loss</th><th>val_loss</th><th>Δval_loss</th>"
        "<th>Duration<br>(min)</th><th>ΔDuration<br>(min)</th>"
        "</tr></thead>"
    )
    lines.append("<tbody><tr>")

    for cell in [
        ts_end,
        model_version or "",
        cause_cell,
        params_html,
        fmt_val(acc),
        fmt_val(vacc),
        fmt_arrow_delta(pvacc, vacc),
        fmt_val(los),
        fmt_val(vlos),
        fmt_arrow_delta(plos, vlos),
        dur_tx,
        fmt_arrow_minutes(d_dur),
    ]:
        lines.append(
            "<td style='text-align:center; vertical-align:middle; "
            "word-break:break-word; max-width:100%'>" + str(cell) + "</td>"
        )

    lines.append("</tr></tbody></table>\n")

    # ---- Section 2: Performance history (last 10 runs) ----
    lines.append("## 2) Model Performance (val_accuracy)\n")

    # chart: use local .mlops/val_accuracy.svg/png if exists
    svg_path = None
    for cand in [".mlops/val_accuracy.svg", ".mlops/val_accuracy.png", "val_accuracy.svg", "val_accuracy.png"]:
        if os.path.exists(cand):
            svg_path = cand
            break
    if svg_path:
        lines.append(f"<img alt='val_accuracy trend' src='{svg_path}' style='width:100%; height:auto; display:block;'/>")
    else:
        lines.append("<em>val_accuracy chart not available.</em>")

    lines.append("")  # blank line

    lines.append(
        "<table style='width:100%; table-layout:auto; border-collapse:collapse;'>"
    )
    lines.append("<thead><tr>")
    for col in [
        f"Timestamp<br>({LOCAL_TZ.split('/')[-1]})",
        "Branch",
        "Author",
        "Cause",
        "val_accuracy",
        "Δval_accuracy",
        "Duration<br>(min)",
        "Commit",
    ]:
        lines.append(f"<th style='text-align:center'>{col}</th>")
    lines.append("</tr></thead><tbody>")

    # last 10 rows oldest -> newest
    tail = rows[-10:]
    # compute Δval_accuracy within tail
    prev_va = None
    for row in tail:
        ts_cell = row.get(ts_field, "")
        br = row.get("Branch", row.get("branch", ""))
        au = row.get("Author", row.get("author", ""))
        cause_row = row.get("cause_mlops", row.get("Cause", "None")) or "None"
        va_raw = row.get("val_accuracy", "")
        va = parse_float(va_raw)
        delta_val = va - prev_va if (va is not None and prev_va is not None) else 0.0
        prev_va = va if va is not None else prev_va
        dur_row = row.get("train_min", row.get("training_duration", ""))

        commit_id = row.get("commit_id", "")
        short = commit_id[:7] if commit_id else ""
        url = (
            f"https://github.com/{REPO}/commit/{commit_id}"
            if (REPO and commit_id)
            else ""
        )
        commit_cell_html = (
            f'<a href="{url}"><code>{html.escape(short)}</code></a>'
            if url
            else f"<code>{html.escape(short)}</code>"
        )

        cells = [
            ts_cell,
            f"<code>{html.escape(br)}</code>",
            f"<code>{html.escape(au)}</code>",
            f"<code>{html.escape(cause_row)}</code>",
            fmt_val(va_raw),
            fmt_arrow_from_value(delta_val),
            fmt_dur(dur_row),
            commit_cell_html,
        ]

        # sharp-change highlighting
        try:
            dv = float(delta_val)
        except Exception:
            dv = None
        is_sharp = abs(dv) >= SHARP_DELTA if dv is not None else False
        if is_sharp:
            cells = [f"<strong><ins>{c}</ins></strong>" for c in cells]

        lines.append("<tr>")
        for cell in cells:
            lines.append(
                "<td style='text-align:center; vertical-align:middle; "
                "word-break:break-word; max-width:100%'>" + str(cell) + "</td>"
            )
        lines.append("</tr>")

    lines.append("</tbody></table>\n")

    # ---- Section 3: Code ----
    lines.append("## 3) Code\n")
    if BRANCH:
        lines.append(f"- **Branch:** `{BRANCH}`")
    if ACTOR:
        lines.append(f"- **Author:** `{ACTOR}`")

    if SHA and REPO:
        msg = COMMIT_MSG or ""
        lines.append(
            f'- **Commit:** <a href="https://github.com/{REPO}/commit/{SHA}">'
            f"<code>{SHA[:7]}</code> — {html.escape(msg)}</a>"
        )

    # Model source – we just look at TRAINED flag for now
    if trained_flag:
        lines.append("- **Model source in APK:** Trained in this run")

    lines.append("- **Status:** Success")

    job_end_str = ts_from_epoch(os.getenv("JOB_END", "")) if os.getenv("JOB_END") else ""
    if job_end_str:
        lines.append(f"- **Job finished at:** {job_end_str}")

    # ---- Section 4: Artifacts ----
    lines.append("\n## 4) Artifacts\n")
    lines.append("| Artifact | Size |")
    lines.append("|---|---:|")

    artifact_entries = []
    for root in ["artifacts", ".mlops"]:
        if not os.path.isdir(root):
            continue
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                rel = p.as_posix()
                if rel.endswith("commitHistory.csv"):
                    continue
                sz = p.stat().st_size
                sz_kb = f"{sz/1024:.1f} KB" if sz else ""
                artifact_entries.append((rel, sz_kb))

    if artifact_entries:
        for rel, sz_kb in artifact_entries:
            lines.append(f"| [{rel}]({rel}) | {sz_kb} |")
    else:
        lines.append("| _None_ | |")

    # ---- Section 5: Commit History ----
    gist_url = os.getenv("GIST_URL", "")
    lines.append("\n## 5) Commit History\n")
    if gist_url:
        lines.append(f"- **Commit History:** [commitHistory.csv]({gist_url})")
    elif csv_path:
        rel = csv_path.replace("\\", "/")
        lines.append(f"- **Commit History:** [{os.path.basename(rel)}]({rel})")
    else:
        lines.append("- **Commit History:** `commitHistory.csv`")

    lines.append("\n_Job summary generated at run-time_\n")

    with open(SUMMARY, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Summary generated.")


if __name__ == "__main__":
    main()

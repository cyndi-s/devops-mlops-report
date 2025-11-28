#!/usr/bin/env python3
"""
Generate a 5-part Pipeline Summary for the GitHub Actions Job Summary.

Sections:
1) Model in APK: from this workflow run
2) Model Performance (val_accuracy)
3) Code
4) Artifacts
5) Commit History

Data source is a commitHistory.csv ledger produced by the mlops-plugin.
"""

import os
import sys
import io
from pathlib import Path

import pandas as pd


# ---------- Helpers ----------

def find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def ensure_timestamp_col(df: pd.DataFrame) -> str:
    """
    Ensure df has a 'Timestamp (Toronto)' column and return its name.
    We don't actually convert timezone here; we just normalize header and format.
    """
    col = None
    for cand in df.columns:
        low = str(cand).lower()
        if "timestamp (toronto" in low or "timestamp" in low:
            col = cand
            break

    if col is None:
        df["Timestamp (Toronto)"] = ""
        return "Timestamp (Toronto)"

    df["Timestamp (Toronto)"] = df[col].astype(str)
    return "Timestamp (Toronto)"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "branch":
            rename[c] = "Branch"
        elif lc == "author":
            rename[c] = "Author"
        elif lc.startswith("status"):
            rename[c] = "status"
        elif "training_duration" in lc and "(min" in lc:
            rename[c] = "train_min"
        elif "commit_id" in lc or lc == "sha":
            rename[c] = "commit_id"
    if rename:
        df = df.rename(columns=rename)
    return df


def format_params_row(row: pd.Series) -> str:
    # Preferred hyperparams
    keys = [
        "epochs",
        "opt_name",
        "opt_learning_rate",
        "monitor",
        "patience",
        "restore_best_weights",
    ]
    parts = []
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]) != "":
            parts.append(f"{k}={row[k]}")
    # fall back to any param_* columns
    for c in row.index:
        if str(c).startswith("param_") and pd.notna(row[c]) and str(row[c]) != "":
            parts.append(f"{c[6:]}={row[c]}")
    if not parts:
        return "—"
    return ", ".join(parts)


def to_md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def main() -> None:
    # ---------- Locate inputs ----------
    csv_path = find_first(
        [".mlops/commitHistory.csv", "artifacts/commitHistory.csv", "commitHistory.csv"]
    )
    chart_path = find_first(
        [
            ".mlops/val_accuracy.png",
            ".mlops/val_accuracy.svg",
            "artifacts/val_accuracy.png",
            "artifacts/val_accuracy.svg",
            "val_accuracy.png",
            "val_accuracy.svg",
        ]
    )

    out = io.StringIO()

    if not csv_path or not os.path.exists(csv_path):
        out.write("# Pipeline Summary\n\n")
        out.write("No `commitHistory.csv` found. The mlops-plugin did not log any runs yet.\n")
        sys.stdout.write(out.getvalue())
        return

    # ---------- Read CSV ----------
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        out.write("# Pipeline Summary\n\n")
        out.write(f"Failed to read `{csv_path}`: {e}\n")
        sys.stdout.write(out.getvalue())
        return

    df = normalize_columns(df)
    ts_col = ensure_timestamp_col(df)

    # Compute Δval_accuracy per Branch (if metric exists)
    if "val_accuracy" in df.columns:
        tmp = df.copy()
        tmp["val_accuracy_num"] = pd.to_numeric(tmp["val_accuracy"], errors="coerce")
        tmp.sort_values(by=["Branch", ts_col], inplace=True, ignore_index=True)
        tmp["prev"] = tmp.groupby("Branch")["val_accuracy_num"].shift(1)
        tmp["Δval_accuracy"] = (tmp["val_accuracy_num"] - tmp["prev"]).round(3)
        df["Δval_accuracy"] = tmp["Δval_accuracy"]
    else:
        df["Δval_accuracy"] = pd.NA

    # Training duration column (train_min) if not already normalized
    if "train_min" not in df.columns:
        dur_col = None
        for c in df.columns:
            if "training_duration" in str(c).lower():
                dur_col = c
                break
        if dur_col:
            df["train_min"] = df[dur_col]
        else:
            df["train_min"] = ""

    # Row for "this workflow run": last row in CSV
    latest_row = df.iloc[-1]

    # ---------- Section 1: Model in APK ----------
    sec1_cols = [
        "Timestamp (Toronto)",
        "model_version",
        "Cause",
        "Parameters",
        "accuracy",
        "val_accuracy",
        "Δval_accuracy",
        "loss",
        "val_loss",
        "Duration (min)",
    ]
    sec1_df = pd.DataFrame(columns=sec1_cols)

    params = format_params_row(latest_row)
    accuracy = latest_row.get("accuracy", "")
    val_accuracy = latest_row.get("val_accuracy", "")
    delta = latest_row.get("Δval_accuracy", "")
    loss = latest_row.get("loss", "")
    val_loss = latest_row.get("val_loss", "")
    duration = latest_row.get("train_min", "")

    sec1_df.loc[0] = [
        latest_row.get("Timestamp (Toronto)", ""),
        latest_row.get("model_version", ""),
        latest_row.get("Cause", latest_row.get("CAUSE_MLOPS", "")),
        params,
        accuracy,
        val_accuracy,
        delta,
        loss,
        val_loss,
        duration,
    ]

    # ---------- Section 2: Performance trend ----------
    if chart_path:
        chart_md = f"![val_accuracy]({chart_path})"
    else:
        chart_md = "_No chart generated._"

    cols2 = [
        "Timestamp (Toronto)",
        "Branch",
        "Author",
        "Cause",
        "val_accuracy",
        "Δval_accuracy",
        "Duration (min)",
        "Commit",
    ]
    sec2_df = pd.DataFrame(columns=cols2)

    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "").strip("/")
    base_url = f"{server}/{repo}" if repo else ""

    last10 = df.tail(10).copy()
    for _, row in last10.iterrows():
        sha_full = str(row.get("commit_id", ""))
        sha = sha_full[:7] if sha_full else ""
        if base_url and sha_full:
            commit_link = f"[{sha}]({base_url}/commit/{sha_full})"
        else:
            commit_link = sha or ""
        sec2_df.loc[len(sec2_df)] = [
            row.get("Timestamp (Toronto)", ""),
            row.get("Branch", ""),
            row.get("Author", ""),
            row.get("Cause", row.get("CAUSE_MLOPS", "")),
            row.get("val_accuracy", ""),
            row.get("Δval_accuracy", ""),
            row.get("train_min", ""),
            commit_link,
        ]

    # ---------- Section 3: Code ----------
    code_lines = []
    code_lines.append(f"- **Branch:** {latest_row.get('Branch', '')}")
    code_lines.append(f"- **Author:** {latest_row.get('Author', '')}")
    if base_url and latest_row.get("commit_id", ""):
        sha_full = str(latest_row.get("commit_id", ""))
        sha_short = sha_full[:7]
        code_lines.append(f"- **Commit:** [{sha_short}]({base_url}/commit/{sha_full})")
    else:
        code_lines.append(f"- **Commit:** {latest_row.get('commit_id', '')}")
    code_lines.append(f"- **Status:** {latest_row.get('status', '')}")
    code_lines.append(
        f"- **Trained model this run:** "
        f"{latest_row.get('trained_model', latest_row.get('TRAINED', ''))}"
    )
    code_md = "\n".join(code_lines)

    # ---------- Section 4: Artifacts ----------
    artifacts_rows = []
    for root in ["artifacts", ".mlops"]:
        if not os.path.isdir(root):
            continue
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                rel = p.as_posix()
                if rel.endswith("commitHistory.csv"):
                    continue
                size = p.stat().st_size
                artifacts_rows.append({"Artifact": rel, "Size": size})
    if artifacts_rows:
        sec4_df = pd.DataFrame(artifacts_rows)
    else:
        sec4_df = pd.DataFrame(columns=["Artifact", "Size"])

    # ---------- Section 5: Commit History ----------
    history_link = csv_path.replace("\\", "/")

    # ---------- Emit Markdown ----------
    out.write("# Pipeline Summary\n\n")

    # 1) Model in APK
    out.write("## 1) Model in APK: from this workflow run\n\n")
    out.write(to_md_table(sec1_df))
    out.write("\n\n")

    # 2) Performance
    out.write("## 2) Model Performance (val_accuracy)\n\n")
    out.write(chart_md + "\n\n")
    out.write(to_md_table(sec2_df))
    out.write("\n\n")

    # 3) Code
    out.write("## 3) Code\n\n")
    out.write(code_md + "\n\n")

    # 4) Artifacts
    out.write("## 4) Artifacts\n\n")
    if sec4_df.empty:
        out.write("_None_\n\n")
    else:
        out.write(to_md_table(sec4_df) + "\n\n")

    # 5) Commit History
    out.write("## 5) Commit History\n\n")
    out.write(f"- Commit History: [{os.path.basename(history_link)}]({history_link})\n\n")
    out.write("_Job summary generated at run-time_\n")

    sys.stdout.write(out.getvalue())


if __name__ == "__main__":
    main()

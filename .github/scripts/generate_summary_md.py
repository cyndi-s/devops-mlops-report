#!/usr/bin/env python3
import os, csv, json, textwrap
from pathlib import Path
from datetime import datetime

CSV = Path("commitHistory.csv")
SVG = Path("val_accuracy.svg")

def read_csv_tail(path, n=10):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows[-n:]

def md_table(rows):
    if not rows:
        return "_No history yet._\n"
    # Select columns to keep the table readable
    cols = [
        "commit_id","branch","status","trained_model","cause_mlops",
        "model_version","run_id","val_accuracy","training_duration\n(min)"
    ]
    # Fix fieldnames that contain newline
    fixed_rows = []
    for row in rows:
        new = dict(row)
        dur_key = "training_duration\n(min)"
        if dur_key in new:
            new["training_duration"] = new.pop(dur_key)
        fixed_rows.append(new)
    headers = ["commit_id","branch","status","trained","cause","model_ver","run_id","val_acc","train_min"]
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"]*len(headers)) + "|"]
    for r in fixed_rows:
        out.append("|" + "|".join([
            r.get("commit_id",""),
            r.get("branch",""),
            r.get("status",""),
            r.get("trained_model",""),
            r.get("cause_mlops",""),
            r.get("model_version",""),
            r.get("run_id",""),
            r.get("val_accuracy",""),
            r.get("training_duration",""),
        ]) + "|")
    return "\n".join(out) + "\n"

def link_gist_from_env():
    gist_id = os.getenv("GIST_ID","") or ""
    gist_filename = os.getenv("GIST_FILENAME","commitHistory.csv")
    if not gist_id:
        return ""
    raw = f"https://gist.github.com/{gist_id}"
    return f"[Gist CSV]({raw})"

def main():
    print("# Summary\n")
    # chart if available
    if SVG.exists():
        print("![val_accuracy](val_accuracy.svg)\n")

    rows = read_csv_tail(CSV, 10)
    print("## Latest 10 runs")
    print(md_table(rows))

    # extras
    link = link_gist_from_env()
    if link:
        print(f"**CSV ledger:** {link}")

if __name__ == "__main__":
    main()
    print("Summary generated.")

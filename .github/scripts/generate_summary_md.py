import argparse
import json
import os


FIXED_MLFLOW_MISSING_MSG_MD = """
**MLflow project not detected**

No `MLproject` file was found in this repository.

The `devops-mlops-report` expects an MLflow Project with MLflow runs
(metrics and params logged to an MLflow tracking server).

You can generate an MLproject file using the [GoMLOps](https://github.com/yorku-ease/GoMLOps) tool.

After adding `MLproject` file and `arg2pipeline/` folder, re-run this workflow.
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gist-url", required=True)
    ap.add_argument("--detect-json", required=True)
    ap.add_argument("--mlflow-json", required=True)
    ap.add_argument("--devops-json", required=True)
    args = ap.parse_args()

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(args.detect_json, "r", encoding="utf-8") as f:
        det = json.load(f)

    with open(args.mlflow_json, "r", encoding="utf-8") as f:
        ml = json.load(f)

    with open(args.devops_json, "r", encoding="utf-8") as f:
        dev = json.load(f)

    mlflow_project_detected = bool(det.get("mlflow_project_detected"))
    missing_reason = (det.get("missing_reason") or "").strip()

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("# devops-mlops-report\n\n")

        # Section 1
        f.write("## Section 1 — Latest Model\n\n")
        if not mlflow_project_detected:
            f.write(FIXED_MLFLOW_MISSING_MSG_MD + "\n\n")
            if missing_reason:
                f.write(f"_Detected issue: {missing_reason}_\n\n")
        else:
            is_trained = ml.get("is_trained", "No")
            run_id = (ml.get("run_id") or "").strip()
            reason = (ml.get("reason") or "").strip()

            f.write(f"- Training detected: **{is_trained}**\n")
            if run_id:
                f.write(f"- MLflow run id: `{run_id}`\n")
            if reason and is_trained != "Yes":
                f.write(f"- Note: {reason}\n")
            f.write("\n")

            if is_trained == "Yes":
                f.write("Placeholder (Step 2: MLflow run extraction not implemented yet)\n\n")

        # Section 2 (not gated)
        f.write("## Section 2 — Model Performance Trend\n\n")
        f.write("Placeholder (Step 2: trend SVG not implemented yet)\n\n")

        # Section 3
        f.write("## Section 3 — Code\n\n")
        f.write(f"- Branch: `{dev.get('branch', '')}`\n")
        f.write(f"- Author: `{dev.get('author', '')}`\n")

        short_sha = (dev.get("commit_sha") or "")[:7]
        commit_msg = (dev.get("commit_msg") or "").strip()
        commit_url = (dev.get("commit_url") or "").strip()

        if commit_url:
            f.write(f"- Commit: [{short_sha} — {commit_msg}]({commit_url})\n")
        else:
            f.write(f"- Commit: {short_sha} — {commit_msg}\n")


        f.write(f"- Model source in APK: {dev.get('model_source_in_apk', 'Unknown (not implemented yet)')}\n")
        f.write(f"- Status: {dev.get('status', '')}\n")
        f.write(f"- Job finished at: {dev.get('finished_at', '')}\n\n")

        # Section 4
        f.write("## Section 4 — Artifacts\n\n")
        f.write("Placeholder (Step 2)\n\n")

        # Section 5
        f.write("## Section 5 — Commit History\n\n")
        f.write(f"- Commit History: [commitHistory.csv]({gist_url})\n\n")

if __name__ == "__main__":
    main()

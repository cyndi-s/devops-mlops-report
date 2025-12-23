import argparse
import json
import os

SUMMARY = os.getenv("GITHUB_STEP_SUMMARY", "/dev/stdout")
REPO = os.getenv("GITHUB_REPOSITORY", "")
SHA = os.getenv("GITHUB_SHA", "")
BRANCH_ENV = os.getenv("GITHUB_REF_NAME", "")
ACTOR_ENV = os.getenv("GITHUB_ACTOR", "")
CSV_URL = os.getenv("COMMIT_CSV_RAW_URL", "")

# Not always present; keep as optional fallback
COMMIT_MSG_ENV = os.getenv("GITHUB_EVENT_HEAD_COMMIT_MESSAGE", "")

commit_url_env = f"https://github.com/{REPO}/commit/{SHA}" if (REPO and SHA) else ""

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

    summary_path = SUMMARY
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

        branch = BRANCH_ENV.strip()
        author = ACTOR_ENV.strip()

        commit_sha = SHA.strip()
        commit_url = commit_url_env.strip()

        commit_msg = (dev.get("commit_msg", "") or "").strip()
        if not commit_msg:
            commit_msg = (COMMIT_MSG_ENV or "").strip()

        f.write(f"- Branch: `{branch}`\n")
        f.write(f"- Author: `{author}`\n")

        if commit_url and commit_sha:
            # clickable short SHA + message together
            f.write(f"- Commit: [{commit_sha[:7]} — {commit_msg}]({commit_url})\n")
        else:
            f.write(f"- Commit: {commit_sha[:7]} — {commit_msg}\n")

        f.write(f"- Model source in APK: {dev.get('model_source_in_apk', 'Unknown (not implemented yet)')}\n")
        f.write(f"- Status: {dev.get('status', '')}\n")
        f.write(f"- Job finished at: {dev.get('finished_at', '')}\n\n")

        # Section 4
        f.write("## Section 4 — Artifacts\n\n")
        f.write("Placeholder (Step 2)\n\n")

        # Section 5
        f.write("## Section 5 — Commit History\n\n")
        # if CSV_URL:
        #     f.write(f"- **Commit History:** [commitHistory.csv]({CSV_URL})\n")
        # else:
        #     gist_url = (args.gist_url or "").strip().strip('"').strip("'").rstrip("/")
        #     gist_id = gist_url.split("/")[-1] if gist_url else ""
        #     if gist_id:
        #         f.write(f"- **Commit History:** [commitHistory.csv](https://gist.github.com/{gist_id})\n")
        #     else:
        #         f.write("- **Commit History:** commitHistory.csv\n")
        gist_url = (args.gist_url or "").strip().strip('"').strip("'").rstrip("/")
        gist_id = gist_url.split("/")[-1] if gist_url else ""
        if gist_id:
            f.write(f"- **Commit History:** [commitHistory.csv](https://gist.github.com/{gist_id})\n")
        else:
            f.write("- **Commit History:** commitHistory.csv\n")
if __name__ == "__main__":
    main()

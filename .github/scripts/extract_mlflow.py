import argparse
import json
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to report-config.yml in caller repo")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    # Load config (we won't extract metrics/params; this is only for future-proofing)
    with open(args.config, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}

    # Phase 2.3 rule: uniquely identified MLflow run => is_trained = Yes
    # Minimal, reliable signal: MLFLOW_RUN_ID provided by the workflow (from training step output).
    run_id = os.environ.get("MLFLOW_RUN_ID", "").strip()

    payload = {
        "is_trained": "Yes" if run_id else "No",
        "run_id": run_id,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

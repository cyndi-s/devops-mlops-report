import argparse
import os
import subprocess
import sys
import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="report-config.yml path")
    ap.add_argument("--caller-root", required=True, help="caller repo path")
    ap.add_argument("--cause", default="", help="Data/Script/Both/''")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    caller_root = os.path.abspath(args.caller_root)
    mlproject = os.path.join(caller_root, "MLproject")

    # Only train if MLproject exists AND cause indicates something changed
    cause = (args.cause or "").strip()
    if not os.path.exists(mlproject):
        print("Skip training: MLproject not found.")
        return 0
    if not cause:
        print("Skip training: no cause detected.")
        return 0

    # Run MLflow Project entry point "main"
    cmd = ["mlflow", "run", ".", "-e", "main"]

    # Optional: pass parameters from config if you ever want
    # (keep empty for now to use MLproject defaults)

    print("Trigger training:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=caller_root)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())

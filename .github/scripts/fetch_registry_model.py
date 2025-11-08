#!/usr/bin/env python3
import os
from pathlib import Path
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "DefaultModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
OUT_PATH = Path(os.getenv("MODEL_OUT", "model.tflite"))

client = MlflowClient()

def main():
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}' and current_stage='{MODEL_STAGE}'")
        if not versions:
            print("No model in requested stage; skip.")
            return
        mv = versions[0]
        uri = mv.source  # artifact URI
        # In practice you'd download artifact from uri; for now just print
        print(f"Selected model version {mv.version} from {MODEL_STAGE} at {uri}")
        # Placeholder: adopters implement artifact fetch according to their storage
        OUT_PATH.write_bytes(b"")  # create empty file as placeholder
        print(f"Wrote placeholder {OUT_PATH}")
    except Exception as e:
        print(f"Fetch failed: {e}")

if __name__ == "__main__":
    main()

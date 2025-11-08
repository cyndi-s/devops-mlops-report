# Test: detect_cause_mlops.sh

**Goal:** confirm cause tagging works on file changes.

### Steps
1) Modify a file under `POC-Image-Classification/data/` → commit.
2) Run CI and check the "Detect cause" step logs.
3) Repeat by changing a training script (e.g., `train*.py`) or `conda.yaml`.

### Expected
- Data change → `CAUSE_MLOPS=Data`
- Script change → `CAUSE_MLOPS=Script`
- Both changed → `CAUSE_MLOPS=Both`

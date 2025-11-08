# Test: append_commitHistory.py

**Goal:** verify a row is appended to `commitHistory.csv` and pushed to your Gist.

### Steps
1) Ensure repo variables/secrets are set: `GIST_ID`, `GIST_TOKEN`.
2) Push any commit (no training needed).
3) Open the workflow run → check the "Append to commitHistory.csv" step says “Appended …”.
4) Open your Gist → confirm a **new row** with this commit SHA exists.

### Expected
- `commitHistory.csv` contains a new row (even if training didn’t run).
- Summary tab shows a table once the CSV has at least one row.

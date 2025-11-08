# mlops-plugin

**mlops-plugin** is a lightweight GitHub Actions plugin for MLOps.  
It links each commit to its MLflow run, generates a Summary report, and records results in a Gist CSV ledger.

---

## What it does
- Collects run metrics and parameters from **MLflow / DagsHub**  
- Appends a one-line summary to a **Gist CSV ledger** (no repo churn)  
- Generates a **GitHub Actions Summary Tab** with charts and tables  
- (Optional) Uses **GoMLOps** to detect whether changes came from Data, Script, or Model updates  

---

## How to use

### 1. Add required secrets
In your repository → **Settings → Secrets → Actions**, add:
- `MLFLOW_TRACKING_URI` — your DagsHub MLflow tracking URI  
- `MLFLOW_TOKEN` — your MLflow/DagsHub access token  
- (optional) `GIST_TOKEN` — token with *gist* scope if you want to record results to a Gist  

### 2. Create a workflow
Add this to `.github/workflows/mlops.yml`:

```yaml
name: MLOps CI
on: [push, pull_request]

jobs:
  mlops:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cyndi-s/mlops-plugin/action@v1
        with:
          mlflow_tracking_uri: ${{ secrets.MLFLOW_TRACKING_URI }}
          mlflow_token: ${{ secrets.MLFLOW_TOKEN }}
          gist_id: "YOUR_GIST_ID"
          gist_filename: "commitHistory.csv"

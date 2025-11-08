# Adopters Guide

1) Create a **Gist** with a file named `commitHistory.csv` and copy its **ID**.
2) Add repo **Secrets**:  
   - `MLFLOW_TRACKING_URI` (DagsHub MLflow URL)  
   - `MLFLOW_TOKEN` (DagsHub access token)  
   - `GIST_TOKEN` (GitHub PAT with `gist` scope)
3) Add repo **Variable**: `GIST_ID` (your Gist’s ID).
4) Paste the `use-mlops-plugin.yml` workflow into `.github/workflows/` of your ML repo.
5) Push a commit.  
   - Summary shows charts/tables  
   - Gist CSV gets a new row per run

# Examples

This folder shows how another repository can call **mlops-plugin**.

- `minimal-ml-repo/.github/workflows/use-mlops-plugin.yml`  
  Copy this file into any ML repo to enable Commit→MLflow→Gist traceability.

**Prereqs for the target repo**
- GitHub Secrets: `MLFLOW_TRACKING_URI`, `MLFLOW_TOKEN`, `GIST_TOKEN`
- GitHub Variable (or Secret): `GIST_ID`

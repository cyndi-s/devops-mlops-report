# devops-mlops-report

A repository-driven **CI reporting system** that produces a unified **DevOps + MLOps summary** for every commit.

This project focuses on **commit-level observability**:
each commit generates a clear summary showing **what ran, what changed, and how performance evolved**, regardless of whether model training occurred.

---

## What This Project Does

- Generates a **commit-level summary report** in GitHub Actions
- Integrates **DevOps and MLOps workflows in parallel**
- Preserves **Commit → Run → Performance** traceability
- Maintains a persistent history (`commitHistory.csv`) for trend analysis
- Supports **optional model registry integration** (e.g., DagsHub)

The summary is always rendered:
- **with ML training**, or
- **without ML training** (DevOps-only commits)

---

## Core Ideas

- **CI-first**: Every commit triggers reporting
- **DevOps always-on, MLOps conditional**
- **CSV as reporting source of truth**
- **MLflow used only at runtime**
- **Model registry is optional**
- **Safe defaults and graceful fallbacks**

This is **not** an MLOps pipeline and **not** a GitHub Action plugin.

---

## How It Works (High Level)

1. A commit triggers a GitHub Actions workflow.
2. DevOps steps run (build, test, etc.).
3. If configured, model training runs and logs metrics via MLflow.
4. Metrics and metadata are extracted from the current run.
5. Results are appended to `commitHistory.csv`.
6. A structured summary is written to the GitHub Actions Job Summary.
7. (Optional) Models are logged to a remote registry.

---

## Configuration Overview

When a repository adopts this system, it provides a repo-level configuration file.

### Required
- **Gist CSV link** — location of `commitHistory.csv`
- **Highlight metric name** — metric used for performance trend visualization

### Optional
- **Model registry configuration** (e.g., DagsHub)

All credentials and tokens are supplied via **GitHub Secrets**.

---

## Modes of Operation

- **Lite Mode (default)**  
  CSV-based reporting only; no model registry required.

- **Full Mode (optional)**  
  Enables model versioning and retrieval via a remote registry.

---

## What This Project Is NOT

- Not a GitHub Action plugin
- Not a single-repository demo
- Not model- or framework-specific
- Not dependent on repository visibility (works for private and public repos)

---

## Project Status

This repository defines the **architecture, constraints, and reporting model**.

Implementation details and scripts are intentionally separated to keep the design clear and reusable.

For full system behavior and design rationale, see:

➡️ **`guideline.md`**

---

## License / Usage

This project is intended for research, experimentation, and CI observability use cases.


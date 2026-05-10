# Submission Cover Note

Course: MLOps (S2-25_AMLCSZG523)
Project: Heart Disease Risk Classification - End-to-End MLOps
Date: 10 May 2026

This submission provides a complete end-to-end MLOps implementation for heart disease risk prediction using the UCI Heart Disease dataset.

Included highlights:

- Automated data download and preprocessing pipeline.
- EDA workflow with class balance, histograms, and correlation heatmap outputs.
- Two baseline classifiers (Logistic Regression and Random Forest) with cross-validation and evaluation metrics.
- MLflow-based experiment tracking with logged metrics and artifacts.
- Reproducible model packaging with sklearn pipeline and serialized model artifact.
- FastAPI model serving with health, prediction, and metrics endpoints.
- Docker packaging and Kubernetes + Helm deployment manifests.
- CI workflow covering lint, tests, training, and artifact upload.
- Unit tests for preprocessing and API prediction behavior.

Validation summary:

- Lint passed in project virtual environment.
- Tests passed (2 passed).
- Training smoke execution passed with max-samples configuration.
- Docker daemon was not active during local image build attempt, so container build verification depends on local Docker runtime state.

Repository evidence map:

- Assignment and final readiness checklist: ASSIGNMENT.md
- Main setup and run instructions: README.md
- CI workflow: .github/workflows/main.yml
- Source code: src/
- Tests: tests/
- Deployment assets: deploy/k8s-manifests/, deploy/helm-chart/
- Reporting evidence guidance: screenshots/README.md
- Report artifact guidance: reports/README.md
- Final report document: final_report.docx

Local access instructions are documented in README.md for environments where a public endpoint URL is not submitted.

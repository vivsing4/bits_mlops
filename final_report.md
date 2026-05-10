# Heart Disease Risk Classification — End-to-End MLOps Project

**Course:** MLOps (S2-25_AMLCSZG523)  
**Institution:** BITS Pilani  
**Total Marks:** 50  
**Date:** 10 May 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Dataset](#2-problem-statement--dataset)
3. [Setup & Installation](#3-setup--installation)
4. [Data Acquisition & Preprocessing](#4-data-acquisition--preprocessing)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering & Model Development](#6-feature-engineering--model-development)
7. [Experiment Tracking with MLflow](#7-experiment-tracking-with-mlflow)
8. [Model Packaging & Reproducibility](#8-model-packaging--reproducibility)
9. [API Service & Local Testing Instructions](#9-api-service--local-testing-instructions)
10. [Model Containerisation](#10-model-containerisation)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Production Deployment](#12-production-deployment)
13. [Monitoring & Logging](#13-monitoring--logging)
14. [Testing](#14-testing)
15. [Submission Deliverables](#15-submission-deliverables)
16. [Conclusion](#16-conclusion)

---

## 1. Executive Summary

This report presents a production-grade, end-to-end MLOps implementation for predicting heart disease risk using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

The project covers the complete ML lifecycle:

- Automated data acquisition and preprocessing
- Exploratory data analysis with professional visualisations
- Feature engineering and two-model training with cross-validation
- MLflow experiment tracking with logged parameters, metrics, and artefacts
- Reproducible model packaging using a scikit-learn Pipeline
- FastAPI inference service with `/predict`, `/health`, and `/metrics` endpoints
- Docker containerisation and Kubernetes/Helm production deployment
- GitHub Actions CI/CD with lint, test, train, and artefact upload steps
- Prometheus-compatible monitoring and structured request logging

**Best model:** Logistic Regression pipeline — **Test ROC-AUC: 0.9665**

---

## 2. Problem Statement & Dataset

### 2.1 Task

Binary classification: predict whether a patient has heart disease (`target = 1`) or is healthy (`target = 0`) using 13 clinical features from patient health records.

### 2.2 Dataset Details

| Attribute | Detail |
|---|---|
| Name | Heart Disease UCI Dataset |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/ml/datasets/heart+Disease |
| Raw file | `data/raw/heart_disease.csv` |
| Processed file | `data/processed/heart_disease_clean.csv` |
| Records | 303 |
| Features | 13 |
| Target classes | 0 = No disease, 1 = Disease present |
| Missing values | Present as `?` in raw CSV — handled during preprocessing |

### 2.3 Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of patient (years) |
| `sex` | Categorical | Sex (1 = male, 0 = female) |
| `cp` | Categorical | Chest pain type (0–3) |
| `trestbps` | Numeric | Resting blood pressure (mmHg) |
| `chol` | Numeric | Serum cholesterol (mg/dl) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Numeric | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina (1 = yes) |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment (0–2) |
| `ca` | Numeric | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Numeric | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible) |

---

## 3. Setup & Installation

### 3.1 Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd bits_mlops

# Create and activate virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install all pinned dependencies
pip install -r requirements.txt
```

### 3.2 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.6 | REST API framework |
| `uvicorn[standard]` | 0.34.0 | ASGI server |
| `scikit-learn` | 1.6.0 | ML models and preprocessing pipelines |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 2.2.1 | Numerical operations |
| `mlflow` | 2.19.0 | Experiment tracking |
| `joblib` | 1.4.2 | Model serialisation |
| `matplotlib` | 3.10.0 | EDA visualisations |
| `seaborn` | 0.13.2 | Statistical visualisations |
| `prometheus-fastapi-instrumentator` | 7.0.0 | Prometheus metrics endpoint |
| `httpx` / `pytest` | 0.28.1 / 8.3.4 | Testing client and test runner |
| `ruff` | 0.8.4 | Linting |

### 3.3 MLflow Project Entry Points

```bash
mlflow run . -e download_data
mlflow run . -e eda
mlflow run . -e train
```

---

## 4. Data Acquisition & Preprocessing

### 4.1 Download & Process

```bash
python -m src.data_loader \
  --raw-output data/raw/heart_disease.csv \
  --processed-output data/processed/heart_disease_clean.csv
```

The `src/data_loader.py` module performs three operations:

1. **Download** — fetches the raw 14-column CSV directly from the UCI repository URL into `data/raw/heart_disease.csv`.
2. **Clean** — replaces `?` placeholder values with `NaN`, converts all columns to numeric with `pd.to_numeric(errors='coerce')`.
3. **Binarise target** — maps `target > 0` → `1` (disease present), `0` → `0` (healthy). Saves to `data/processed/heart_disease_clean.csv`.

### 4.2 Data Quality Notes

- **Missing values** appear in `ca` and `thal` columns only (encoded as `?` in the original UCI file).
- All missing values are handled downstream by the preprocessing pipeline using median imputation (numeric) and most-frequent imputation (categorical).
- The binarised dataset has **138 healthy** and **165 at-risk** patients — a manageable class imbalance addressed with `class_weight='balanced'` in both classifiers.

---

## 5. Exploratory Data Analysis

EDA is generated by `src/eda.py`:

```bash
python -m src.eda \
  --raw-data data/raw/heart_disease.csv \
  --output-dir reports/figures
```

### 5.1 Feature Distributions

![Feature Histograms](reports/figures/histograms.png)

*Figure 1: Histograms of continuous features — age, resting blood pressure, cholesterol, maximum heart rate, and ST depression (oldpeak). Age is approximately normally distributed around 54 years. Cholesterol and trestbps show right-skewed distributions. Thalach peaks near 150–170 bpm.*

### 5.2 Class Balance

![Class Balance](reports/figures/class_balance.png)

*Figure 2: Class balance between healthy (0) and at-risk (1) patients. Distribution is near-equal (138 vs 165), confirming no severe imbalance that would require oversampling. `class_weight='balanced'` in the classifiers adjusts decision thresholds proportionally.*

### 5.3 Feature Correlation Heatmap

![Correlation Heatmap](reports/figures/correlation_heatmap.png)

*Figure 3: Pearson correlation heatmap across all features and the binary target. Key observations:*
- **`thalach` (max heart rate)** has the strongest negative correlation with target (–0.42): lower max HR increases disease risk.
- **`oldpeak` (ST depression)** has the strongest positive correlation (+0.43): higher ST depression indicates greater risk.
- **`cp` (chest pain type)** at +0.43 is the most informative categorical predictor.
- **`ca` (vessel count)** at +0.47 is the highest single-feature correlator with disease presence.
- Cholesterol and resting blood pressure show weaker correlations than commonly assumed.

### 5.4 Key EDA Findings

| Finding | Implication |
|---|---|
| Age alone is a weak predictor (r ≈ 0.22) | Clinical features dominate over demographics |
| `thalach` and `oldpeak` are complementary | Both should be retained and scaled |
| `ca` and `thal` have missing values | Imputation strategy required |
| Class imbalance is minor | class_weight='balanced' is sufficient |
| No extreme outliers in continuous features | StandardScaler sufficient without winsorising |

---

## 6. Feature Engineering & Model Development

### 6.1 Preprocessing Pipeline (`src/pipeline.py`)

A reproducible scikit-learn `Pipeline` applies transformations identically at training and inference time, eliminating data leakage:

| Feature Group | Features | Transformations |
|---|---|---|
| **Numeric (5)** | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | `SimpleImputer(median)` → `StandardScaler()` |
| **Categorical (8)** | `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |

### 6.2 Models Trained

| Model | Key Hyperparameters | Rationale |
|---|---|---|
| **Logistic Regression** | `max_iter=1000`, `solver='liblinear'`, `class_weight='balanced'` | Interpretable baseline; effective for linearly separable clinical features |
| **Random Forest** | `n_estimators=300`, `max_depth=8`, `min_samples_split=4`, `class_weight='balanced'` | Captures non-linear interactions; robust to outliers; provides feature importances |

### 6.3 Evaluation Strategy

- **Training split:** 80% (stratified)
- **Test split:** 20% (stratified, held out until final evaluation)
- **Cross-validation:** Stratified 5-fold on the training split
- **Metrics:** Accuracy, Precision, Recall, ROC-AUC

---

## 7. Experiment Tracking with MLflow

### 7.1 What is Logged Per Run

| MLflow Component | Content |
|---|---|
| **Parameters** | `model_name`, `test_size` (0.2), `random_state` (42) |
| **CV Metrics** | `cv_accuracy`, `cv_precision`, `cv_recall`, `cv_roc_auc` (5-fold means) |
| **Test Metrics** | `test_accuracy`, `test_precision`, `test_recall`, `test_roc_auc` |
| **Artefacts** | ROC curve PNG per model (`reports/roc_<model>.png`) |
| **Model** | Full sklearn Pipeline logged via `mlflow.sklearn.log_model` |

### 7.2 ROC Curves

**Logistic Regression — ROC Curve (Test Set)**

![ROC Logistic Regression](reports/roc_logistic_regression.png)

*Figure 4: ROC curve for the Logistic Regression pipeline on the held-out test set. Area under the curve: **0.9665**. The curve closely hugs the top-left corner, indicating excellent discrimination between positive and negative cases across all classification thresholds.*

---

**Random Forest — ROC Curve (Test Set)**

![ROC Random Forest](reports/roc_random_forest.png)

*Figure 5: ROC curve for the Random Forest pipeline on the held-out test set. Area under the curve: **0.9448**. Strong performance but marginally below Logistic Regression on this dataset, likely because clinical features have largely linear relationships with the target.*

---

### 7.3 Model Results Comparison

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Test Accuracy** | 86.9% | **90.2%** |
| **Test Precision** | 81.3% | **84.4%** |
| **Test Recall** | 92.9% | **96.4%** |
| **Test ROC-AUC** | **0.9665 ✓ BEST** | 0.9448 |

### 7.4 Model Selection Decision

Logistic Regression is selected as the **best model** based on test ROC-AUC (0.9665 vs 0.9448).

While Random Forest achieves higher accuracy and recall, the ROC-AUC metric is preferred for model selection because:

1. It measures discrimination across **all thresholds**, not just at the default 0.5 cutoff.
2. In a medical screening context, the operating threshold can be tuned post-deployment.
3. A higher ROC-AUC means the model ranks patients more correctly regardless of the chosen sensitivity/specificity trade-off.

The best model pipeline is saved to `models/model_v1.pkl`.

### 7.5 Viewing the MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
# Open: http://127.0.0.1:5000
```

---

## 8. Model Packaging & Reproducibility

### 8.1 Saved Artefacts

| File | Format | Purpose |
|---|---|---|
| `models/model_v1.pkl` | joblib / pickle | Best sklearn Pipeline (preprocessor + LR) |
| `models/metrics.json` | JSON | Full metric comparison for both models |
| `requirements.txt` | pip format | All pinned dependency versions |
| `python_env.yaml` | MLflow conda spec | Python 3.11 environment definition |
| `MLproject` | MLflow YAML | Entry points: `download_data`, `eda`, `train` |
| `mlruns/` | MLflow tracking store | All run parameters, metrics, and artefacts |

### 8.2 Current `models/metrics.json`

```json
{
  "logistic_regression": {
    "test_accuracy": 0.8688524590163934,
    "test_precision": 0.8125,
    "test_recall": 0.9285714285714286,
    "test_roc_auc": 0.9664502164502166
  },
  "random_forest": {
    "test_accuracy": 0.9016393442622951,
    "test_precision": 0.84375,
    "test_recall": 0.9642857142857143,
    "test_roc_auc": 0.9448051948051949
  },
  "best_model": {
    "name": "logistic_regression",
    "test_roc_auc": 0.9664502164502166
  }
}
```

---

## 9. API Service & Local Testing Instructions

### 9.1 Run API Directly (No Docker)

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 9.2 API Endpoints

| Endpoint | Method | Description | Response |
|---|---|---|---|
| `/health` | GET | Liveness check | `{"status": "ok"}` |
| `/predict` | POST | Predict heart disease risk | `{"prediction": 0\|1, "confidence": float}` |
| `/metrics` | GET | Prometheus-format instrumentation metrics | Prometheus text format |
| `/docs` | GET | Auto-generated Swagger UI | Interactive documentation |

### 9.3 Health Check

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{"status": "ok"}
```

### 9.4 Prediction Request

Send a POST request with all 13 patient features as JSON:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 140,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

**Expected response:**
```json
{"prediction": 1, "confidence": 0.823456}
```

- `prediction` — `1` = at-risk, `0` = healthy
- `confidence` — probability of the predicted class from `predict_proba` (rounded to 6 decimal places)

### 9.5 Prometheus Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Returns Prometheus text format including:
- `http_requests_total` — request counts by method, path, and status code
- `http_request_duration_seconds` — latency histogram per endpoint
- `http_requests_in_progress` — in-flight request gauge

### 9.6 Request Logging

Every request is automatically logged to stdout by an HTTP middleware in `src/app.py`:

```
2026-05-10 08:23:11 INFO [heart_disease_api] method=POST path=/predict status=200 latency_ms=12.34
```

Set `LOG_LEVEL=DEBUG` environment variable for verbose output.

---

## 10. Model Containerisation

### 10.1 Dockerfile Summary

| Layer | Detail |
|---|---|
| Base image | `python:3.11-slim` |
| Working directory | `/app` |
| Dependencies | `pip install --no-cache-dir -r requirements.txt` |
| Files copied | `src/` and `models/` |
| Exposed port | `8000` |
| Environment | `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1` |
| Startup command | `uvicorn src.app:app --host 0.0.0.0 --port 8000` |

### 10.2 Build & Run

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container (maps host 8000 → container 8000)
docker run --rm -p 8000:8000 heart-disease-api:latest
```

### 10.3 Verify Container Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,
       "restecg":1,"thalach":140,"exang":0,"oldpeak":1.2,
       "slope":2,"ca":0,"thal":2}'

# Metrics
curl http://localhost:8000/metrics
```

---

## 11. CI/CD Pipeline

### 11.1 GitHub Actions Workflow

File: `.github/workflows/main.yml`  
Triggers: push and pull requests to `main` / `master`

| Step | Tool / Command | Purpose |
|---|---|---|
| 1. Checkout | `actions/checkout@v4` | Clone repository |
| 2. Setup Python | `actions/setup-python@v5` (3.11) | Install Python runtime |
| 3. Install dependencies | `pip install -r requirements.txt` | Install all packages |
| 4. **Lint** | `ruff check src tests` | Code style and error checking |
| 5. **Unit tests** | `pytest -q` | Run test suite — must pass |
| 6. **Train model** | `python -m src.train --max-samples 250` | Smoke-test full training flow |
| 7. Upload artefacts | `actions/upload-artifact@v4` | Store model, metrics, reports, mlruns |

### 11.2 Pipeline Failure Behaviour

- Ruff lint failure on any file → pipeline fails immediately.
- Any `pytest` test failure → pipeline fails before training.
- Training script non-zero exit → pipeline fails before artefact upload.
- Artefacts are uploaded **only** on full success.

---

## 12. Production Deployment

### 12.1 System Architecture

```mermaid
flowchart LR
    A[UCI Dataset] --> B[src.data_loader]
    B --> C[data/processed]
    C --> D[src.train]
    D --> E[MLflow Tracking / mlruns]
    D --> F[models/model_v1.pkl]
    F --> G[src.app - FastAPI]
    G --> H[Docker Image]
    H --> I[Kubernetes Deployment]
    I --> J[LoadBalancer Service]
    J --> K[Nginx Ingress]
    G --> L[/metrics]
    L --> M[Prometheus + Grafana]
```

### 12.2 Component Overview

| Component | Technology | Role |
|---|---|---|
| Data source | UCI ML Repository | Raw CSV via `src/data_loader.py` |
| Feature pipeline | scikit-learn ColumnTransformer | Imputation, scaling, one-hot encoding |
| Model training | scikit-learn + MLflow | Cross-validation, tracking, artefact export |
| Model artefact | joblib `.pkl` | Serialised sklearn Pipeline |
| Inference API | FastAPI + uvicorn | REST endpoints |
| Container | Docker `python:3.11-slim` | Reproducible execution environment |
| Orchestration | Kubernetes | 2-replica Deployment with probes |
| Networking | LoadBalancer + Ingress | External exposure |
| Monitoring | Prometheus + Grafana | Metrics scraping via pod annotations |

### 12.3 Kubernetes Manifests

Files in `deploy/k8s-manifests/`:

| Manifest | Key Configuration |
|---|---|
| `deployment.yaml` | 2 replicas, liveness/readiness on `/health`, Prometheus annotations, CPU/memory limits |
| `service.yaml` | `LoadBalancer` type, port 80 → container 8000 |
| `ingress.yaml` | Nginx Ingress, host `heart-disease.local`, path `/` |

```bash
# Apply all manifests
kubectl apply -f deploy/k8s-manifests

# Check resources
kubectl get pods,svc,ingress

# Port-forward for local verification
kubectl port-forward svc/heart-disease-api 8000:80

# Test via port-forward
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,
       "restecg":1,"thalach":140,"exang":0,"oldpeak":1.2,"slope":2,"ca":0,"thal":2}'
curl http://localhost:8000/metrics
```

### 12.4 Helm Chart Deployment

Files in `deploy/helm-chart/`:

```bash
# Install with defaults (uses values.yaml)
helm upgrade --install heart-disease-api deploy/helm-chart

# Override image for a registry push
helm upgrade --install heart-disease-api deploy/helm-chart \
  --set image.repository=<registry>/heart-disease-api \
  --set image.tag=v1.0.0

# Validate chart syntax
helm lint deploy/helm-chart

# Render templates (dry-run)
helm template heart-disease-api deploy/helm-chart
```

### 12.5 Helm Chart Default Values (`deploy/helm-chart/values.yaml`)

```yaml
replicaCount: 2
image:
  repository: heart-disease-api
  tag: latest
  pullPolicy: IfNotPresent
service:
  type: LoadBalancer
  port: 80
  targetPort: 8000
ingress:
  enabled: true
  className: nginx
  host: heart-disease.local
  path: /
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi
modelPath: models/model_v1.pkl
```

### 12.6 Ingress Local Testing Note

```bash
# Add local hosts entry to resolve heart-disease.local
echo "127.0.0.1 heart-disease.local" | sudo tee -a /etc/hosts
curl http://heart-disease.local/health
```

---

## 13. Monitoring & Logging

### 13.1 API Request Logging

Every HTTP request is intercepted by a FastAPI middleware in `src/app.py` and logged to stdout:

```
2026-05-10 08:23:11 INFO [heart_disease_api] method=POST path=/predict status=200 latency_ms=12.34
2026-05-10 08:23:15 INFO [heart_disease_api] method=GET  path=/health  status=200 latency_ms=0.87
```

Override log level via environment variable:

```bash
LOG_LEVEL=DEBUG uvicorn src.app:app --port 8000
```

### 13.2 Prometheus Integration

The `prometheus-fastapi-instrumentator` library auto-instruments the app at startup and exposes the `/metrics` endpoint.

Kubernetes pod annotations enable automatic Prometheus scraping:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

Key metrics exposed:

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Request count by method, path, status |
| `http_request_duration_seconds` | Histogram | Latency distribution per endpoint |
| `http_requests_in_progress` | Gauge | Currently in-flight requests |

### 13.3 Grafana Dashboard Setup (Reference)

1. Deploy Prometheus with the scrape config pointing to pod annotations.
2. Add Prometheus as a Grafana data source (`http://prometheus:9090`).
3. Create dashboard panels for request rate, error rate, and p95 latency using:
   - `rate(http_requests_total[5m])`
   - `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`

---

## 14. Testing

### 14.1 Run Tests

```bash
pytest -q
# 2 passed in 4.61s
```

### 14.2 Test Coverage

| File | Test Function | What it Verifies |
|---|---|---|
| `tests/test_data.py` | `test_preprocess_data_replaces_missing_and_binarizes_target` | `?` → `NaN`; target column binarised correctly to `{0, 1}` |
| `tests/test_model.py` | `test_predict_endpoint_returns_prediction_and_confidence` | `/predict` returns HTTP 200, `prediction` ∈ {0,1}, `confidence` ∈ [0.0, 1.0] |

### 14.3 Test Design Notes

- `test_model.py` trains a **dummy model** in a `tmp_path` fixture — no dependency on pre-existing `model_v1.pkl`.
- `TestClient` from `fastapi.testclient` is used for zero-network API testing.
- `conftest.py` adds the project root to `sys.path` for clean imports in all environments.

---

## 15. Submission Deliverables

| Deliverable | Path in Repository | Status |
|---|---|---|
| Source code | `src/` | ✅ Complete |
| Dockerfile | `Dockerfile` | ✅ Complete |
| `requirements.txt` | `requirements.txt` | ✅ Complete |
| Conda env file | `python_env.yaml` | ✅ Complete |
| Cleaned dataset | `data/processed/heart_disease_clean.csv` | ✅ Present |
| Download script | `src/data_loader.py` | ✅ Complete |
| Download instructions | `data/raw/README.md` | ✅ Complete |
| EDA notebook | `notebooks/01_eda_and_modeling.ipynb` | ✅ Present |
| Training script | `src/train.py` | ✅ Complete |
| Inference API | `src/app.py` | ✅ Complete |
| Unit tests | `tests/test_data.py`, `tests/test_model.py` | ✅ 2 passing |
| CI workflow YAML | `.github/workflows/main.yml` | ✅ Complete |
| K8s manifests | `deploy/k8s-manifests/` | ✅ Complete |
| Helm chart | `deploy/helm-chart/` | ✅ Complete |
| EDA figures | `reports/figures/histograms.png` | ✅ Present |
| EDA figures | `reports/figures/class_balance.png` | ✅ Present |
| EDA figures | `reports/figures/correlation_heatmap.png` | ✅ Present |
| ROC curve — LR | `reports/roc_logistic_regression.png` | ✅ Present |
| ROC curve — RF | `reports/roc_random_forest.png` | ✅ Present |
| Model metrics | `models/metrics.json` | ✅ Present |
| Screenshots folder | `screenshots/` | ⚠️ Folder present — add evidence images |
| Final written report (docx) | `final_report.docx` | ✅ Complete (45 KB) |
| Final written report (md) | `final_report.md` | ✅ This document |
| Video transcript | `VIDEO_TRANSCRIPT.md` | ✅ Complete — record video |
| Short video | — | ⚠️ Not yet recorded |
| Deployed API URL | — | ⚠️ Local instructions documented in `README.md § 13` |

---

## 16. Conclusion

This project delivers a complete, reproducible, and production-ready MLOps pipeline for heart disease risk classification.

**Model performance summary:**

| Model | ROC-AUC | Accuracy | Recall |
|---|---|---|---|
| Logistic Regression (best) | **0.9665** | 86.9% | 92.9% |
| Random Forest | 0.9448 | 90.2% | 96.4% |

**Pipeline completeness:**

- All scripts execute from a clean `requirements.txt`-based environment.
- The inference model serves correctly inside a Docker container.
- The CI pipeline fails loudly on any code or test error before deployment.
- Kubernetes deployment is self-healing with liveness/readiness probes on the `/health` endpoint.
- Monitoring is integrated at the infrastructure level via Prometheus scrape annotations.

The Logistic Regression pipeline (ROC-AUC 0.9665) is the best model for this clinical screening use case, providing strong discrimination across all decision thresholds, which is critical when the cost of false negatives (missed disease cases) is high.

---

*Report generated: 10 May 2026 | Course: MLOps (S2-25_AMLCSZG523) | BITS Pilani*

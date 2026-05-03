# Heart Disease Risk Classification - End-to-End MLOps Project

This project implements the complete assignment workflow for heart disease risk prediction using the UCI Heart Disease dataset. It includes:

- Automated data acquisition and preprocessing
- EDA plotting (histograms, class balance, correlation heatmap)
- Two classification models (Logistic Regression, Random Forest)
- Cross-validation and test metrics (accuracy, precision, recall, ROC-AUC)
- MLflow experiment tracking (params, metrics, artifacts)
- Reproducible model packaging with sklearn pipeline + joblib artifact
- FastAPI inference service with `/predict`, `/health`, `/metrics`
- Dockerized deployment and Kubernetes/Helm manifests
- CI workflow for linting, tests, training, and artifacts

## 1. Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data Acquisition and Processing

Download and preprocess from UCI:

```bash
python -m src.data_loader \
	--raw-output data/raw/heart_disease.csv \
	--processed-output data/processed/heart_disease_clean.csv
```

## 3. EDA

Generate EDA plots:

```bash
python -m src.eda --raw-data data/raw/heart_disease.csv --output-dir reports/figures
```

Outputs:

- `reports/figures/histograms.png`
- `reports/figures/class_balance.png`
- `reports/figures/correlation_heatmap.png`

## 4. Train and Track Experiments

```bash
python -m src.train
```

Key outputs:

- `models/model_v1.pkl` (best model pipeline)
- `models/metrics.json` (model comparison summary)
- `mlruns/` (MLflow tracking artifacts)

Open MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## 5. Run API Locally

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

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

Prometheus metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

## 6. Docker

Build image:

```bash
docker build -t heart-disease-api:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 heart-disease-api:latest
```

## 7. CI/CD

GitHub Actions workflow in `.github/workflows/main.yml` performs:

- Ruff linting
- Pytest unit tests
- End-to-end training run
- Upload model/tracking artifacts

## 8. Kubernetes Deployment

Static manifests:

```bash
kubectl apply -f deploy/k8s-manifests
```

Helm deployment:

```bash
helm upgrade --install heart-disease-api deploy/helm-chart
```

## 9. Architecture

```mermaid
flowchart LR
		A[UCI Dataset] --> B[src.data_loader]
		B --> C[data/processed]
		C --> D[src.train]
		D --> E[MLflow Tracking]
		D --> F[models/model_v1.pkl]
		F --> G[src.app FastAPI]
		G --> H[Docker Image]
		H --> I[Kubernetes Service/Ingress]
		G --> J[/metrics -> "Prometheus/Grafana"]
```

## 10. Tests

```bash
pytest -q
```

## 11. MLflow Project

MLflow commands defined in `MLproject`:

- `download_data`
- `eda`
- `train`


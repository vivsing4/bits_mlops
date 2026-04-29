# Heart Disease Risk Classification: End-to-End MLOps Pipeline

## Project Overview
This repository contains a production-grade machine learning solution designed to predict the risk of heart disease using the UCI Heart Disease Dataset. The project demonstrates a complete MLOps lifecycle, including automated data acquisition, experiment tracking, continuous integration/continuous deployment (CI/CD), and containerized cloud deployment.

## Key Features
* **Data Pipeline:** Automated scripts for data acquisition and cleaning of the UCI Heart Disease CSV dataset.
* **Experiment Tracking:** Integration with **MLflow** to track hyperparameters, model versions, and performance metrics (Accuracy, Precision, Recall, ROC-AUC).
* **CI/CD Pipeline:** A **GitHub Actions** workflow that automates linting, unit testing with Pytest, and model training upon every code push.
* **Containerization:** A **FastAPI** application packaged within a **Docker** container for scalable model serving.
* **Orchestration:** Deployment manifests and **Helm charts** for deploying the API to Kubernetes (Minikube/Cloud).
* **Monitoring:** Integrated logging and monitoring setup using **Prometheus and Grafana** to track API health and request metrics.

## Repository Structure
* `src/`: Core Python scripts for data processing, training, and inference.
* `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and model prototyping.
* `tests/`: Unit tests for validating data integrity and model logic.
* `deploy/`: Kubernetes manifests and Helm charts for production deployment.
* `.github/workflows/`: YAML configurations for the CI/CD pipeline.
* `Dockerfile`: Configuration for building the model-serving container.

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD Pipeline (Linting, Testing, Training) [cite: 34, 65]
├── data/
│   ├── raw/                      # Original dataset (Heart Disease UCI) [cite: 8, 63]
│   └── processed/                # Cleaned/Encoded data for training [cite: 15]
├── deploy/
│   ├── k8s-manifests/            # Kubernetes Deployment & Service YAMLs [cite: 42, 66]
│   └── helm-chart/               # Helm charts for scalable deployment [cite: 43, 66]
├── models/
│   └── model_v1.pkl              # Saved model artifact (MLflow/Pickle) [cite: 29]
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # EDA and initial model experiments [cite: 13, 64]
├── src/                          # Core application logic [cite: 64]
│   ├── __init__.py
│   ├── data_loader.py            # Script to download/clean data [cite: 15, 63]
│   ├── train.py                  # Training script with MLflow logging [cite: 19, 23]
│   ├── pipeline.py               # Preprocessing/transformers logic 
│   └── app.py                    # FastAPI/Flask API entry point [cite: 38]
├── tests/                        # Automated unit tests [cite: 33, 64]
│   ├── test_data.py              # Tests for data processing logic [cite: 33]
│   └── test_model.py             # Tests for model inference and API [cite: 33]
├── .dockerignore                 # Files to exclude from Docker build
├── .gitignore                    # Files to exclude from Git (data, venv, pyc)
├── Dockerfile                    # Containerization instructions [cite: 38, 62]
├── MLproject                     # MLflow project definition (Optional) [cite: 23]
├── README.md                     # Project documentation and setup [cite: 48, 54]
└── requirements.txt              # Project dependencies [cite: 29, 62, 72]
```

## Getting Started
1. **Clone the Repo:** `git clone <repo-url>`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Locally:** Follow the instructions in the `docs/` folder to build the Docker image and run the API.

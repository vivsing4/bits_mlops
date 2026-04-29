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

## Getting Started
1. **Clone the Repo:** `git clone <repo-url>`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Locally:** Follow the instructions in the `docs/` folder to build the Docker image and run the API.

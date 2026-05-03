"""Model training script with MLflow tracking and artifact export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	roc_auc_score,
	RocCurveDisplay,
)
from sklearn.model_selection import cross_validate, train_test_split

from src.data_loader import download_dataset, prepare_training_data
from src.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_models, build_training_pipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train heart disease classifiers")
	parser.add_argument("--raw-data", default="data/raw/heart_disease.csv")
	parser.add_argument("--processed-data", default="data/processed/heart_disease_clean.csv")
	parser.add_argument("--model-output", default="models/model_v1.pkl")
	parser.add_argument("--metrics-output", default="models/metrics.json")
	parser.add_argument("--mlflow-uri", default="file:./mlruns")
	parser.add_argument("--experiment-name", default="heart-disease-risk")
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Optional row cap for faster CI runs.",
	)
	return parser.parse_args()


def _log_roc_plot(y_test: pd.Series, y_proba: list[float], path: Path) -> None:
	roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
	roc_display.ax_.set_title("ROC Curve")
	path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(path)
	plt.close()


def main() -> None:
	args = parse_args()

	raw_path = Path(args.raw_data)
	processed_path = Path(args.processed_data)
	model_output = Path(args.model_output)
	metrics_output = Path(args.metrics_output)
	model_output.parent.mkdir(parents=True, exist_ok=True)
	metrics_output.parent.mkdir(parents=True, exist_ok=True)

	if not raw_path.exists():
		download_dataset(raw_path)

	df = prepare_training_data(raw_path, processed_path)
	if args.max_samples is not None:
		df = df.sample(min(args.max_samples, len(df)), random_state=args.random_state)

	feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
	X = df[feature_cols]
	y = df["target"]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=y,
	)

	mlflow.set_tracking_uri(args.mlflow_uri)
	mlflow.set_experiment(args.experiment_name)

	best_model = None
	best_model_name = ""
	best_score = -1.0
	summary: dict[str, dict[str, float]] = {}

	for model_name, model in build_models().items():
		pipeline = build_training_pipeline(model)
		cv_results = cross_validate(
			pipeline,
			X_train,
			y_train,
			cv=5,
			scoring=["accuracy", "precision", "recall", "roc_auc"],
			n_jobs=-1,
		)

		with mlflow.start_run(run_name=model_name):
			mlflow.log_params(
				{
					"model_name": model_name,
					"test_size": args.test_size,
					"random_state": args.random_state,
				}
			)

			for metric_name in ["accuracy", "precision", "recall", "roc_auc"]:
				cv_mean = float(cv_results[f"test_{metric_name}"].mean())
				mlflow.log_metric(f"cv_{metric_name}", cv_mean)

			pipeline.fit(X_train, y_train)
			preds = pipeline.predict(X_test)
			proba = pipeline.predict_proba(X_test)[:, 1]

			metrics = {
				"test_accuracy": float(accuracy_score(y_test, preds)),
				"test_precision": float(precision_score(y_test, preds)),
				"test_recall": float(recall_score(y_test, preds)),
				"test_roc_auc": float(roc_auc_score(y_test, proba)),
			}
			mlflow.log_metrics(metrics)

			roc_plot_path = Path("reports") / f"roc_{model_name}.png"
			_log_roc_plot(y_test, proba, roc_plot_path)
			mlflow.log_artifact(str(roc_plot_path))
			mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{model_name}")

			summary[model_name] = metrics
			if metrics["test_roc_auc"] > best_score:
				best_score = metrics["test_roc_auc"]
				best_model = pipeline
				best_model_name = model_name

	if best_model is None:
		raise RuntimeError("No model was trained successfully.")

	summary["best_model"] = {
		"name": best_model_name,
		"test_roc_auc": best_score,
	}

	joblib.dump(best_model, model_output)
	with metrics_output.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Saved model to: {model_output}")
	print(f"Saved metrics to: {metrics_output}")
	print(f"Best model: {best_model_name} (ROC-AUC={best_score:.4f})")


if __name__ == "__main__":
	main()


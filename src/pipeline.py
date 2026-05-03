"""Reusable preprocessing and modeling pipeline builders."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor() -> ColumnTransformer:
	"""Create a column-wise preprocessing transformer."""
	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	return ColumnTransformer(
		transformers=[
			("num", numeric_transformer, NUMERIC_FEATURES),
			("cat", categorical_transformer, CATEGORICAL_FEATURES),
		]
	)


def build_models() -> dict[str, object]:
	"""Return supported baseline models for comparison."""
	return {
		"logistic_regression": LogisticRegression(
			max_iter=1000,
			class_weight="balanced",
			solver="liblinear",
			random_state=42,
		),
		"random_forest": RandomForestClassifier(
			n_estimators=300,
			max_depth=8,
			min_samples_split=4,
			class_weight="balanced",
			random_state=42,
		),
	}


def build_training_pipeline(model: object) -> Pipeline:
	"""Attach the preprocessor and model into a single sklearn pipeline."""
	return Pipeline(
		steps=[
			("preprocessor", build_preprocessor()),
			("model", model),
		]
	)


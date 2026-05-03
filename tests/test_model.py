from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.app import create_app
from src.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_training_pipeline


def _train_dummy_model(tmp_path: Path) -> Path:
	rng = np.random.default_rng(42)
	n = 80
	frame = pd.DataFrame(
		{
			"age": rng.integers(30, 80, size=n),
			"trestbps": rng.integers(90, 180, size=n),
			"chol": rng.integers(150, 330, size=n),
			"thalach": rng.integers(70, 200, size=n),
			"oldpeak": rng.uniform(0, 5, size=n),
			"sex": rng.integers(0, 2, size=n),
			"cp": rng.integers(0, 4, size=n),
			"fbs": rng.integers(0, 2, size=n),
			"restecg": rng.integers(0, 3, size=n),
			"exang": rng.integers(0, 2, size=n),
			"slope": rng.integers(0, 3, size=n),
			"ca": rng.integers(0, 4, size=n),
			"thal": rng.integers(0, 4, size=n),
		}
	)
	y = rng.integers(0, 2, size=n)

	from sklearn.linear_model import LogisticRegression

	model = build_training_pipeline(LogisticRegression(max_iter=300, solver="liblinear"))
	model.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y)

	model_path = tmp_path / "model.pkl"
	joblib.dump(model, model_path)
	return model_path


def test_predict_endpoint_returns_prediction_and_confidence(tmp_path: Path) -> None:
	model_path = _train_dummy_model(tmp_path)
	app = create_app(model_path=model_path)
	client = TestClient(app)

	payload = {
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
		"thal": 2,
	}

	response = client.post("/predict", json=payload)
	body = response.json()

	assert response.status_code == 200
	assert "prediction" in body
	assert "confidence" in body
	assert body["prediction"] in [0, 1]
	assert 0.0 <= body["confidence"] <= 1.0


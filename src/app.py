"""FastAPI inference service for heart disease prediction."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger("heart_disease_api")
logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


class HeartDiseaseFeatures(BaseModel):
	age: float
	sex: int
	cp: int
	trestbps: float
	chol: float
	fbs: int
	restecg: int
	thalach: float
	exang: int
	oldpeak: float
	slope: int
	ca: float
	thal: float


def _load_model(model_path: str | Path):
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Model file not found at {path}")
	return joblib.load(path)


def create_app(model_path: str | Path | None = None, allow_missing_model: bool = False) -> FastAPI:
	app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
	resolved_path = model_path or os.getenv("MODEL_PATH", "models/model_v1.pkl")
	model: Any | None = None
	try:
		model = _load_model(resolved_path)
	except (FileNotFoundError, EOFError) as exc:
		if not allow_missing_model:
			raise
		logger.warning("Model unavailable at startup: %s", exc)

	@app.middleware("http")
	async def log_requests(request: Request, call_next):
		start = time.perf_counter()
		response = await call_next(request)
		latency_ms = (time.perf_counter() - start) * 1000
		logger.info(
			"method=%s path=%s status=%s latency_ms=%.2f",
			request.method,
			request.url.path,
			response.status_code,
			latency_ms,
		)
		return response

	@app.get("/health")
	def health() -> dict[str, str]:
		return {"status": "ok"}

	@app.post("/predict")
	def predict(payload: HeartDiseaseFeatures) -> dict[str, float | int]:
		if model is None:
			raise HTTPException(status_code=503, detail="Model is not loaded yet")
		try:
			features = pd.DataFrame([payload.model_dump()])
			prediction = int(model.predict(features)[0])
			confidence = float(model.predict_proba(features)[0][prediction])
			return {
				"prediction": prediction,
				"confidence": round(confidence, 6),
			}
		except Exception as exc:  # pragma: no cover - defensive boundary for API robustness.
			logger.exception("Prediction failed")
			raise HTTPException(status_code=500, detail=str(exc)) from exc

	Instrumentator().instrument(app).expose(app, endpoint="/metrics")
	return app


app = create_app(allow_missing_model=True)


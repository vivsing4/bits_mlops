"""Data acquisition and preparation utilities for the UCI Heart Disease dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_URL = (
	"https://archive.ics.uci.edu/ml/machine-learning-databases/"
	"heart-disease/processed.cleveland.data"
)

HEART_COLUMNS = [
	"age",
	"sex",
	"cp",
	"trestbps",
	"chol",
	"fbs",
	"restecg",
	"thalach",
	"exang",
	"oldpeak",
	"slope",
	"ca",
	"thal",
	"target",
]


def download_dataset(output_path: str | Path) -> Path:
	"""Download the raw UCI dataset to a local CSV path."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(DATA_URL, header=None, names=HEART_COLUMNS)
	df.to_csv(output, index=False)
	return output


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
	"""Load raw CSV with standardized columns and placeholder missing values."""
	df = pd.read_csv(input_path)
	if list(df.columns) != HEART_COLUMNS:
		df.columns = HEART_COLUMNS
	return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Convert types, handle missing values markers, and binarize target."""
	clean_df = df.copy()
	clean_df = clean_df.replace("?", pd.NA)
	for col in HEART_COLUMNS:
		clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

	clean_df["target"] = (clean_df["target"] > 0).astype(int)
	return clean_df


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
	"""Persist processed data to disk."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output, index=False)
	return output


def prepare_training_data(raw_path: str | Path, processed_path: str | Path) -> pd.DataFrame:
	"""Load raw data, preprocess it, and write processed data."""
	raw_df = load_raw_data(raw_path)
	processed_df = preprocess_data(raw_df)
	save_processed_data(processed_df, processed_path)
	return processed_df


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download and preprocess heart disease data")
	parser.add_argument(
		"--raw-output",
		default="data/raw/heart_disease.csv",
		help="Path where raw downloaded CSV will be saved.",
	)
	parser.add_argument(
		"--processed-output",
		default="data/processed/heart_disease_clean.csv",
		help="Path where cleaned CSV will be saved.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	raw_path = download_dataset(args.raw_output)
	df = prepare_training_data(raw_path, args.processed_output)
	print(f"Raw data saved to: {raw_path}")
	print(f"Processed data shape: {df.shape}")
	print(f"Processed data saved to: {args.processed_output}")


if __name__ == "__main__":
	main()


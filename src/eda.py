"""Generate EDA visualizations for the heart disease dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import HEART_COLUMNS, load_raw_data, preprocess_data

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and save plots")
    parser.add_argument("--raw-data", default="data/raw/heart_disease.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(args.raw_data)
    if list(df.columns) != HEART_COLUMNS:
        df.columns = HEART_COLUMNS
    clean_df = preprocess_data(df)

    plt.figure(figsize=(10, 6))
    clean_df[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png")
    plt.close("all")

    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=clean_df)
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    corr = clean_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    print(f"Saved EDA plots to: {output_dir}")


if __name__ == "__main__":
    main()

import pandas as pd

from src.data_loader import HEART_COLUMNS, preprocess_data


def test_preprocess_data_replaces_missing_and_binarizes_target() -> None:
	df = pd.DataFrame(
		[
			[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 2],
			[67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 0],
		],
		columns=HEART_COLUMNS,
	)

	processed = preprocess_data(df)

	assert processed["ca"].isna().sum() == 1
	assert set(processed["target"].unique()) == {0, 1}


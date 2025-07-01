"""Unit tests for drop_rows_with_na_features and remove_na_rows functions."""

import pandas as pd
import numpy as np
import tempfile
import os
from jump_profiling_recipe.correct.corrections import (
    drop_rows_with_na_features,
    remove_na_rows,
)


def test_drops_rows_with_any_na():
    """Test default behavior drops rows with any NA values."""
    df = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02", "A03"],
            "Feature_1": [1.0, np.nan, 3.0],
            "Feature_2": [4.0, 5.0, 6.0],
        }
    )
    result = drop_rows_with_na_features(df)
    assert len(result) == 2
    assert "A02" not in result["Metadata_Well"].values


def test_threshold_parameter_works():
    """Test na_threshold parameter controls which rows are dropped."""
    df = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02"],
            "Feature_1": [1.0, np.nan],
            "Feature_2": [np.nan, np.nan],
            "Feature_3": [3.0, 4.0],
        }
    )
    # A02 has 2/3 = 66% NA, should be dropped with 0.5 threshold
    result = drop_rows_with_na_features(df, na_threshold=0.5)
    assert len(result) == 1
    assert result["Metadata_Well"].iloc[0] == "A01"


def test_file_wrapper_works():
    """Test remove_na_rows reads and writes files correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        df = pd.DataFrame({"Metadata_Well": ["A01", "A02"], "Feature_1": [1.0, np.nan]})
        input_path = os.path.join(tmpdir, "input.parquet")
        output_path = os.path.join(tmpdir, "output.parquet")
        df.to_parquet(input_path)

        # Run function
        remove_na_rows(input_path, output_path, na_threshold=0.0)

        # Check result
        result = pd.read_parquet(output_path)
        assert len(result) == 1
        assert result["Metadata_Well"].iloc[0] == "A01"

#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from jump_profiling_recipe.cli.converter import convert_command, read_mandatory_features

# Ensure the repository root is on PYTHONPATH
repo_root = Path(__file__).parent.parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# Set up base test fixtures directory
BASE_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def input_file_path():
    """Path to the test input file."""
    return Path(
        "tests/fixtures/inputs/profiles/source_4/workspace/profiles/2021_04_26_Batch1/BR00117037/BR00117037.parquet"
    )


@pytest.fixture
def mandatory_features_path():
    """Path to the mandatory features file."""
    return Path("tests/fixtures/inputs/metadata/test_mandatory_feature_columns.txt")


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def temp_file_list(input_file_path, tmp_path):
    """Create a temporary file with the input file path."""
    file_list_path = tmp_path / "file_list.txt"
    with open(file_list_path, "w") as f:
        f.write(str(input_file_path.absolute()))
    return file_list_path


def test_convert_command(
    input_file_path, mandatory_features_path, temp_output_dir, temp_file_list
):
    """Test the convert command with a real input file."""
    # Skip test if input file doesn't exist
    if not input_file_path.exists():
        pytest.skip(f"Input file not found: {input_file_path}")

    # Skip test if mandatory features file doesn't exist
    if not mandatory_features_path.exists():
        pytest.skip(f"Mandatory features file not found: {mandatory_features_path}")

    # Run the convert command
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(temp_file_list),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "source_4",
            "--mandatory-features-file",
            str(mandatory_features_path),
            "--verbose",
        ],
    )

    # Check that the command ran successfully
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

    # Determine expected output path
    expected_output_path = (
        temp_output_dir / "2021_04_26_Batch1" / "BR00117037" / "BR00117037.parquet"
    )

    # Check that the output file exists
    assert expected_output_path.exists(), (
        f"Output file not created: {expected_output_path}"
    )

    # Read the original and converted files
    original_df = pd.read_parquet(input_file_path)
    converted_df = pd.read_parquet(expected_output_path)

    # Load mandatory features
    mandatory_features = read_mandatory_features(mandatory_features_path)

    # Verify that the output file has the expected structure
    # 1. Should have Metadata_Source column with value "source_4"
    assert "Metadata_Source" in converted_df.columns
    assert all(converted_df["Metadata_Source"] == "source_4")

    # 2. Should have Metadata_Plate and Metadata_Well from original file
    assert "Metadata_Plate" in converted_df.columns
    assert "Metadata_Well" in converted_df.columns
    assert all(converted_df["Metadata_Plate"] == original_df["Metadata_Plate"])
    assert all(converted_df["Metadata_Well"] == original_df["Metadata_Well"])

    # 3. Should only have mandatory features
    feature_columns = [
        col for col in converted_df.columns if not col.startswith("Metadata_")
    ]
    assert set(feature_columns).issubset(mandatory_features)

    # 4. All features in the output should match the original values
    for feature in feature_columns:
        assert feature in original_df.columns
        pd.testing.assert_series_equal(
            converted_df[feature], original_df[feature], check_names=False
        )

    # 5. Verify we have all available mandatory features
    available_mandatory_features = [
        f for f in mandatory_features if f in original_df.columns
    ]
    assert set(feature_columns) == set(available_mandatory_features)


def test_continue_on_error(input_file_path, temp_output_dir, tmp_path):
    """Test the --continue-on-error option."""
    # Create a file list with a non-existent file followed by a valid file
    file_list_path = tmp_path / "mixed_file_list.txt"
    with open(file_list_path, "w") as f:
        f.write(f"{tmp_path}/nonexistent.csv\n")
        f.write(str(input_file_path.absolute()))

    # Skip test if input file doesn't exist
    if not input_file_path.exists():
        pytest.skip(f"Input file not found: {input_file_path}")

    # Run the convert command with --continue-on-error
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(file_list_path),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "source_4",
            "--continue-on-error",
            "--verbose",
        ],
    )

    # Check that the command ran successfully despite the error
    assert result.exit_code == 0, f"Command failed with: {result.output}"

    # The nonexistent file is skipped during read_input_files, not during processing
    # So we won't see it in the processing output, but we should still see the valid file was processed

    # Check that the valid file was processed
    expected_output_path = (
        temp_output_dir / "2021_04_26_Batch1" / "BR00117037" / "BR00117037.parquet"
    )
    assert expected_output_path.exists(), "Valid file was not processed"

    # Verify the parquet file was created with the right data
    processed_df = pd.read_parquet(expected_output_path)
    assert "Metadata_Source" in processed_df.columns
    assert all(processed_df["Metadata_Source"] == "source_4")

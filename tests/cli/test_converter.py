#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from jump_profiling_recipe.cli.converter import (
    convert_command,
    read_mandatory_feature_cols,
    extract_batch_from_path,
)

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
def mandatory_feature_cols_file():
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


@pytest.fixture
def mock_csv_with_jcp2022(tmp_path):
    """Create a mock CSV file with a JCP2022 column."""
    file_path = tmp_path / "2021_04_26_Batch1" / "BR00117037" / "mock_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a dataframe with minimal required columns and a JCP2022 column
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001"] * 5,
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05"],
            "Metadata_PlateType": ["COMPOUND"] * 5,
            "JCP2022_ID": [
                "JCP2022_001122",
                "JCP2022_002233",
                "JCP2022_003344",
                "JCP2022_004455",
                "JCP2022_005566",
            ],
            "Feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Feature2": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    df.to_csv(file_path, index=False)

    # Create a file list for this mock CSV
    file_list_path = tmp_path / "mock_file_list.txt"
    with open(file_list_path, "w") as f:
        f.write(str(file_path.absolute()))

    return file_list_path


def test_extract_batch_from_path():
    """Test the extract_batch_from_path function."""
    # Test with a typical path structure
    path = Path("a/b/c/d/2021_04_26_Batch1/BR00117037/BR00117037.csv")
    assert extract_batch_from_path(path) == "2021_04_26_Batch1"

    # Test with a shorter path
    path = Path("2021_04_26_Batch1/BR00117037/BR00117037.csv")
    assert extract_batch_from_path(path) == "2021_04_26_Batch1"

    # Test with a minimal path
    path = Path("BR00117037/BR00117037.csv")
    assert extract_batch_from_path(path) == "BR00117037"

    # Test with a single file (fallback case)
    path = Path("BR00117037.csv")
    assert extract_batch_from_path(path) == "unknown_batch"


def test_convert_command(
    input_file_path, mandatory_feature_cols_file, temp_output_dir, temp_file_list
):
    """Test the convert command with a real input file."""
    # Skip test if input file doesn't exist
    if not input_file_path.exists():
        pytest.skip(f"Input file not found: {input_file_path}")

    # Skip test if mandatory features file doesn't exist
    if not mandatory_feature_cols_file.exists():
        pytest.skip(f"Mandatory features file not found: {mandatory_feature_cols_file}")

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
            "--mandatory-feature-cols-file",
            str(mandatory_feature_cols_file),
            "--verbose",
            "--default-plate-type",
            "TEST_PLATE_TYPE",
        ],
    )

    # Check that the command ran successfully
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

    # Base output directory paths for different file types
    base_profile_dir = temp_output_dir / "profiles" / "2021_04_26_Batch1" / "BR00117037"
    base_metadata_dir = (
        temp_output_dir / "metadata" / "2021_04_26_Batch1" / "BR00117037"
    )

    # Expected output paths
    expected_profile_path = base_profile_dir / "BR00117037.parquet"
    expected_plate_metadata_path = base_metadata_dir / "plate.parquet"
    expected_well_metadata_path = base_metadata_dir / "well.parquet"

    # Check that the output files exist
    assert expected_profile_path.exists(), (
        f"Profile file not created: {expected_profile_path}"
    )
    assert expected_plate_metadata_path.exists(), (
        f"Plate metadata file not created: {expected_plate_metadata_path}"
    )
    assert expected_well_metadata_path.exists(), (
        f"Well metadata file not created: {expected_well_metadata_path}"
    )

    # Read the original and converted files
    original_df = pd.read_parquet(input_file_path)
    converted_df = pd.read_parquet(expected_profile_path)
    plate_metadata_df = pd.read_parquet(expected_plate_metadata_path)
    well_metadata_df = pd.read_parquet(expected_well_metadata_path)

    # Load mandatory features
    mandatory_features = read_mandatory_feature_cols(mandatory_feature_cols_file)

    # Verify profile file has the expected structure
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

    # Verify plate metadata file has the expected structure
    # 1. Should have exactly one row
    assert len(plate_metadata_df) == 1, "Plate metadata should have exactly one row"

    # 2. Should have the expected columns
    expected_plate_columns = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_PlateType",
    ]
    assert set(plate_metadata_df.columns) == set(expected_plate_columns)

    # 3. Values should be as expected
    assert plate_metadata_df["Metadata_Source"].iloc[0] == "source_4"
    assert plate_metadata_df["Metadata_Batch"].iloc[0] == "2021_04_26_Batch1"
    assert (
        plate_metadata_df["Metadata_Plate"].iloc[0]
        == original_df["Metadata_Plate"].iloc[0]
    )

    # 4. PlateType should be set to default if not in original data
    if "Metadata_PlateType" in original_df.columns:
        assert (
            plate_metadata_df["Metadata_PlateType"].iloc[0]
            == original_df["Metadata_PlateType"].iloc[0]
        )
    else:
        assert plate_metadata_df["Metadata_PlateType"].iloc[0] == "TEST_PLATE_TYPE"

    # Verify well metadata file has the expected structure
    # 1. Should have the expected columns
    expected_well_columns = [
        "Metadata_Source",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_JCP2022",
    ]
    assert set(well_metadata_df.columns) == set(expected_well_columns)

    # 2. Source, Plate, Well values should match the profile data
    assert all(well_metadata_df["Metadata_Source"] == "source_4")
    assert all(well_metadata_df["Metadata_Plate"].isin(original_df["Metadata_Plate"]))
    assert all(well_metadata_df["Metadata_Well"].isin(original_df["Metadata_Well"]))

    # 3. Should have no duplicate well entries
    well_combinations = well_metadata_df[
        ["Metadata_Plate", "Metadata_Well"]
    ].drop_duplicates()
    assert len(well_combinations) == len(well_metadata_df), (
        "Well metadata contains duplicates"
    )


def test_jcp2022_handling(mock_csv_with_jcp2022, temp_output_dir):
    """Test the JCP2022 column handling functionality."""
    # Run the convert command with JCP2022 column specified
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(mock_csv_with_jcp2022),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "test_source",
            "--verbose",
            "--jcp2022-col",
            "JCP2022_ID",
        ],
    )

    # Check that the command ran successfully
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

    # Get the output well metadata file
    well_metadata_path = (
        temp_output_dir
        / "metadata"
        / "2021_04_26_Batch1"
        / "BR00117037"
        / "well.parquet"
    )
    assert well_metadata_path.exists(), (
        f"Well metadata file not created: {well_metadata_path}"
    )

    # Read the well metadata
    well_df = pd.read_parquet(well_metadata_path)

    # Verify JCP2022 column is properly populated
    assert "Metadata_JCP2022" in well_df.columns
    # We should have 5 unique JCP2022 values from our mock data
    assert set(well_df["Metadata_JCP2022"]) == {
        "JCP2022_001122",
        "JCP2022_002233",
        "JCP2022_003344",
        "JCP2022_004455",
        "JCP2022_005566",
    }

    # Check plate metadata as well
    plate_metadata_path = (
        temp_output_dir
        / "metadata"
        / "2021_04_26_Batch1"
        / "BR00117037"
        / "plate.parquet"
    )
    assert plate_metadata_path.exists(), (
        f"Plate metadata file not created: {plate_metadata_path}"
    )

    plate_df = pd.read_parquet(plate_metadata_path)

    # Verify plate type is properly set from the source data
    assert plate_df["Metadata_PlateType"].iloc[0] == "COMPOUND"


def test_multiple_plates_error(tmp_path, temp_output_dir):
    """Test that an error is raised when a file contains multiple plates."""
    # Create a mock CSV file with multiple plates
    file_path = tmp_path / "multiple_plates.csv"

    # Create a dataframe with multiple plates
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001", "PLATE001", "PLATE002", "PLATE002"],
            "Metadata_Well": ["A01", "A02", "A01", "A02"],
            "Feature1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    df.to_csv(file_path, index=False)

    # Create a file list for this mock CSV
    file_list_path = tmp_path / "multi_plate_file_list.txt"
    with open(file_list_path, "w") as f:
        f.write(str(file_path.absolute()))

    # Run the convert command
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(file_list_path),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "test_source",
        ],
    )

    # Check that the command failed with the expected error
    assert result.exit_code != 0
    assert "Multiple plates found" in result.output


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
        temp_output_dir
        / "profiles"
        / "2021_04_26_Batch1"
        / "BR00117037"
        / "BR00117037.parquet"
    )
    assert expected_output_path.exists(), "Valid file was not processed"

    # Verify the parquet file was created with the right data
    processed_df = pd.read_parquet(expected_output_path)
    assert "Metadata_Source" in processed_df.columns
    assert all(processed_df["Metadata_Source"] == "source_4")

    # Verify metadata files were also created
    expected_plate_path = (
        temp_output_dir
        / "metadata"
        / "2021_04_26_Batch1"
        / "BR00117037"
        / "plate.parquet"
    )
    expected_well_path = (
        temp_output_dir
        / "metadata"
        / "2021_04_26_Batch1"
        / "BR00117037"
        / "well.parquet"
    )

    assert expected_plate_path.exists(), "Plate metadata file was not created"
    assert expected_well_path.exists(), "Well metadata file was not created"

#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path
import logging

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


@pytest.fixture
def mock_csv_with_multiple_jcp2022_cols(tmp_path):
    """Create a mock CSV file with multiple columns for JCP2022 concatenation."""
    file_path = tmp_path / "2021_04_26_Batch1" / "BR00117037" / "mock_multi_data.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a dataframe with minimal required columns and multiple JCP2022-related columns
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001"] * 3,
            "Metadata_Well": ["A01", "A02", "A03"],
            "Compound_ID": ["C123", "C456", "C789"],
            "Concentration": ["10nM", "100nM", "1uM"],
            "Feature1": [1.0, 2.0, 3.0],
        }
    )

    df.to_csv(file_path, index=False)

    # Create a file list for this mock CSV
    file_list_path = tmp_path / "mock_multi_file_list.txt"
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
            "--jcp2022-cols",
            "Metadata_Well",  # Because there's not a JCP2022 column in the fixture file
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
    mandatory_feature_cols = read_mandatory_feature_cols(mandatory_feature_cols_file)

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
    assert set(feature_columns).issubset(mandatory_feature_cols)

    # 4. All features in the output should match the original values
    for feature in feature_columns:
        assert feature in original_df.columns
        pd.testing.assert_series_equal(
            converted_df[feature], original_df[feature], check_names=False
        )

    # 5. Verify we have all available mandatory features
    available_mandatory_feature_cols = [
        f for f in mandatory_feature_cols if f in original_df.columns
    ]
    assert set(feature_columns) == set(available_mandatory_feature_cols)

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
            "--jcp2022-cols",
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


def test_multiple_jcp2022_cols_concatenation(
    mock_csv_with_multiple_jcp2022_cols, temp_output_dir
):
    """Test the concatenation of multiple columns for JCP2022."""
    # Run the convert command with multiple JCP2022 columns specified
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(mock_csv_with_multiple_jcp2022_cols),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "test_source",
            "--verbose",
            "--jcp2022-cols",
            "Compound_ID,Concentration",  # Specify multiple columns
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

    # Verify JCP2022 column is properly populated with concatenated values
    assert "Metadata_JCP2022" in well_df.columns

    # We should have 3 unique JCP2022 values with the expected concatenation format
    expected_values = {"C123:10nM", "C456:100nM", "C789:1uM"}
    assert set(well_df["Metadata_JCP2022"]) == expected_values, (
        f"Expected concatenated values {expected_values}, got {set(well_df['Metadata_JCP2022'])}"
    )


def test_multiple_plates_error(tmp_path, temp_output_dir):
    """Test that an error is raised when a file contains multiple plates."""
    # Create a mock CSV file with multiple plates
    file_path = tmp_path / "multiple_plates.csv"

    # Create a dataframe with multiple plates
    df = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001", "PLATE001", "PLATE002", "PLATE002"],
            "Metadata_Well": ["A01", "A02", "A01", "A02"],
            "Metadata_JCP2022": [
                "JCP2022_001122",
                "JCP2022_002233",
                "JCP2022_003344",
                "JCP2022_004455",
            ],
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
            "--jcp2022-cols",
            "Metadata_JCP2022",
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
            "--jcp2022-cols",
            "Metadata_Well",  # Because there's not a JCP2022 column in the fixture file
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


@pytest.fixture
def multiple_input_files(tmp_path):
    """Create multiple mock CSV files in different directories to test collation."""
    # Setup directory structure
    batch_dir1 = tmp_path / "2021_04_26_Batch1"
    batch_dir2 = tmp_path / "2021_04_27_Batch2"

    plate_dirs = [
        batch_dir1 / "PLATE001",
        batch_dir1 / "PLATE002",
        batch_dir2 / "PLATE003",
    ]

    # Create directories
    for plate_dir in plate_dirs:
        plate_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV files with test data
    files = []
    for i, plate_dir in enumerate(plate_dirs):
        plate_id = f"PLATE00{i + 1}"
        file_path = plate_dir / f"{plate_id}.csv"

        # Create test data with different wells for each plate
        wells = [f"{row}{col}" for row in "ABC" for col in range(1, 3)]
        df = pd.DataFrame(
            {
                "Metadata_Plate": [plate_id] * len(wells),
                "Metadata_Well": wells,
                "Metadata_JCP2022": [f"JCP2022_{i:06d}" for i in range(len(wells))],
                "Feature1": [float(i) for i in range(len(wells))],
            }
        )

        df.to_csv(file_path, index=False)
        files.append(file_path)

    # Create file list
    file_list_path = tmp_path / "multi_file_list.txt"
    with open(file_list_path, "w") as f:
        for file_path in files:
            f.write(f"{file_path}\n")

    return file_list_path


def test_metadata_collation(multiple_input_files, temp_output_dir):
    """Test the collation of metadata files after processing multiple files."""
    # Run the convert command with multiple input files
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(multiple_input_files),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "test_source",
            "--verbose",
            "--jcp2022-cols",
            "Metadata_JCP2022",
        ],
    )

    # Check that the command ran successfully
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

    # Verify that collated metadata files were created
    collated_plate_path = temp_output_dir / "metadata" / "plate.parquet"
    collated_well_path = temp_output_dir / "metadata" / "well.parquet"

    assert collated_plate_path.exists(), "Collated plate metadata file not created"
    assert collated_well_path.exists(), "Collated well metadata file not created"

    # Verify collated plate metadata
    plate_df = pd.read_parquet(collated_plate_path)

    # Should have 3 unique plates
    assert len(plate_df) == 3, f"Expected 3 plates, got {len(plate_df)}"

    # Check if plates from both batches are present
    assert "2021_04_26_Batch1" in plate_df["Metadata_Batch"].values
    assert "2021_04_27_Batch2" in plate_df["Metadata_Batch"].values

    # Verify all expected plates exist
    expected_plates = {"PLATE001", "PLATE002", "PLATE003"}
    assert set(plate_df["Metadata_Plate"]) == expected_plates

    # Verify collated well metadata
    well_df = pd.read_parquet(collated_well_path)

    # Should have 18 unique wells (3 plates × 6 wells)
    assert len(well_df) == 18, f"Expected 18 wells, got {len(well_df)}"

    # Check if wells from all plates are present
    for plate in expected_plates:
        assert plate in well_df["Metadata_Plate"].values

    # Check that there are no duplicate combinations of plate+well
    well_combinations = well_df[["Metadata_Plate", "Metadata_Well"]].drop_duplicates()
    assert len(well_combinations) == len(well_df), "Well metadata contains duplicates"


@pytest.fixture
def multiple_input_files_with_duplicates(tmp_path):
    """Create input files with duplicate metadata to test duplicate detection."""
    # Create different batch directories to avoid overwriting metadata files
    batch_dir1 = tmp_path / "2021_04_26_Batch1"
    batch_dir2 = tmp_path / "2021_04_26_Batch2"  # Using a different batch directory

    plate_dir1 = batch_dir1 / "PLATE001"
    plate_dir2 = batch_dir2 / "PLATE001"  # Same plate name but different batch
    plate_dir3 = batch_dir1 / "PLATE002"

    for dir_path in [plate_dir1, plate_dir2, plate_dir3]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create first CSV with data for PLATE001 in Batch1
    file1_path = plate_dir1 / "PLATE001_file1.csv"
    df1 = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001"] * 4,
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "Metadata_JCP2022": [
                "JCP2022_001122",
                "JCP2022_002233",
                "JCP2022_003344",
                "JCP2022_004455",
            ],
            "Feature1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df1.to_csv(file1_path, index=False)

    # Create a second file with the SAME plate in Batch2
    # This creates duplicate wells when collated
    file2_path = plate_dir2 / "PLATE001_file2.csv"
    df2 = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE001"] * 3,
            "Metadata_Well": ["A03", "A04", "A05"],  # A03 and A04 will be duplicates
            "Metadata_JCP2022": [
                "JCP2022_001122",
                "JCP2022_002233",
                "JCP2022_003344",
            ],
            "Feature1": [30.0, 40.0, 5.0],
        }
    )
    df2.to_csv(file2_path, index=False)

    # Create file for PLATE002 (no duplicates here)
    file3_path = plate_dir3 / "PLATE002.csv"
    df3 = pd.DataFrame(
        {
            "Metadata_Plate": ["PLATE002"] * 3,
            "Metadata_Well": ["B01", "B02", "B03"],
            "Metadata_JCP2022": [
                "JCP2022_001122",
                "JCP2022_002233",
                "JCP2022_003344",
            ],
            "Feature1": [6.0, 7.0, 8.0],
        }
    )
    df3.to_csv(file3_path, index=False)

    # Create file list
    file_list_path = tmp_path / "duplicate_file_list.txt"
    with open(file_list_path, "w") as f:
        f.write(f"{file1_path}\n")
        f.write(f"{file2_path}\n")
        f.write(f"{file3_path}\n")

    return file_list_path


def test_metadata_collation_with_duplicates(
    multiple_input_files_with_duplicates, temp_output_dir, caplog
):
    """Test the detection and reporting of duplicates during metadata collation."""
    # Set the log level to capture error messages
    caplog.set_level(logging.ERROR)

    # Run the convert command with files containing duplicate metadata
    runner = CliRunner()
    result = runner.invoke(
        convert_command,
        [
            str(multiple_input_files_with_duplicates),
            "--output-dir",
            str(temp_output_dir),
            "--source",
            "test_source",
            "--jcp2022-cols",
            "Metadata_JCP2022",
            "--verbose",
            "--continue-on-error",  # Add this to ensure all files are processed
        ],
    )

    # Command should still succeed but with warnings
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

    # Check if collated metadata files were created
    collated_plate_path = temp_output_dir / "metadata" / "plate.parquet"
    collated_well_path = temp_output_dir / "metadata" / "well.parquet"

    assert collated_plate_path.exists(), "Collated plate metadata file not created"
    assert collated_well_path.exists(), "Collated well metadata file not created"

    # Verify that the duplicate detection triggered warnings in the logs
    duplicate_log_messages = [
        record.message
        for record in caplog.records
        if "duplicate entries in well metadata" in record.message.lower()
        or "found duplicate" in record.message.lower()
    ]
    assert duplicate_log_messages, "No duplicate warnings found in logs"

    # Check that the duplicates CSV file was created
    duplicates_csv_path = temp_output_dir / "metadata" / "well_duplicates.csv"
    assert duplicates_csv_path.exists(), "Duplicates CSV file was not created"

    # Verify the duplicates CSV contains the expected entries
    duplicates_df = pd.read_csv(duplicates_csv_path)

    # Should have at least 4 rows (2 duplicated wells × 2 occurrences each)
    assert len(duplicates_df) >= 4, "Duplicates file doesn't have enough entries"

    # Check if the known duplicate wells are in the file
    duplicate_wells = duplicates_df[duplicates_df["Metadata_Plate"] == "PLATE001"][
        "Metadata_Well"
    ]
    assert "A03" in duplicate_wells.values, "Known duplicate A03 not found"
    assert "A04" in duplicate_wells.values, "Known duplicate A04 not found"

    # Verify the collated well metadata still exists and contains the duplicates
    well_df = pd.read_parquet(collated_well_path)

    # Get the full set of wells for PLATE001
    plate001_wells = well_df[well_df["Metadata_Plate"] == "PLATE001"][
        "Metadata_Well"
    ].tolist()

    # Count occurrences of A03 and A04 (our known duplicates)
    a03_count = plate001_wells.count("A03")
    a04_count = plate001_wells.count("A04")

    # We should have 2 of each
    assert a03_count == 2, f"Expected 2 occurrences of A03, got {a03_count}"
    assert a04_count == 2, f"Expected 2 occurrences of A04, got {a04_count}"

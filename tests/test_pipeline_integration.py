import sys
import shutil
from pathlib import Path
from snakemake import snakemake
import pytest
import pandas as pd

# Ensure the repository root is on PYTHONPATH so that modules like 'correct' can be found
repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def test_workspace(tmp_path):
    """
    Set up a temporary working directory with necessary files:
    - Snakefile
    - Config file
    - rules directory
    - cell_counts and metadata folders
    - source_13 data from test fixtures
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Copy Snakefile
    snakefile_src = Path.cwd() / "Snakefile"
    snakefile_dst = workspace / "Snakefile"
    shutil.copy(snakefile_src, snakefile_dst)

    # Copy the entire rules directory so includes will work
    rules_src = Path.cwd() / "rules"
    rules_dst = workspace / "rules"
    shutil.copytree(rules_src, rules_dst)

    # Copy the cell_counts folder
    cell_counts_src = Path.cwd() / "inputs" / "cell_counts"
    cell_counts_dst = workspace / "inputs" / "cell_counts"
    shutil.copytree(cell_counts_src, cell_counts_dst)

    # Copy the metadata folder
    metadata_src = Path.cwd() / "inputs" / "metadata"
    metadata_dst = workspace / "inputs" / "metadata"
    shutil.copytree(metadata_src, metadata_dst)

    # Copy the source_13 data from test fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    source_13_src = fixtures_dir / "inputs" / "source_13"
    source_13_dst = workspace / "inputs" / "source_13"
    shutil.copytree(source_13_src, source_13_dst)

    # Copy config file
    configfile_src = (
        Path(__file__).parent / "fixtures" / "inputs" / "config" / "crispr_trimmed.json"
    )
    configfile_dst = workspace / "crispr_trimmed.json"
    shutil.copy(configfile_src, configfile_dst)

    return workspace


def test_full_pipeline(test_workspace):
    """
    An integration test that runs the Snakemake pipeline using the provided configuration
    and checks that the final output files exist and match expected values.
    """
    workspace = test_workspace
    snakefile = workspace / "Snakefile"
    configfile = workspace / "crispr_trimmed.json"

    success = snakemake(
        snakefile=str(snakefile),
        configfiles=[str(configfile)],
        cores=1,
        workdir=str(workspace),
        quiet=True,  # Suppress output
        latency_wait=10,  # Increase if there are filesystem delays
    )
    assert success, "Snakemake pipeline did not complete successfully"

    # Verify that the final target from the 'all' rule exists.
    done_file = workspace / "outputs" / "crispr_trimmed" / "reformat.done"
    assert done_file.exists(), f"Expected output file {done_file} was not created"

    # Verify that the corrected profiles parquet file exists and matches expected values
    profiles_file = (
        workspace / "outputs" / "crispr_trimmed_public" / "profiles_trimmed.parquet"
    )
    assert profiles_file.exists(), (
        f"Expected output file {profiles_file} was not created"
    )

    # Load actual and expected parquet files
    actual_df = pd.read_parquet(profiles_file)
    expected_df = pd.read_parquet(
        Path(__file__).parent
        / "fixtures"
        / "outputs"
        / "crispr_trimmed_public"
        / "profiles_trimmed.parquet"
    )

    # First try comparing only Metadata columns
    metadata_cols = [col for col in actual_df.columns if col.startswith("Metadata_")]
    try:
        pd.testing.assert_frame_equal(
            actual_df[metadata_cols], expected_df[metadata_cols], check_dtype=True
        )
        print("Metadata columns match exactly")
    except AssertionError as e:
        print("Differences found in Metadata columns:")
        print(e)

    # Then compare full DataFrames with detailed error reporting
    try:
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True)
    except AssertionError as e:
        # Get differences in column names if any
        actual_cols = set(actual_df.columns)
        expected_cols = set(expected_df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            print(f"Missing columns in actual: {missing_cols}")
        if extra_cols:
            print(f"Extra columns in actual: {extra_cols}")

        # If columns are same, show sample of differing values
        if not missing_cols and not extra_cols:
            common_cols = list(actual_cols)
            differences = (actual_df != expected_df).any()
            diff_cols = differences[differences].index.tolist()

            if diff_cols:
                print("\nColumns with differences:", diff_cols)
                print("\nSample of differing values:")
                for col in diff_cols[:5]:  # Show first 5 differing columns
                    mismatch_mask = actual_df[col] != expected_df[col]
                    if mismatch_mask.any():
                        print(f"\nColumn: {col}")
                        print("Actual vs Expected (first 5 differences):")
                        mismatched_rows = mismatch_mask[mismatch_mask].index[:5]
                        for idx in mismatched_rows:
                            print(
                                f"Row {idx}: {actual_df.loc[idx, col]} vs {expected_df.loc[idx, col]}"
                            )

        raise AssertionError("DataFrames are not equal") from e

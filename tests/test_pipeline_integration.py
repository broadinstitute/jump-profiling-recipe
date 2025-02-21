import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from snakemake.api import (
    ConfigSettings,
    OutputSettings,
    ResourceSettings,
    SnakemakeApi,
    StorageSettings,
)


def run_workflow(snakefile: Path, configfile: Path):
    """Run programmatically a snakefile using the given config"""
    resource = ResourceSettings(cores=1)
    config = ConfigSettings(configfiles=[configfile])
    with SnakemakeApi(
        OutputSettings(
            verbose=False,
            show_failed_logs=True,
        ),
    ) as snakemake_api:
        workflow_api = snakemake_api.workflow(
            storage_settings=StorageSettings(),
            resource_settings=resource,
            config_settings=config,
            snakefile=snakefile,
            workdir=snakefile.parent,
        )
        dag_api = workflow_api.dag()
        dag_api.execute_workflow()


# Ensure the repository root is on PYTHONPATH so that modules like 'correct' can be found
repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def test_workspace(tmp_path):
    """
    Set up a temporary working directory with necessary files copied from the fixtures folder
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Copy entire input and output folders from fixtures
    for folder in ["inputs", "outputs"]:
        src = fixtures_dir / folder
        dst = workspace / folder
        if src.exists():
            shutil.copytree(src, dst)

    # Copy Snakefile and rules directory
    shutil.copy(fixtures_dir / "Snakefile", workspace / "Snakefile")
    shutil.copytree(fixtures_dir / "rules", workspace / "rules")

    return workspace


def test_full_pipeline(test_workspace):
    """
    Integration test that runs the Snakemake pipeline and verifies outputs.
    """
    workspace = test_workspace
    snakefile = workspace / "Snakefile"
    configfile = workspace / "inputs" / "config" / "crispr_trimmed.json"

    # Run the pipeline
    run_workflow(snakefile, configfile)

    # Verify outputs
    done_file = workspace / "outputs" / "crispr" / "reformat.done"
    assert done_file.exists(), f"Expected output file {done_file} was not created"

    expected_parquet_files = {
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected.parquet": True,  # Allow approximate comparison
        "profiles_wellpos_cc_var_mad_outlier_featselect.parquet": False,  # Require exact comparison
    }

    def compare_dataframes(actual_df, expected_df, filename, allow_approximate):
        """Helper function to compare DataFrames with detailed reporting"""
        # First compare metadata columns
        metadata_cols = [
            col for col in actual_df.columns if col.startswith("Metadata_")
        ]
        try:
            pd.testing.assert_frame_equal(
                actual_df[metadata_cols], expected_df[metadata_cols], check_dtype=True
            )
        except AssertionError as e:
            raise AssertionError(
                f"Metadata columns don't match in {filename}:\n{str(e)}"
            )

        # Then compare numerical columns with or without tolerance based on allow_approximate
        numerical_cols = [col for col in actual_df.columns if col not in metadata_cols]
        try:
            # TODO: remove abs function by updating the fixture.
            pd.testing.assert_frame_equal(
                actual_df[numerical_cols].abs(),
                expected_df[numerical_cols].abs(),
                rtol=1e-4 if allow_approximate else 0,  # Only use tolerance if allowed
                atol=1e-4 if allow_approximate else 0,  # Only use tolerance if allowed
                check_dtype=False,  # Allow float32/float64 differences
            )
        except AssertionError as e:
            raise AssertionError(
                f"Numerical columns don't match in {filename}:\n{str(e)}"
            )

    for parquet_filename, allow_approximate in expected_parquet_files.items():
        profiles_file = workspace / "outputs" / "crispr_public" / parquet_filename
        assert profiles_file.exists(), (
            f"Expected output file {profiles_file} was not created"
        )

        actual_df = pd.read_parquet(profiles_file)
        expected_file_path = (
            Path(__file__).parent
            / "fixtures"
            / "outputs"
            / "crispr_public"
            / parquet_filename
        )
        expected_df = pd.read_parquet(expected_file_path)

        try:
            compare_dataframes(
                actual_df, expected_df, parquet_filename, allow_approximate
            )
        except AssertionError as e:
            # Get differences in column names if any
            actual_cols = set(actual_df.columns)
            expected_cols = set(expected_df.columns)
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols

            if missing_cols or extra_cols:
                raise AssertionError(
                    f"Column mismatch in {parquet_filename}:\n"
                    f"Missing columns: {missing_cols}\n"
                    f"Extra columns: {extra_cols}\n"
                    f"Original error: {str(e)}"
                )
            raise

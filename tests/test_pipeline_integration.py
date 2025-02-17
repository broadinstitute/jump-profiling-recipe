import sys
import json
import tempfile
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
    configfile_src = Path(__file__).parent / "fixtures" / "inputs" / "config" / "crispr_trimmed.json"
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
         latency_wait=10  # Increase if there are filesystem delays
    )
    assert success, "Snakemake pipeline did not complete successfully"

    # Verify that the final target from the 'all' rule exists.
    done_file = workspace / "outputs" / "crispr_trimmed" / "reformat.done"
    assert done_file.exists(), f"Expected output file {done_file} was not created"

    # Verify that the corrected profiles parquet file exists and matches expected values
    profiles_file = workspace / "outputs" / "crispr_trimmed" / "profiles_trimmed_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected.parquet"
    assert profiles_file.exists(), f"Expected output file {profiles_file} was not created"
    
    # Load actual and expected parquet files
    actual_df = pd.read_parquet(profiles_file)
    expected_df = pd.read_parquet(Path(__file__).parent / "fixtures" / "outputs" / "crispr_trimmed_public" / "profiles_trimmed_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected.parquet")
    
    # Compare DataFrames
    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True) 
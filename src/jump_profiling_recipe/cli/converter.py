#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List, Optional, Set
import time

import click
import pandas as pd
import tqdm

# Set up logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_output_path(input_file: Path, output_dir: Path) -> Path:
    """
    Generate output path preserving two directory levels above the input file.

    Example:
        input:  a/b/c/d/2021_04_17_Batch1/BR00121331/BR00121331.csv.gz
        output: {output_dir}/profiles/2021_04_17_Batch1/BR00121331/BR00121331.parquet
    """
    # Get the last three parts of the path (two dirs + filename)
    parts = input_file.parts[-3:]
    if len(parts) < 3:
        # If path is not deep enough, just use available parts
        parts = input_file.parts

    # Construct new path: output_dir/dir1/dir2/filename
    relative_path = Path(*parts[:-1]) / input_file.stem
    return output_dir / "profiles" / relative_path.with_suffix(".parquet")


def read_input_files(file_list: Path) -> List[Path]:
    """Read list of input files from a text file."""
    if not file_list.exists():
        raise click.ClickException(f"Input file list not found: {file_list}")

    files = []
    with file_list.open() as f:
        for line in f:
            # Strip whitespace and skip empty lines
            filepath = line.strip()
            if not filepath:
                continue

            path = Path(filepath)
            if not path.exists():
                logger.warning(f"File not found, skipping: {path}")
                continue

            if not path.is_file():
                logger.warning(f"Not a file, skipping: {path}")
                continue

            # Check for both simple and compound extensions
            name = path.name.lower()
            if not (
                name.endswith(".csv")
                or name.endswith(".csv.gz")
                or name.endswith(".parquet")
            ):
                logger.warning(f"Unsupported file type, skipping: {path}")
                continue

            files.append(path)

    if not files:
        raise click.ClickException("No valid input files found in the list")

    return files


def read_mandatory_feature_cols(feature_file: Path) -> Set[str]:
    """Read mandatory feature names from a file.

    Args:
        feature_file: Path to text file containing feature names (one per line)

    Returns:
        Set of feature names
    """
    if not feature_file.exists():
        raise click.ClickException(f"Mandatory features file not found: {feature_file}")

    features = set()
    with feature_file.open() as f:
        for line in f:
            # Strip whitespace and skip empty lines
            feature = line.strip()
            if feature:
                features.add(feature)

    if not features:
        raise click.ClickException("No features found in mandatory features file")

    logger.info(f"Loaded {len(features)} mandatory features from {feature_file}")
    return features


def process_file(
    input_file: Path,
    output_dir: Path,
    source: str,
    mandatory_feature_cols: Optional[Set[str]] = None,
    mandatory_metadata_cols: List[str] = ["Metadata_Plate", "Metadata_Well"],
) -> None:
    """
    Process a single input file and save it as parquet.

    Args:
        input_file: Path to input file
        output_dir: Directory to save output
        source: Value to set in Metadata_Source column
        mandatory_feature_cols: Optional set of feature column names that must be included
        mandatory_metadata_cols: List of required metadata columns (default: ["Metadata_Plate", "Metadata_Well"])
    """
    logger.info(f"Processing file: {input_file}")

    # Generate output path preserving directory structure
    output_file = get_output_path(input_file, output_dir)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine input file type using full name
    name = input_file.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        df = pd.read_csv(input_file)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    # Ensure required Metadata columns exist
    for col in mandatory_metadata_cols:
        if col not in df.columns:
            raise click.ClickException(
                f"Required column '{col}' not found in {input_file}"
            )

    # Prepare the metadata columns
    metadata_dict = {"Metadata_Source": [source] * len(df)}
    for col in mandatory_metadata_cols:
        metadata_dict[col] = df[col]

    metadata_df = pd.DataFrame(metadata_dict)

    # Get non-metadata columns
    feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]

    # If mandatory features are specified, check for missing ones and filter
    if mandatory_feature_cols:
        missing_features = mandatory_feature_cols - set(feature_cols)
        if missing_features:
            logger.warning(
                f"Missing mandatory features in {input_file}: {sorted(missing_features)}"
            )

        # Filter to keep only mandatory features (that exist in the data)
        feature_cols = [col for col in feature_cols if col in mandatory_feature_cols]
        logger.info(
            f"Keeping {len(feature_cols)} out of {len(mandatory_feature_cols)} mandatory features for {input_file}"
        )
        if not feature_cols:
            logger.warning(f"No mandatory features found in {input_file}")

    feature_df = df[feature_cols]

    # Combine metadata and feature columns efficiently
    new_df = pd.concat([metadata_df, feature_df], axis=1)

    logger.debug(f"Columns in output: {list(new_df.columns)}")

    # Save as parquet
    new_df.to_parquet(output_file)
    logger.info(f"Saved processed file to: {output_file}")

    # Log column changes if in verbose mode
    if logger.getEffectiveLevel() <= logging.DEBUG:
        dropped_metadata = set(
            col for col in df.columns if col.startswith("Metadata_")
        ) - set(mandatory_metadata_cols)
        if dropped_metadata:
            logger.debug(f"Dropped metadata columns: {sorted(dropped_metadata)}")


def process_files(
    input_files: List[Path],
    output_dir: Path,
    source: str,
    mandatory_feature_cols: Optional[Set[str]] = None,
    continue_on_error: bool = False,
    mandatory_metadata_cols: List[str] = ["Metadata_Plate", "Metadata_Well"],
) -> None:
    """Process multiple input files.

    Args:
        input_files: List of input file paths
        output_dir: Directory to save output files
        source: Value to set in Metadata_Source column
        mandatory_feature_cols: Optional set of feature column names that must be included
        continue_on_error: If True, continue processing other files when one fails
        mandatory_metadata_cols: List of required metadata columns to preserve
    """
    failures = 0
    start_time = time.time()
    total_files = len(input_files)

    logger.info(f"Processing {total_files} files...")

    # Use tqdm for progress tracking if we have multiple files
    for i, input_file in enumerate(
        tqdm.tqdm(input_files, desc="Processing files", unit="file")
    ):
        try:
            file_start_time = time.time()
            process_file(
                input_file,
                output_dir,
                source,
                mandatory_feature_cols,
                mandatory_metadata_cols,
            )
            file_elapsed = time.time() - file_start_time
            logger.debug(f"Processed {input_file} in {file_elapsed:.2f} seconds")
        except Exception as e:
            failures += 1
            logger.error(f"Error processing {input_file}: {str(e)}")
            if not continue_on_error:
                logger.error(
                    f"Processing halted after {i + 1}/{total_files} files. Use --continue-on-error to process all files."
                )
                raise click.ClickException(str(e))

    elapsed = time.time() - start_time
    logger.info(
        f"Processed {total_files - failures}/{total_files} files in {elapsed:.2f} seconds"
    )

    if failures > 0:
        logger.warning(f"Completed with {failures} file(s) failed to process")


@click.command("convert")
@click.argument(
    "file_list", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Output directory for processed Parquet files",
)
@click.option(
    "-s",
    "--source",
    type=str,
    required=True,
    help="Source identifier to add as Metadata_Source column",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "-m",
    "--mandatory-feature-cols-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to file containing mandatory feature names (one per line). If provided, only these features will be included.",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue processing other files if an error occurs with one file",
)
@click.option(
    "--mandatory-metadata",
    type=str,
    default="Metadata_Plate,Metadata_Well",
    help="Comma-separated list of required metadata columns (default: 'Metadata_Plate,Metadata_Well')",
)
def convert_command(
    file_list: Path,
    output_dir: Path,
    source: str,
    verbose: bool,
    mandatory_feature_cols_file: Optional[Path],
    continue_on_error: bool,
    mandatory_metadata: str = "Metadata_Plate,Metadata_Well",
):
    """Convert CSV/Parquet files to processed Parquet files.

    Takes a text file containing a list of input files (one per line) and processes
    them into Parquet files in the specified output directory. The output files
    will maintain their original names but with .parquet extension.

    Each output file will have a Metadata_Source column added as the first column,
    filled with the specified source value.

    The input files can be either CSV or Parquet format. Invalid or missing files
    will be skipped with a warning.

    If a mandatory features file is provided, only those features will be kept in the
    output, and warnings will be logged for any missing features. The features file
    should contain one feature name per line.

    Arguments:
        file_list: Text file containing list of input files (one per line)
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process mandatory features if file provided
    mandatory_feature_cols = None
    if mandatory_feature_cols_file:
        mandatory_feature_cols = read_mandatory_feature_cols(
            mandatory_feature_cols_file
        )

    # Read and validate input files
    input_files = read_input_files(file_list)
    logger.info(f"Found {len(input_files)} valid input files")

    # Parse required metadata columns
    mandatory_metadata_cols = [
        col.strip() for col in mandatory_metadata.split(",") if col.strip()
    ]
    logger.info(f"Using required metadata columns: {mandatory_metadata_cols}")

    # Process files
    process_files(
        input_files,
        output_dir,
        source,
        mandatory_feature_cols,
        continue_on_error,
        mandatory_metadata_cols,
    )


if __name__ == "__main__":
    convert_command()

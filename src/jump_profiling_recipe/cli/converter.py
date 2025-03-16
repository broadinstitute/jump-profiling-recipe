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


def get_output_path(
    input_file: Path,
    output_dir: Path,
    file_type: str = "profiles",
    metadata_level: Optional[str] = None,
) -> Path:
    """
    Generate output path preserving two directory levels above the input file.

    Example (for profiles):
        input:  a/b/c/d/2021_04_17_Batch1/BR00121331/BR00121331.csv.gz
        output: {output_dir}/profiles/2021_04_17_Batch1/BR00121331/BR00121331.parquet

    Example (for plate metadata):
        input:  a/b/c/d/2021_04_17_Batch1/BR00121331/BR00121331.csv.gz
        output: {output_dir}/metadata/2021_04_17_Batch1/BR00121331/plate.parquet

    Example (for well metadata):
        input:  a/b/c/d/2021_04_17_Batch1/BR00121331/BR00121331.csv.gz
        output: {output_dir}/metadata/2021_04_17_Batch1/BR00121331/well.parquet

    Args:
        input_file: Path to input file
        output_dir: Base output directory
        file_type: Type of output file (profiles or metadata)
        metadata_level: For metadata files, specifies plate or well level

    Returns:
        Path to output file
    """
    # Get the last three parts of the path (two dirs + filename)
    parts = input_file.parts[-3:]
    if len(parts) < 3:
        # If path is not deep enough, just use available parts
        parts = input_file.parts

    # For profiles, use the name of the directory one level up from the file
    if file_type == "profiles":
        # Construct new path: output_dir/profiles/dir1/dir2/dir2.parquet
        # (using directory name instead of filename)
        relative_path = Path(*parts[:-1]) / parts[-2]
        return output_dir / file_type / relative_path.with_suffix(".parquet")
    # For metadata files, use plate.parquet or well.parquet
    elif file_type == "metadata" and metadata_level in ["plate", "well"]:
        # Construct new path: output_dir/metadata/dir1/dir2/metadata_level.parquet
        relative_path = Path(*parts[:-1]) / f"{metadata_level}"
        return output_dir / file_type / relative_path.with_suffix(".parquet")
    else:
        raise ValueError(
            f"Invalid file_type or metadata_level: {file_type}, {metadata_level}"
        )


def extract_batch_from_path(input_file: Path) -> str:
    """
    Extract batch information from input file path.

    The batch is considered to be the directory name two levels above the file.

    Example:
        input: a/b/c/d/2021_04_17_Batch1/BR00121331/BR00121331.csv.gz
        batch: 2021_04_17_Batch1

    Args:
        input_file: Path to input file

    Returns:
        Batch name as string
    """
    # Get the parts of the path and extract the batch name (two levels up from the file)
    parts = input_file.parts

    # Need at least 3 parts (batch_dir/plate_dir/file)
    if len(parts) >= 3:
        return parts[-3]
    elif len(parts) == 2:
        # If only two levels deep, use the top directory
        return parts[0]
    else:
        # Fallback if path is not deep enough
        return "unknown_batch"


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
    jcp2022_cols: str,
    mandatory_feature_cols: Optional[Set[str]] = None,
    mandatory_metadata_cols: List[str] = ["Metadata_Plate", "Metadata_Well"],
    default_plate_type: str = "UNKNOWN",
) -> None:
    """
    Process a single input file and save it as parquet.

    Args:
        input_file: Path to input file
        output_dir: Directory to save output
        source: Value to set in Metadata_Source column
        jcp2022_cols: Comma-separated list of columns to be treated as Metadata_JCP2022
        mandatory_feature_cols: Optional set of feature column names that must be included
        mandatory_metadata_cols: List of required metadata columns (default: ["Metadata_Plate", "Metadata_Well"])
        default_plate_type: Default value for Metadata_PlateType when not present in data
    """
    logger.info(f"Processing file: {input_file}")

    # Generate output paths preserving directory structure
    output_profile_file = get_output_path(input_file, output_dir, "profiles")
    output_plate_metadata_file = get_output_path(
        input_file, output_dir, "metadata", "plate"
    )
    output_well_metadata_file = get_output_path(
        input_file, output_dir, "metadata", "well"
    )

    # Create output directories if they don't exist
    output_profile_file.parent.mkdir(parents=True, exist_ok=True)
    # Both metadata files will be in the same directory
    output_plate_metadata_file.parent.mkdir(parents=True, exist_ok=True)

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

    logger.debug(f"Columns in output profile: {list(new_df.columns)}")

    # Save profile as parquet
    new_df.to_parquet(output_profile_file)
    logger.info(f"Saved processed profile file to: {output_profile_file}")

    # Create well metadata DataFrame
    # Columns: Metadata_Source, Metadata_Plate, Metadata_Well, Metadata_JCP2022
    well_metadata = {
        "Metadata_Source": metadata_df["Metadata_Source"],
        "Metadata_Plate": metadata_df["Metadata_Plate"],
        "Metadata_Well": metadata_df["Metadata_Well"],
    }

    # Add JCP2022 column using the columns from jcp2022_cols
    # Split the comma-separated list
    cols_list = [col.strip() for col in jcp2022_cols.split(",") if col.strip()]

    if not cols_list:
        # Handle empty string or all-whitespace input
        raise click.ClickException(
            f"Empty JCP2022 columns value for {input_file}. "
            f"Please specify valid column names using the --jcp2022-cols option."
        )

    # Check if all specified columns exist
    missing_cols = [col for col in cols_list if col not in df.columns]
    if missing_cols:
        raise click.ClickException(
            f"JCP2022 columns not found in {input_file}: {missing_cols}. "
            f"All JCP2022 columns must exist in the input file."
        )

    # For a single column, just use its values
    if len(cols_list) == 1:
        well_metadata["Metadata_JCP2022"] = df[cols_list[0]]
    # For multiple columns, concatenate values with ":" as delimiter
    else:
        well_metadata["Metadata_JCP2022"] = (
            df[cols_list].astype(str).apply(lambda row: ":".join(row), axis=1)
        )

    # Create well metadata DataFrame
    well_df = pd.DataFrame(well_metadata)

    # Check for empty Metadata_JCP2022 values (NULL/NA/NaN values or empty strings)
    empty_jcp2022 = pd.isna(well_df["Metadata_JCP2022"]) | (
        well_df["Metadata_JCP2022"].astype(str).str.strip() == ""
    )

    if empty_jcp2022.any():
        empty_count = empty_jcp2022.sum()

        # Show some examples of the empty values (up to 5)
        empty_examples = well_df[empty_jcp2022].head(5)
        example_str = "\n".join(
            f"  Plate: {row['Metadata_Plate']}, Well: {row['Metadata_Well']}, "
            f"JCP2022 Value: '{row['Metadata_JCP2022']}'"
            for _, row in empty_examples.iterrows()
        )

        warning_msg = (
            f"Found {empty_count} wells with empty Metadata_JCP2022 values in {input_file}.\n"
            f"Examples of wells with empty values:\n{example_str}\n"
            f"These will be replaced with 'UNSPECIFIED'"
        )
        logger.warning(warning_msg)

        # Replace empty values with UNSPECIFIED
        # We intentionally do not replace with JCP2022_UNKNOWN because the current implementation
        # of get_well_metadata filters out JCP2022_UNKNOWN wells.
        well_df.loc[empty_jcp2022, "Metadata_JCP2022"] = "UNSPECIFIED"

    # Save well metadata
    well_df.to_parquet(output_well_metadata_file)
    logger.info(f"Saved well metadata file to: {output_well_metadata_file}")

    # Create plate metadata DataFrame
    # Columns: Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_PlateType

    # Extract batch from path
    batch = extract_batch_from_path(input_file)

    # Verify that there's only one unique plate in the file
    unique_plates = metadata_df["Metadata_Plate"].unique()
    if len(unique_plates) > 1:
        raise click.ClickException(
            f"Multiple plates found in {input_file}: {unique_plates}. Each file should contain data for only one plate."
        )

    # Get the single plate value
    plate = unique_plates[0]

    # Create plate metadata with a single row
    plate_metadata = {
        "Metadata_Source": [source],
        "Metadata_Batch": [batch],
        "Metadata_Plate": [plate],
    }

    # Use Metadata_PlateType from data if it exists, otherwise use default
    if "Metadata_PlateType" in df.columns:
        # Get the first plate type (assuming it's consistent within the file)
        plate_type = df["Metadata_PlateType"].iloc[0]
        plate_metadata["Metadata_PlateType"] = [plate_type]
    else:
        # Use default plate type
        plate_metadata["Metadata_PlateType"] = [default_plate_type]

    # Create plate metadata DataFrame (single row)
    plate_df = pd.DataFrame(plate_metadata)

    # Save plate metadata
    plate_df.to_parquet(output_plate_metadata_file)
    logger.info(f"Saved plate metadata file to: {output_plate_metadata_file}")

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
    jcp2022_cols: str,
    mandatory_feature_cols: Optional[Set[str]] = None,
    continue_on_error: bool = False,
    mandatory_metadata_cols: List[str] = ["Metadata_Plate", "Metadata_Well"],
    default_plate_type: str = "UNKNOWN",
) -> None:
    """Process multiple input files.

    Args:
        input_files: List of input file paths
        output_dir: Directory to save output files
        source: Value to set in Metadata_Source column
        jcp2022_cols: Comma-separated list of columns to be treated as Metadata_JCP2022
        mandatory_feature_cols: Optional set of feature column names that must be included
        continue_on_error: If True, continue processing other files when one fails
        mandatory_metadata_cols: List of required metadata columns to preserve
        default_plate_type: Default value for Metadata_PlateType when not present
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
                input_file=input_file,
                output_dir=output_dir,
                source=source,
                jcp2022_cols=jcp2022_cols,
                mandatory_feature_cols=mandatory_feature_cols,
                mandatory_metadata_cols=mandatory_metadata_cols,
                default_plate_type=default_plate_type,
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

    # Collate all plate and well metadata files into single files
    if (
        total_files > failures
    ):  # Only collate if at least one file was processed successfully
        collate_metadata_files(output_dir)


def collate_metadata_files(output_dir: Path) -> None:
    """
    Collate all plate and well metadata files into single files.

    Args:
        output_dir: Base output directory containing metadata files
    """
    metadata_dir = output_dir / "metadata"

    # Ensure the base metadata directory exists
    if not metadata_dir.exists():
        logger.warning(
            f"No metadata directory found at {metadata_dir}, skipping collation"
        )
        return

    # Collate plate metadata files
    _collate_metadata_type(
        metadata_dir, "plate", ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    )

    # Collate well metadata files
    _collate_metadata_type(
        metadata_dir, "well", ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    )


def _collate_metadata_type(
    metadata_dir: Path, metadata_type: str, dedup_columns: List[str]
) -> None:
    """
    Helper function to collate a specific type of metadata files.

    Args:
        metadata_dir: Base metadata directory
        metadata_type: Type of metadata files to collate ('plate' or 'well')
        dedup_columns: Columns to check for duplicates
    """
    # Find all metadata files of the specified type
    metadata_files = list(metadata_dir.glob(f"**/{metadata_type}.parquet"))

    if not metadata_files:
        logger.warning(f"No {metadata_type} metadata files found to collate")
        return

    logger.info(f"Collating {len(metadata_files)} {metadata_type} metadata files")

    # Load and concatenate all metadata files
    dfs = []
    for file_path in metadata_files:
        try:
            df = pd.read_parquet(file_path)
            dfs.append(df)
        except Exception as e:
            logger.error(
                f"Error reading {metadata_type} metadata file {file_path}: {str(e)}"
            )

    if not dfs:
        logger.warning(f"Failed to read any {metadata_type} metadata files")
        return

    # Concatenate all dataframes
    collated_df = pd.concat(dfs, ignore_index=True)

    # Check for duplicates based on specified columns
    duplicates = collated_df.duplicated(subset=dedup_columns, keep=False)
    if duplicates.any():
        duplicate_rows = collated_df[duplicates].sort_values(by=dedup_columns)
        num_duplicates = len(duplicate_rows) - len(
            duplicate_rows.drop_duplicates(subset=dedup_columns)
        )

        logger.error(
            f"Found {num_duplicates} duplicate entries in {metadata_type} metadata"
        )
        logger.error(f"First few duplicates in {metadata_type} metadata:")
        for i, (_, row) in enumerate(duplicate_rows.iterrows()):
            if i < 5:  # Only show the first 5 duplicates to avoid log spam
                logger.error(f"  {dict(row[dedup_columns])}")
            else:
                break

        logger.error(
            f"Saving full duplicates to {metadata_dir}/{metadata_type}_duplicates.csv for inspection"
        )
        duplicate_rows.to_csv(
            metadata_dir / f"{metadata_type}_duplicates.csv", index=False
        )

        # Continue with collation but log a warning
        logger.warning(f"Collated {metadata_type} metadata will contain duplicates")
    else:
        logger.info(f"No duplicates found in {metadata_type} metadata")

    # Save the collated file (with duplicates intact)
    collated_path = metadata_dir / f"{metadata_type}.parquet"
    collated_df.to_parquet(collated_path)
    logger.info(f"Saved collated {metadata_type} metadata to {collated_path}")


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
@click.option(
    "--jcp2022-cols",
    type=str,
    required=True,
    help="Comma-separated list of columns to be treated as Metadata_JCP2022. If multiple, values will be concatenated with ':' as delimiter. This option is required.",
)
@click.option(
    "--default-plate-type",
    type=str,
    default="UNKNOWN",
    help="Default value for Metadata_PlateType when not present in data",
)
def convert_command(
    file_list: Path,
    output_dir: Path,
    source: str,
    verbose: bool,
    mandatory_feature_cols_file: Optional[Path],
    continue_on_error: bool,
    jcp2022_cols: str,
    mandatory_metadata: str = "Metadata_Plate,Metadata_Well",
    default_plate_type: str = "UNKNOWN",
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

    JCP2022 columns are used to build the Metadata_JCP2022 value in the well metadata.
    This is a required parameter and must specify valid columns in the input files.

    Arguments:
        file_list: Text file containing list of input files (one per line)
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if output directory exists and is not empty
    if output_dir.exists() and any(output_dir.iterdir()):
        raise click.ClickException(
            f"Output directory '{output_dir}' is not empty. Please provide an empty directory."
        )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Log JCP2022 columns if provided
    if jcp2022_cols:
        logger.info(f"Using JCP2022 columns: {jcp2022_cols}")

    # Log default plate type
    logger.info(f"Using default plate type: {default_plate_type}")

    # Process files
    process_files(
        input_files=input_files,
        output_dir=output_dir,
        source=source,
        jcp2022_cols=jcp2022_cols,
        mandatory_feature_cols=mandatory_feature_cols,
        continue_on_error=continue_on_error,
        mandatory_metadata_cols=mandatory_metadata_cols,
        default_plate_type=default_plate_type,
    )


if __name__ == "__main__":
    convert_command()

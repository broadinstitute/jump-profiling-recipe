"""
Functions to load metadata information

File Structure:
- Constants & Configuration: Global constants and configuration variables
- Core Utilities: Basic helper functions for column and path operations
- Metadata Loading: Functions for loading and filtering plate/well data
- Integration: High-level functions that combine multiple operations
"""

import logging
import pandas as pd
import re
from .utils import validate_columns
from collections.abc import Iterable


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------
# Constants & Configuration
# ------------------------------

UNTREATED = "JCP2022_999999"
UNKNOWN = "JCP2022_UNKNOWN"
BAD_CONSTRUCT = "JCP2022_900001"

MICRO_CONFIG = pd.read_csv("inputs/metadata/microscope_config.csv")

MICRO_CONFIG["Metadata_Source"] = "source_" + MICRO_CONFIG["Metadata_Source"].astype(
    str
)
MICRO_CONFIG = MICRO_CONFIG.set_index("Metadata_Source")["Metadata_Microscope_Name"]

POSCON_CODES = [
    "JCP2022_012818",
    "JCP2022_050797",
    "JCP2022_064022",
    "JCP2022_035095",
    "JCP2022_046054",
    "JCP2022_025848",
    "JCP2022_037716",
    "JCP2022_085227",
    "JCP2022_805264",
    "JCP2022_915132",
]

NEGCON_CODES = [
    "JCP2022_800001",
    "JCP2022_800002",
    "JCP2022_033924",
    "JCP2022_915131",
    "JCP2022_915130",
    "JCP2022_915129",
    "JCP2022_915128",
]

SOURCE3_BATCH_REDLIST = {
    "CP_32_all_Phenix1",
    "CP_33_all_Phenix1",
    "CP_34_mix_Phenix1",
    "CP_35_all_Phenix1",
    "CP_36_all_Phenix1",
    "CP59",
    "CP60",
}

# ------------------------------
# Column Classification
# ------------------------------

# Regex patterns for column classification
METADATA_PATTERN = "^(Metadata_)"
FEATURE_PATTERN = "^(?!Metadata_)"


def get_feature_columns(cols: Iterable[str] | pd.DataFrame) -> list[str]:
    """Get column names for features.

    Parameters
    ----------
    cols : Union[Iterable[str], pd.DataFrame]
        Collection of column names or DataFrame to analyze

    Returns
    -------
    Union[list[str], pd.Index]
        Feature column names
        Returns pd.Index if input is DataFrame, list otherwise
    """
    if isinstance(cols, pd.DataFrame):
        return cols.filter(regex=FEATURE_PATTERN).columns.tolist()
    return [c for c in cols if re.match(FEATURE_PATTERN, c)]


def get_metadata_columns(cols: Iterable[str] | pd.DataFrame) -> list[str]:
    """Get column names for metadata.

    Parameters
    ----------
    cols : Union[Iterable[str], pd.DataFrame]
        Collection of column names or DataFrame to analyze

    Returns
    -------
    Union[list[str], pd.Index]
        Metadata column names
        Returns pd.Index if input is DataFrame, list otherwise
    """
    if isinstance(cols, pd.DataFrame):
        return cols.filter(regex=METADATA_PATTERN).columns.tolist()
    return [c for c in cols if re.match(METADATA_PATTERN, c)]


# ------------------------------
# Core Utilities
# ------------------------------


def build_path(row: pd.Series, profile_type: str | None = None) -> str:
    """Create the path to the parquet file.

    Parameters
    ----------
    row : pd.Series
        Row containing metadata information with required fields:
        Metadata_Source, Metadata_Batch, Metadata_Plate.
    profile_type : str | None
        If provided, indicates a deep learning profile type

    Returns
    -------
    str
        Formatted path string to the parquet file.
    """
    required_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    missing_cols = [col for col in required_cols if col not in row.index]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if profile_type:
        template = (
            f"./inputs/profiles_{profile_type}/"
            "{Metadata_Source}/workspace/profiles/"
            "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
        )
    else:
        template = (
            "./inputs/profiles/"
            "{Metadata_Source}/workspace/profiles/"
            "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
        )
    return template.format(**row.to_dict())


def _load_and_concat_data(
    default_csv_path: str,
    additional_parquet_files: list[str] = None,
    description: str = "data",
) -> pd.DataFrame:
    """Helper function to load a CSV file and additional parquet files, then concatenate them.

    Parameters
    ----------
    default_csv_path : str
        Path to the default CSV file to load
    additional_parquet_files : list[str], optional
        List of additional parquet files to load and concatenate
    description : str, optional
        Description of the data being loaded, used in error messages

    Returns
    -------
    pd.DataFrame
        Concatenated data from default CSV and additional parquet files

    Raises
    ------
    ValueError
        If duplicates are found after concatenation
    """
    # Load default CSV data
    data = pd.read_csv(default_csv_path)

    # Load and concatenate additional parquet files if provided
    if additional_parquet_files:
        additional_dfs = [pd.read_parquet(file) for file in additional_parquet_files]
        if additional_dfs:
            additional_dfs.insert(0, data)  # Add default data at the beginning
            data = pd.concat(additional_dfs, ignore_index=True)

            # Check for duplicates after concatenation
            duplicate_mask = data.duplicated(keep=False)
            if duplicate_mask.any():
                raise ValueError(
                    f"Duplicate {description} found after concatenation. "
                    f"There are {duplicate_mask.sum()} duplicated rows."
                )

    return data


# ------------------------------
# Metadata Loading
# ------------------------------


def get_orf_plate_redlist(plate_types: list[str]) -> set[str]:
    """Get set of plate_id's that should not be considered in the analysis.

    Parameters
    ----------
    plate_types : list[str]
        List of plate types to consider.

    Returns
    -------
    set[str]
        Set of plate IDs to exclude from analysis.

    Notes
    -----
    Excludes:
    - Low concentration plates
    - Plates from specific batches (e.g., Batch12)
    """
    # https://github.com/jump-cellpainting/jump-orf-analysis/issues/1#issuecomment-921888625
    # Low concentration plates
    redlist = set(["BR00127147", "BR00127148", "BR00127145", "BR00127146"])
    # https://github.com/jump-cellpainting/aws/issues/70#issuecomment-1182444836
    redlist.add("BR00123528A")

    # filter ORF plates.
    metadata = pd.read_csv("inputs/metadata/experiment-metadata.tsv", sep="\t")
    query = 'Batch=="Batch12"'
    bad_plates = set(metadata.query(query).Assay_Plate_Barcode)
    redlist |= bad_plates
    return redlist


def get_plate_metadata(
    sources: list[str], plate_types: list[str], additional_plate_files: list[str] = None
) -> pd.DataFrame:
    """Create filtered metadata DataFrame from plate-level metadata.

    Loads plate metadata from './inputs/metadata/plate.csv.gz' and applies filtering based on
    source identifiers and plate types. Also performs special filtering for ORF plates and
    source_3 batches.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to include (e.g., ['source_1', 'source_2']).
        Must match values in the Metadata_Source column.
    plate_types : list[str]
        List of plate types to include (e.g., ['ORF', 'CRISPR', 'TARGET2']).
        Must match values in the Metadata_PlateType column.
    additional_plate_files : list[str], optional
        List of paths to additional parquet files containing plate metadata to be loaded
        and concatenated with the default CSV data.

    Returns
    -------
    pd.DataFrame
        Filtered plate metadata containing columns Metadata_Source, Metadata_Plate,
        Metadata_PlateType, and Metadata_Batch.

    Raises
    ------
    ValueError
        If any required columns are missing from the plate metadata file.
        If duplicate rows are found after concatenation.

    Notes
    -----
    Special filtering rules:
    - For ORF plates: Excludes plates in the redlist from get_orf_plate_redlist()
    - For source_3: Excludes batches in SOURCE3_BATCH_REDLIST unless plate type is TARGET2
    """
    # Load and check for duplicates
    plate_metadata = _load_and_concat_data(
        "./inputs/metadata/plate.csv.gz", additional_plate_files, description="plates"
    )

    required_cols = [
        "Metadata_Source",
        "Metadata_Plate",
        "Metadata_PlateType",
        "Metadata_Batch",
    ]
    validate_columns(plate_metadata, required_cols)

    # Filter plates from source_4
    if "ORF" in plate_types:
        redlist = get_orf_plate_redlist(plate_types)
        plate_metadata = plate_metadata[~plate_metadata["Metadata_Plate"].isin(redlist)]

    # Filter plates from source_3 batches without DMSO
    plate_metadata = plate_metadata[
        (~plate_metadata["Metadata_Batch"].isin(SOURCE3_BATCH_REDLIST))
        | (plate_metadata["Metadata_PlateType"] == "TARGET2")
    ]

    plate_metadata = plate_metadata[plate_metadata["Metadata_Source"].isin(sources)]
    plate_metadata = plate_metadata[
        plate_metadata["Metadata_PlateType"].isin(plate_types)
    ]
    return plate_metadata


def get_well_metadata(
    plate_types: list[str], additional_well_files: list[str] = None
) -> pd.DataFrame:
    """Load and process well-level metadata with optional ORF/CRISPR annotations.

    Loads well metadata from './inputs/metadata/well.csv.gz' and optionally merges with
    ORF or CRISPR specific metadata based on plate types.

    Parameters
    ----------
    plate_types : list[str]
        List of plate types to process (e.g., ['ORF', 'CRISPR', 'TARGET2']).
        If 'ORF' is included, merges with ORF metadata from './inputs/metadata/orf.csv.gz'.
        If 'CRISPR' is included, merges with CRISPR metadata from './inputs/metadata/crispr.csv.gz'.
    additional_well_files : list[str], optional
        List of paths to additional parquet files containing well metadata to be loaded
        and concatenated with the default CSV data.

    Returns
    -------
    pd.DataFrame
        Well metadata DataFrame containing at minimum:
        - Metadata_JCP2022: JCP2022 identifier
        Additional columns when ORF/CRISPR metadata is merged.
        Excludes wells with JCP2022 codes: UNTREATED, UNKNOWN, BAD_CONSTRUCT.

    Raises
    ------
    ValueError
        If Metadata_JCP2022 column is missing from the well metadata file.
        If duplicate rows are found after concatenation.

    Notes
    -----
    The function performs left joins when merging ORF/CRISPR metadata, meaning wells
    without matching ORF/CRISPR data will have NULL values in the merged columns.
    """
    # Load and check for duplicates
    well_metadata = _load_and_concat_data(
        "./inputs/metadata/well.csv.gz", additional_well_files, description="wells"
    )

    validate_columns(well_metadata, ["Metadata_JCP2022"])

    if "ORF" in plate_types:
        orf_metadata = pd.read_csv("./inputs/metadata/orf.csv.gz")
        well_metadata = well_metadata.merge(
            orf_metadata, how="left", on="Metadata_JCP2022"
        )
        # well_metadata = well_metadata[well_metadata['Metadata_pert_type']!='poscon']
    if "CRISPR" in plate_types:
        crispr_metadata = pd.read_csv("./inputs/metadata/crispr.csv.gz")
        well_metadata = well_metadata.merge(
            crispr_metadata, how="left", on="Metadata_JCP2022"
        )
    # Filter out wells

    well_metadata = well_metadata[
        ~well_metadata["Metadata_JCP2022"].isin([UNTREATED, UNKNOWN, BAD_CONSTRUCT])
    ]

    return well_metadata


# ------------------------------
# Metadata Integration
# ------------------------------


def load_metadata(
    sources: list[str],
    plate_types: list[str],
    additional_plate_files: list[str] = None,
    additional_well_files: list[str] = None,
) -> pd.DataFrame:
    """Load and merge plate and well metadata.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to include.
    plate_types : list[str]
        List of plate types to include.
    additional_plate_files : list[str], optional
        List of paths to additional parquet files containing plate metadata.
    additional_well_files : list[str], optional
        List of paths to additional parquet files containing well metadata.

    Returns
    -------
    pd.DataFrame
        Merged metadata DataFrame containing both plate and well information,
        filtered according to the specified sources and plate types.
    """
    plate = get_plate_metadata(sources, plate_types, additional_plate_files)
    well = get_well_metadata(plate_types, additional_well_files)
    meta = well.merge(plate, on=["Metadata_Source", "Metadata_Plate"])
    return meta

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
import glob


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
    search_additional_metadata: bool = False,
    sources: list[str] = None,
    metadata_type: str = None,
) -> pd.DataFrame:
    """Helper function to load a CSV file and additional parquet files, then concatenate them.

    Parameters
    ----------
    default_csv_path : str
        Path to the default CSV file to load
    additional_parquet_files : list[str], optional
        List of additional parquet files to load and concatenate
    search_additional_metadata : bool, optional
        If True, automatically search for additional metadata files
    sources : list[str], optional
        Source identifiers to search within if search_additional_metadata is True
    metadata_type : str, optional
        Type of metadata to search for and load ('plate', 'well', or custom type).
        Also used for error messages. Defaults to None if not specified.

    Returns
    -------
    pd.DataFrame
        Concatenated data from default CSV and additional parquet files

    Raises
    ------
    ValueError
        If duplicates are found after concatenation
        If search_additional_metadata is True but required parameters are missing
    """
    # Load default CSV data
    data = pd.read_csv(default_csv_path)

    # Search for additional metadata files if requested
    files_to_load = additional_parquet_files or []

    if search_additional_metadata:
        if not sources:
            raise ValueError(
                "sources must be provided when search_additional_metadata is True"
            )
        if metadata_type is None:
            raise ValueError(
                "metadata_type must be specified when search_additional_metadata is True"
            )

        found_files = find_metadata_files(sources, metadata_type)
        if found_files:
            files_to_load.extend(found_files)

    # Load and concatenate additional parquet files if available
    if files_to_load:
        additional_dfs = [pd.read_parquet(file) for file in files_to_load]
        if additional_dfs:
            additional_dfs.insert(0, data)  # Add default data at the beginning
            data = pd.concat(additional_dfs, ignore_index=True)

            # Check for duplicates after concatenation
            duplicate_mask = data.duplicated(keep=False)
            if duplicate_mask.any():
                raise ValueError(
                    f"Duplicate {metadata_type or 'data'} found after concatenation. "
                    f"There are {duplicate_mask.sum()} duplicated rows."
                )

    return data


def find_metadata_files(sources: list[str], metadata_type: str) -> list[str]:
    """Search for additional metadata files in the profiles directory structure.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to search within (e.g., ['source_1', 'source_2'])
    metadata_type : str
        Type of metadata to find, must be either 'plate' or 'well'

    Returns
    -------
    list[str]
        List of paths to found metadata files

    Notes
    -----
    This function searches for metadata in two locations:
    1. Global metadata: inputs/profiles/{source}/workspace/metadata/{type}.parquet
    2. Plate-specific metadata: inputs/profiles/{source}/workspace/metadata/{batch}/{plate}/{type}.parquet
    """
    if metadata_type not in ["plate", "well"]:
        raise ValueError(
            f"metadata_type must be 'plate' or 'well', got '{metadata_type}'"
        )

    found_files = []

    # Search for global metadata files
    for source in sources:
        global_pattern = (
            f"./inputs/profiles/{source}/workspace/metadata/{metadata_type}.parquet"
        )
        global_files = glob.glob(global_pattern)
        found_files.extend(global_files)

        # Search for plate-specific metadata files
        plate_pattern = (
            f"./inputs/profiles/{source}/workspace/metadata/**/{metadata_type}.parquet"
        )
        plate_files = glob.glob(plate_pattern, recursive=True)
        # Filter out global metadata files that might be found again
        specific_files = [f for f in plate_files if f not in global_files]
        found_files.extend(specific_files)

    logger.info(f"Found {len(found_files)} additional {metadata_type} metadata files")
    return found_files


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
    sources: list[str],
    plate_types: list[str],
    search_additional_metadata: bool = False,
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
    search_additional_metadata : bool, optional
        If True, automatically search for additional metadata files in the profiles directory
        structure based on the given sources.

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
        "./inputs/metadata/plate.csv.gz",
        search_additional_metadata=search_additional_metadata,
        sources=sources,
        metadata_type="plate",
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
    plate_types: list[str],
    search_additional_metadata: bool = False,
    sources: list[str] = None,
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
    search_additional_metadata : bool, optional
        If True, automatically search for additional metadata files in the profiles directory
        structure based on the given sources.
    sources : list[str], optional
        List of source identifiers to search within if search_additional_metadata is True.
        Must be provided if search_additional_metadata is True.

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
        If search_additional_metadata is True but sources is not provided.

    Notes
    -----
    The function performs left joins when merging ORF/CRISPR metadata, meaning wells
    without matching ORF/CRISPR data will have NULL values in the merged columns.
    """
    # Load and check for duplicates
    well_metadata = _load_and_concat_data(
        "./inputs/metadata/well.csv.gz",
        search_additional_metadata=search_additional_metadata,
        sources=sources,
        metadata_type="well",
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
    search_additional_metadata: bool = False,
) -> pd.DataFrame:
    """Load and merge plate and well metadata.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to include.
    plate_types : list[str]
        List of plate types to include.
    search_additional_metadata : bool, optional
        If True, automatically search for additional metadata files in the profiles directory
        structure based on the given sources.

    Returns
    -------
    pd.DataFrame
        Merged metadata DataFrame containing both plate and well information,
        filtered according to the specified sources and plate types.
    """
    plate = get_plate_metadata(
        sources,
        plate_types,
        search_additional_metadata,
    )
    well = get_well_metadata(plate_types, search_additional_metadata, sources)
    meta = well.merge(plate, on=["Metadata_Source", "Metadata_Plate"])
    return meta

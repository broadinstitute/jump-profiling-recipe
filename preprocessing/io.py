"""Functions for data input/output and preprocessing operations.

File Structure:
- DataFrame Column Operations: Functions for splitting data into features and metadata
- File Operations: Functions for reading/writing and handling parquet files
- Data Validation: Functions for checking data quality and reporting issues
- Metadata Annotation: Functions for enriching metadata information
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
import logging
from pathlib import Path

from .metadata import (
    build_path,
    load_metadata,
    MICRO_CONFIG,
    get_feature_columns,
    get_metadata_columns,
    POSCON_CODES,
    NEGCON_CODES,
)
from .utils import validate_columns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------
# Data Validation
# ------------------------------


def report_nan_infs_columns(dframe: pd.DataFrame) -> None:
    """Report columns containing NaN and infinite values.

    Parameters
    ----------
    dframe : pd.DataFrame
        DataFrame to check for NaN and infinite values
    """
    logger.info("Checking for NaN and INF")
    feat_cols = get_feature_columns(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    if withnan.shape[0] > 0:
        logger.info(f"Columns with NaN: {withnan}")
    if withinf.shape[0] > 0:
        logger.info(f"Columns with INF: {withinf}")
    if withninf.shape[0] > 0:
        logger.info(f"Columns with NINF: {withninf}")


# ------------------------------
# DataFrame Column Operations
# ------------------------------


def split_parquet(
    dframe_path: str, features: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split a parquet file into metadata and feature arrays.

    This function:
    1. Loads a parquet file into a DataFrame
    2. Automatically detects feature columns if not specified
    3. Extracts feature values into a numpy array
    4. Separates metadata into a separate DataFrame

    Parameters
    ----------
    dframe_path : str
        Path to the parquet file
    features : list[str] | None, optional
        List of feature column names. If None, features are automatically detected

    Returns
    -------
    meta : pd.DataFrame
        DataFrame containing metadata columns
    vals : np.ndarray
        2D array of feature values with shape (n_samples, n_features)
    features : list[str]
        List of feature column names
    """
    dframe = pd.read_parquet(dframe_path)
    if features is None:
        features = get_feature_columns(dframe)
    else:
        # Validate that specified features exist
        missing_cols = set(features) - set(dframe.columns)
        if missing_cols:
            raise ValueError(f"Features not found in data: {missing_cols}")

    vals = np.empty((len(dframe), len(features)), dtype=np.float32)
    for i, c in enumerate(features):
        vals[:, i] = dframe[c]

    meta_cols = get_metadata_columns(dframe)
    if not meta_cols:
        raise ValueError("No metadata columns found in DataFrame")
    meta = dframe[meta_cols].copy()
    return meta, vals, features


def merge_parquet(
    meta: pd.DataFrame, vals: np.ndarray, features: list[str], output_path: str
) -> None:
    """Merge feature values with metadata and save as a parquet file.

    This function:
    1. Creates a DataFrame from the feature values array
    2. Adds metadata columns to the DataFrame
    3. Reports any NaN/Inf values in the data
    4. Saves the combined DataFrame to a parquet file with reset index

    Parameters
    ----------
    meta : pd.DataFrame
        DataFrame containing metadata columns
    vals : np.ndarray
        2D array of feature values with shape (n_samples, n_features)
    features : list[str]
        List of feature column names
    output_path : str
        Path where to save the parquet file

    Returns
    -------
    None
        The function writes to a file and does not return anything
    """
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    logger.info(f"Saving file {output_path.split('/')[-1]}")
    report_nan_infs_columns(dframe)
    dframe.to_parquet(output_path)


# ------------------------------
# Metadata Annotation
# ------------------------------


def add_pert_type(meta: pd.DataFrame, col: str = "Metadata_pert_type") -> None:
    """Add perturbation type column to metadata.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame to modify
    col : str, optional
        Name of the column to add, by default "Metadata_pert_type"
    """
    validate_columns(meta, ["Metadata_JCP2022"])

    meta[col] = "trt"
    meta.loc[meta["Metadata_JCP2022"].isin(POSCON_CODES), col] = "poscon"
    meta.loc[meta["Metadata_JCP2022"].isin(NEGCON_CODES), col] = "negcon"
    meta[col] = meta[col].astype("category")


def add_row_col(meta: pd.DataFrame) -> None:
    """Add Metadata_Row and Metadata_Column to the DataFrame.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame to modify
    """
    validate_columns(meta, ["Metadata_Well"])

    well_regex = r"^(?P<row>[a-zA-Z]{1,2})(?P<column>[0-9]{1,2})$"
    position = meta["Metadata_Well"].str.extract(well_regex)
    if position["row"].isna().any() or position["column"].isna().any():
        raise ValueError("Invalid well format found in Metadata_Well")

    meta["Metadata_Row"] = position["row"].astype("category")
    meta["Metadata_Column"] = position["column"].astype("category")


def add_microscopy_info(meta: pd.DataFrame) -> None:
    """Add microscopy configuration information to metadata.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame to modify
    """
    validate_columns(meta, ["Metadata_Source"])

    configs = meta["Metadata_Source"].map(MICRO_CONFIG)
    if configs.isna().any():
        missing_sources = meta["Metadata_Source"][configs.isna()].unique()
        raise ValueError(f"Missing microscope config for sources: {missing_sources}")

    meta["Metadata_Microscope"] = configs.astype("category")


# ------------------------------
# File Operations
# ------------------------------


def prealloc_params(
    sources: list[str], plate_types: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Get paths to parquet files and corresponding slices for concatenation.

    This function:
    1. Loads metadata for all specified sources and plate types
    2. Builds paths to each unique parquet file
    3. Calculates slice indices for each file to enable efficient concatenation
       where slices[i] contains [start, end] indices for file i in the final array

    For example, if files have lengths [100, 150, 200]:
    - slices will be [[0, 100], [100, 250], [250, 450]]
    - indicating where each file's data should be placed in the final array

    Parameters
    ----------
    sources : list[str]
        List of data sources
    plate_types : list[str]
        List of plate types

    Returns
    -------
    paths : np.ndarray[str]
        1D array of paths to parquet files, one per unique plate
    slices : np.ndarray
        2D array of slice indices with shape (n_files, 2), where each row is
        [start_idx, end_idx] for positioning that file's data in the final array
    """
    meta = load_metadata(sources, plate_types)
    paths = (
        meta[["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]]
        .drop_duplicates()
        .apply(build_path, axis=1)
    ).values

    # Filter out missing paths
    valid_paths = []
    for path in paths:
        if not Path(path).exists():
            logger.warning(f"Missing file: {path}")
            continue
        valid_paths.append(path)

    if not valid_paths:
        raise ValueError("No valid paths found")

    paths = np.array(valid_paths)

    def get_num_rows(path: str) -> int:
        with pq.ParquetFile(path) as file:
            return file.metadata.num_rows

    counts = thread_map(get_num_rows, paths, leave=False, desc="counts")
    slices = np.zeros((len(paths), 2), dtype=int)
    slices[:, 1] = np.cumsum(counts)
    slices[1:, 0] = slices[:-1, 1]
    return paths, slices


def load_data(sources: list[str], plate_types: list[str]) -> pd.DataFrame:
    """Load all plates given the parameters.

    Parameters
    ----------
    sources : list[str]
        List of data sources
    plate_types : list[str]
        List of plate types

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all plates' data
    """
    paths, slices = prealloc_params(sources, plate_types)
    total = slices[-1, 1]

    with pq.ParquetFile(paths[0]) as f:
        meta_cols = get_metadata_columns(f.schema.names)
        feat_cols = get_feature_columns(f.schema.names)
    meta = np.empty([total, len(meta_cols)], dtype="|S128")
    feats = np.empty([total, len(feat_cols)], dtype=np.float32)

    def read_parquet(params):
        path, start, end = params
        df = pd.read_parquet(path)

        meta[int(start) : int(end)] = df[meta_cols].values
        feats[int(start) : int(end)] = df[feat_cols].values

    params = np.concatenate([paths[:, None], slices], axis=1)
    thread_map(read_parquet, params)

    meta = pd.DataFrame(data=meta.astype(str), columns=meta_cols, dtype="category")
    dframe = pd.DataFrame(columns=feat_cols, data=feats)
    for col in meta_cols:
        dframe[col] = meta[col]
    return dframe


def write_parquet(sources: list[str], plate_types: list[str], output_file: str) -> None:
    """Write a combined and preprocessed parquet dataset from multiple source plates.

    This function:
    1. Loads and combines data from multiple plates
    2. Removes Image_ features
    3. Adds metadata annotations (perturbation type, row/column info, microscopy info)
    4. Filters out samples with missing JCP2022 metadata
    5. Converts metadata columns to categorical type
    6. Writes the final DataFrame to a single parquet file

    Parameters
    ----------
    sources : list[str]
        List of data sources
    plate_types : list[str]
        List of plate types
    output_file : str
        Path where to save the output parquet file
    """
    dframe = load_data(sources, plate_types)

    # Drop Image features
    image_col = [col for col in dframe.columns if "Image_" in col]
    dframe.drop(image_col, axis=1, inplace=True)

    # Get metadata
    meta = load_metadata(sources, plate_types)
    add_pert_type(meta)
    add_row_col(meta)
    add_microscopy_info(meta)

    required_cols = [
        "Metadata_Source",
        "Metadata_Plate",
        "Metadata_Well",
    ]
    validate_columns(dframe, required_cols)

    foreign_key = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    meta = dframe[foreign_key].merge(meta, on=foreign_key, how="left")

    # Dropping samples with no metadata
    jcp_col = meta.pop("Metadata_JCP2022").astype("category")
    dframe["Metadata_JCP2022"] = jcp_col
    dframe.dropna(subset=["Metadata_JCP2022"], inplace=True)
    meta = meta[~jcp_col.isna()].copy()
    assert (meta.index == dframe.index).all()

    for c in meta:
        dframe[c] = meta[c].astype("category")

    dframe.reset_index(drop=True, inplace=True)
    dframe.to_parquet(output_file)

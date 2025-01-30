"""
Functions for formatting profiles for public release
"""

import os
import pandas as pd
import numpy as np
import logging
from preprocessing import io

logger = logging.getLogger(__name__)


def create_merged_dataframe(
    meta: pd.DataFrame, vals: np.ndarray, features: list[str]
) -> pd.DataFrame:
    """Merge metadata and feature values into a single DataFrame.

    Parameters
    ----------
    meta : pd.DataFrame
        DataFrame containing metadata columns
    vals : np.ndarray
        Array containing feature values
    features : list[str]
        List of feature column names

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with metadata and feature columns
    """
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    meta_col = list(meta.columns)
    dframe = dframe[meta_col + features]
    return dframe


def restrict_column_type(input_path: str, meta_col_new=None) -> pd.DataFrame:
    """Cast metadata columns to string and feature columns to float32.

    Parameters
    ----------
    input_path : str
        Path to input parquet file
    meta_col_new : list[str], optional
        List of metadata columns to keep, by default None which uses predefined columns

    Returns
    -------
    pd.DataFrame
        DataFrame with properly typed columns
    """
    meta, feat_val, feat_col = io.split_parquet(input_path)
    if meta_col_new is None:
        meta_col_new = [
            "Metadata_Source",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_JCP2022",
        ]
    meta = meta[meta_col_new]
    for c in meta:
        meta[c] = meta[c].astype("string")
    feat_val = feat_val.astype("float32")
    feat_col = [str(c) for c in feat_col]
    dframe = create_merged_dataframe(meta, feat_val, feat_col)
    return dframe


def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize feature names for PCA or harmony transformed profiles.

    Renames feature columns to X_1, X_2, etc., while preserving metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature columns to be renamed

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized feature column names
    """
    feat_col = [col for col in df.columns if "Metadata" not in col]
    feat_col_rename = ["X_" + str(i + 1) for i in range(len(feat_col))]

    df.rename(columns=dict(zip(feat_col, feat_col_rename)), inplace=True)
    return df


def run_format_check(profile_dir: str):
    """Process profile files to ensure consistent data types and column names.

    Reads parquet files from the input directory that contain 'profiles' in their name,
    applies type restrictions to metadata and feature columns, and optionally standardizes
    feature column names for PCA/harmony files. Processed files are saved to a new directory
    named '{scenario}_public/'.

    Parameters
    ----------
    profile_dir : str
        Directory containing profile parquet files to process. The directory name is used
        to create the output directory path.

    Notes
    -----
    - Creates an output directory at 'outputs/{scenario}_public/' if it doesn't exist
    - Only processes files with '.parquet' extension and 'profiles' in the filename
    - For files containing 'harmony' or 'PCA' in their name, feature columns are
      renamed to X_1, X_2, etc.
    """
    logger.info(f"Starting profile format check for directory: {profile_dir}")

    files = os.listdir(profile_dir)
    files = [file for file in files if (".parquet" in file) and ("profiles" in file)]
    logger.info(f"Found {len(files)} profile files to process")

    output_dir = (
        "outputs/" + profile_dir.split("/")[1] + "_public/"
    )  # Save new profiles to new folder "{scenario}_public/"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for file in files:
        file_path = os.path.join(profile_dir, file)
        file_output_path = os.path.join(output_dir, file)
        df = restrict_column_type(file_path)
        if ("harmony" in file) or ("PCA" in file):
            df = standardize_col_names(df)
        df.to_parquet(file_output_path)

    logger.info("Profile format check completed successfully")

"""Casting column types and standardize column names in profiles for public release"""

import os
import pandas as pd
import numpy as np
from preprocessing import io


def merge_parquet(
    meta: pd.DataFrame, vals: np.ndarray, features: list[str]
) -> pd.DataFrame:
    """Save the data in a parquet file resetting the index"""
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    meta_col = list(meta.columns)
    dframe = dframe[meta_col + features]
    return dframe


def restrict_column_type(input_path: str, meta_col_new=None) -> pd.DataFrame:
    """
    Casting all metadata columns to string and feature columns to float32
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
    dframe = merge_parquet(meta, feat_val, feat_col)
    return dframe


def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize feature names for PCA or harmony transformed profiless
    """
    feat_col = [col for col in df.columns if "Metadata" not in col]
    feat_col_rename = ["X_" + str(i + 1) for i in range(len(feat_col))]

    df.rename(columns=dict(zip(feat_col, feat_col_rename)), inplace=True)
    return df


def run_format_check(profile_dir: str):
    """
    Ensure profiles have expected data types and column names
    """
    files = os.listdir(profile_dir)
    files = [file for file in files if (".parquet" in file) and ("profiles" in file)]
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

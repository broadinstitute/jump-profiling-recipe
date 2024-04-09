'''Casting column types and standardize column names in profiles for public release'''
import os
import pandas as pd
from preprocessing import io


def restrict_column_type(input_path, output_path):
    """
    Casting all metadata columns to string and feature columns to float32
    """
    meta, feat_val, feat_col = io.split_parquet(input_path)
    for c in meta:
        meta[c] = meta[c].astype("string")
    feat_val = feat_val.astype("float32")
    feat_col = [str(c) for c in feat_col]
    io.merge_parquet(meta, feat_val, feat_col, output_path)


def standardize_col_names(input_path, output_path):
    """
    Standardize feature names for PCA or harmony transformed profiless
    """
    df = pd.read_parquet(input_path)
    feat_col = [col for col in df.columns if "Metadata" not in col]
    feat_col_rename = ["X_" + str(i + 1) for i in range(len(feat_col))]

    df.rename(columns=dict(zip(feat_col, feat_col_rename)), inplace=True)
    df.to_parquet(output_path, index=False)


def run_format_check(profile_dir, output_path):
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
        restrict_column_type(file_path, file_output_path)
        if ("harmony" in file) or ("PCA" in file):
            standardize_col_names(file_path, file_output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Profiles format checked. New profiles saved at {output_dir}")

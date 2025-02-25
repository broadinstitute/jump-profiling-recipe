#!/usr/bin/env python3
"""
This script copies over selected Parquet files from the production inputs and
filters their columns based on the source:
For all profiles:
- All columns that start with "Metadata"
Plus:
- For deep learning profiles (containing 'cpcnn' or 'efficientnet'):
  the first 3 non-Metadata columns
- For other profiles: columns starting with "Cells_AreaShape_Zernike5"
The resulting files are stored in the tests/fixtures/ directory, maintaining the
same directory structure as inputs/.
"""

import os
import pandas as pd


def is_deep_learning_profile(filepath):
    """
    Check if the filepath is from a deep learning model's output
    """
    dl_identifiers = ["cpcnn", "efficientnet"]
    return any(identifier in filepath.lower() for identifier in dl_identifiers)


def filter_columns(df, filepath):
    """
    Given a DataFrame and filepath, return a new DataFrame containing:
    - For all profiles: all Metadata columns
    Plus:
    - For deep learning profiles: first 3 non-Metadata columns
    - For other profiles: columns starting with 'Cells_AreaShape_Zernike_5'
    """
    # Always include all Metadata columns
    metadata_cols = [col for col in df.columns if col.startswith("Metadata")]
    
    # Get additional columns based on profile type
    if is_deep_learning_profile(filepath):
        additional_cols = [col for col in df.columns if not col.startswith("Metadata")][:3]
    else:
        additional_cols = [col for col in df.columns if col.startswith("Cells_AreaShape_Zernike_5")]
    
    filtered_cols = metadata_cols + additional_cols
    return df[filtered_cols]


def main():
    # Get repository root (assumes script is in tests/scripts/ directory)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # List of production parquet files to be processed
    source_files = [
        "inputs/profiles/source_13/workspace/profiles/20220914_Run1/CP-CC9-R1-04/CP-CC9-R1-04.parquet",
        "inputs/profiles/source_13/workspace/profiles/20221009_Run2/CP-CC9-R2-04/CP-CC9-R2-04.parquet",
        "inputs/profiles/source_4/workspace/profiles/2021_04_26_Batch1/BR00117037/BR00117037.parquet",
        "inputs/profiles/source_4/workspace/profiles/2021_04_26_Batch1/BR00117038/BR00117038.parquet",
        "inputs/profiles/source_5/workspace/profiles/JUMPCPE-20210628-Run03_20210629_064133/APTJUM122/APTJUM122.parquet",
        "inputs/profiles/source_5/workspace/profiles/JUMPCPE-20210903-Run27_20210904_215148/APTJUM422/APTJUM422.parquet",
        "inputs/profiles_cpcnn_zenodo_7114558/source_4/workspace/profiles/2021_04_26_Batch1/BR00117037/BR00117037.parquet",
        "inputs/profiles_cpcnn_zenodo_7114558/source_4/workspace/profiles/2021_04_26_Batch1/BR00117038/BR00117038.parquet",
    ]

    # Convert source paths to absolute paths
    source_files = [os.path.join(repo_root, src) for src in source_files]

    for src in source_files:
        if not os.path.exists(src):
            print(f"Source file {src} does not exist. Skipping.")
            continue

        print(f"Processing {src}")
        # Read the production parquet file
        df = pd.read_parquet(src)

        # Filter the dataframe columns
        filtered_df = filter_columns(df, src)

        # Create the same relative path structure in the fixtures directory
        rel_path = os.path.relpath(src, os.path.join(repo_root, "inputs"))
        dest_file = os.path.join(repo_root, "tests", "fixtures", "inputs", rel_path)

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # Write the filtered DataFrame to a new parquet file
        filtered_df.to_parquet(dest_file)
        print(f"Written filtered file to {dest_file}")


if __name__ == "__main__":
    main()

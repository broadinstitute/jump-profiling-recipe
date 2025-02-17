#!/usr/bin/env python3
"""
This script copies over selected Parquet files from the production inputs and
filters them so that only the columns whose names start with "Metadata" or
"Cells_AreaShape_Zernike5" remain. The resulting files are stored in the
tests/fixtures/ directory, maintaining the same directory structure as inputs/.
"""

import os
import pandas as pd


def filter_columns(df):
    """
    Given a DataFrame, return a new DataFrame containing only columns that start
    with 'Metadata' or 'Cells_AreaShape_Zernike5'.
    """
    filtered_cols = [
        col
        for col in df.columns
        if col.startswith("Metadata") or col.startswith("Cells_AreaShape_Zernike5")
    ]
    return df[filtered_cols]


def main():
    # Get repository root (assumes script is in tests/scripts/ directory)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # List of production parquet files to be processed
    source_files = [
        "inputs/source_13/workspace/profiles/20220914_Run1/CP-CC9-R1-04/CP-CC9-R1-04.parquet",
        "inputs/source_13/workspace/profiles/20221009_Run2/CP-CC9-R2-04/CP-CC9-R2-04.parquet",
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
        filtered_df = filter_columns(df)

        # Create the same relative path structure in the fixtures directory
        rel_path = os.path.relpath(src, os.path.join(repo_root, "inputs"))
        dest_file = os.path.join(repo_root, "tests", "fixtures", rel_path)

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # Write the filtered DataFrame to a new parquet file
        filtered_df.to_parquet(dest_file)
        print(f"Written filtered file to {dest_file}")


if __name__ == "__main__":
    main()

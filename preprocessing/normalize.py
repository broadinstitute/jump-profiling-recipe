"""
Functions for normalizing feature data
"""

import numpy as np
import pandas as pd
from preprocessing.io import merge_parquet, split_parquet, _validate_columns


def mad(variant_feats_path: str, norm_stats_path: str, normalized_path: str) -> None:
    """Perform MAD (Median Absolute Deviation) normalization on feature data.

    Parameters
    ----------
    variant_feats_path : str
        Path to the parquet file containing variant features data.
    norm_stats_path : str
        Path to the parquet file containing normalization statistics.
    normalized_path : str
        Output path for the normalized data.

    Notes
    -----
    The function performs plate-wise MAD normalization using the formula:
    normalized_value = (value - median) / mad

    The normalization is done in-place to save memory.
    """
    meta, vals, features = split_parquet(variant_feats_path)

    # Validate required columns are present
    _validate_columns(meta, ["Metadata_Plate"])

    # Load and filter normalization statistics to match our features
    norm_stats = pd.read_parquet(norm_stats_path)
    norm_stats = norm_stats.query("feature in @features")
    _validate_columns(norm_stats, ["feature", "mad", "median", "Metadata_Plate"])

    # Sort data by plate for efficient processing and broadcasting
    plates, counts = np.unique(meta["Metadata_Plate"], return_counts=True)
    ix = np.argsort(meta["Metadata_Plate"])
    meta = meta.iloc[ix]
    vals = vals[ix]

    # Create matrices of MAD and median values aligned with our plates and features
    # Shape: (n_plates, n_features)
    mads = norm_stats.pivot(index="Metadata_Plate", columns="feature", values="mad")
    mads = mads.loc[plates, features].values
    medians = norm_stats.pivot(
        index="Metadata_Plate", columns="feature", values="median"
    )
    medians = medians.loc[plates, features].values

    # Apply MAD normalization using broadcasting
    # np.repeat expands the stats to match the number of samples per plate
    # Shape: (n_samples, n_features)
    vals -= np.repeat(medians, counts, axis=0)  # Center data
    vals /= np.repeat(mads, counts, axis=0)  # Scale data

    merge_parquet(meta, vals, features, normalized_path)

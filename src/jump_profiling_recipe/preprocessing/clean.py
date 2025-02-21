"""
Functions for generic numerical data cleaning and outlier handling.

This module provides utilities for cleaning numerical feature data without any
domain-specific knowledge.

For biology-specific corrections and annotations, see correct/corrections.py
"""

import pandas as pd
from .metadata import get_feature_columns
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clip_features(dframe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Clip feature values to a given magnitude.

    Parameters
    ----------
    dframe : pd.DataFrame
        Input DataFrame containing features to be clipped.
    threshold : float
        Maximum absolute value allowed for features. Values outside
        [-threshold, threshold] will be clipped.

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped feature values.
    """
    feat_cols = get_feature_columns(dframe.columns)
    counts = (dframe.loc[:, feat_cols].abs() > threshold).sum()[lambda x: x > 0]
    if len(counts) > 0:
        logger.info(f"Clipping {counts.sum()} values in {len(counts)} columns")
        dframe.loc[:, feat_cols].clip(-threshold, threshold, inplace=True)
    return dframe


def drop_outlier_features(
    dframe: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, int]:
    """Remove columns with 1 percentile of absolute values larger than threshold.

    Parameters
    ----------
    dframe : pd.DataFrame
        Input DataFrame containing features to analyze.
    threshold : float
        Maximum allowed value for the 99th percentile of absolute feature values.

    Returns
    -------
    tuple
        - pd.DataFrame: DataFrame with outlier features removed
        - int: Number of columns removed due to large values
    """
    feat_cols = get_feature_columns(dframe.columns)
    large_feat = dframe[feat_cols].abs().quantile(0.99) > threshold
    large_feat = set(large_feat[large_feat].index)
    keep_cols = [c for c in dframe.columns if c not in large_feat]
    num_ignored = dframe.shape[1] - len(keep_cols)
    logger.info(f"{num_ignored} ignored columns due to large values")
    dframe = dframe[keep_cols]
    return dframe, num_ignored


def remove_outliers(input_path: str, output_path: str) -> None:
    """Remove outliers from features in a parquet file.

    Parameters
    ----------
    input_path : str
        Path to input parquet file containing the DataFrame.
    output_path : str
        Path where the cleaned DataFrame will be saved as parquet.

    Returns
    -------
    None
        Writes cleaned DataFrame to output_path.

    Notes
    -----
    This function performs two cleaning steps:
    1. Removes features where the 99th percentile exceeds 1e2
    2. Clips remaining feature values to [-1e2, 1e2]
    """
    dframe = pd.read_parquet(input_path)
    dframe, _ = drop_outlier_features(dframe, threshold=1e2)
    dframe = clip_features(dframe, threshold=1e2)
    dframe.to_parquet(output_path)

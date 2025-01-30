"""
Functions for computing statistics

File Structure:
- Feature & Statistics Operations: Functions for computing descriptive statistics and plate-wise statistics
- Feature Selection & Filtering: Functions for removing NaN/Inf features and selecting variant features
- Metadata Operations: Functions for augmenting stats DataFrame with metadata columns
"""

from functools import partial
from itertools import chain
import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from tqdm.contrib.concurrent import thread_map
from preprocessing.io import merge_parquet, _validate_columns
from .metadata import find_feat_cols, find_meta_cols, NEGCON_CODES

logger = logging.getLogger(__name__)

# ------------------------------
# DataFrame Feature & Statistics Operations
# ------------------------------


def get_feat_stats(
    dframe: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate descriptive statistics (e.g. mean, std, min, max, quartiles, and IQR) for each feature
    column in a DataFrame.

    Parameters
    ----------
    dframe : pd.DataFrame
        The input DataFrame containing feature columns and possibly metadata columns.
    features : list of str, optional
        The names of the feature columns for which to compute statistics.
        If None, automatically determine feature columns using `find_feat_cols`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics for each specified feature, including an
        additional "iqr" column.
    """
    if features is None:
        features = find_feat_cols(dframe)
    desc = thread_map(lambda x: dframe[x].describe(), features, leave=False)
    desc = pd.DataFrame(desc)
    desc["iqr"] = desc["75%"] - desc["25%"]
    return desc


def get_plate_stats(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute plate-wise statistics (median, median absolute deviation, min, max, and count)
    for features in a DataFrame grouped by the "Metadata_Plate" column.

    Parameters
    ----------
    dframe : pd.DataFrame
        The input DataFrame containing feature columns and a "Metadata_Plate" column.

    Returns
    -------
    pd.DataFrame
        A DataFrame of plate-wise statistics in a long-to-wide pivoted format, including
        an additional "abs_coef_var" column (absolute coefficient of variation).
    """
    _validate_columns(dframe, ["Metadata_Plate"])

    mad_fn = partial(median_abs_deviation, nan_policy="omit", axis=0)
    # scale param reproduces pycytominer output. Differences in mAP for
    # Target 2 in scenario 2 are negligible.
    # mad_fn = partial(median_abs_deviation, nan_policy="omit", axis=0, scale=1 / 1.4826)

    feat_cols = find_feat_cols(dframe)
    dframe = dframe[feat_cols + ["Metadata_Plate"]]
    median = dframe.groupby("Metadata_Plate", observed=True).median()
    max_ = dframe.groupby("Metadata_Plate", observed=True).max()
    min_ = dframe.groupby("Metadata_Plate", observed=True).min()
    count = dframe.groupby("Metadata_Plate", observed=True).count()
    mad = dframe.groupby("Metadata_Plate", observed=True).apply(mad_fn)
    mad = pd.DataFrame(index=mad.index, data=np.stack(mad.values), columns=feat_cols)

    median["stat"] = "median"
    mad["stat"] = "mad"
    min_["stat"] = "min"
    max_["stat"] = "max"
    count["stat"] = "count"

    stats = pd.concat([median, mad, min_, max_, count])
    stats.reset_index(inplace=True)
    stats = stats.melt(id_vars=["Metadata_Plate", "stat"], var_name="feature")
    stats = stats.pivot(
        index=["Metadata_Plate", "feature"], columns="stat", values="value"
    )
    stats.reset_index(inplace=True)
    stats["abs_coef_var"] = (
        (stats["mad"] / stats["median"]).fillna(0).abs().replace(np.inf, 0)
    )
    stats = stats.astype(
        {
            "min": np.float32,
            "max": np.float32,
            "count": np.int32,
            "median": np.float32,
            "mad": np.float32,
            "abs_coef_var": np.float32,
            "feature": "category",
        }
    )
    return stats


def compute_stats(parquet_path: str, stats_path: str) -> None:
    """
    Load a parquet file, compute per-feature descriptive statistics, and save them to another
    parquet file.

    Parameters
    ----------
    parquet_path : str
        The file path to the input parquet data.
    stats_path : str
        The file path where the resulting stats parquet file will be saved.

    Returns
    -------
    None
        Writes the per-feature stats DataFrame to the specified parquet file.
    """
    dframe = pd.read_parquet(parquet_path)
    fea_stats = get_feat_stats(dframe)
    fea_stats.to_parquet(stats_path)


def compute_norm_stats(parquet_path: str, df_stats_path: str, use_negcon: bool) -> None:
    """
    Load a parquet file, remove columns containing NaN or infinite values, and compute plate-wise
    statistics for either negative control rows or all rows.

    Parameters
    ----------
    parquet_path : str
        The file path to the input parquet data.
    df_stats_path : str
        The file path where the resulting stats parquet file will be saved.
    use_negcon : bool
        If True, compute stats only on rows whose "Metadata_JCP2022" value is in NEGCON_CODES.
        Otherwise, compute stats on all rows.

    Returns
    -------
    None
        Writes the computed plate-wise stats DataFrame to the specified parquet file.
    """
    logger.info("Loading data")
    dframe = pd.read_parquet(parquet_path)
    if use_negcon:
        _validate_columns(dframe, ["Metadata_JCP2022"])
    logger.info("Removing nan and inf columns")
    dframe = remove_nan_infs_columns(dframe)
    if use_negcon:
        dframe_norm = dframe[dframe["Metadata_JCP2022"].isin(NEGCON_CODES)]
        logger.info("computing plate stats for negcons")
    else:
        dframe_norm = dframe
        logger.info("computing plate stats for all treatments")
    dframe_stats = get_plate_stats(dframe_norm)
    logger.info("stats done.")
    add_metadata(dframe_stats, dframe[find_meta_cols(dframe)])
    dframe_stats.to_parquet(df_stats_path)


# ------------------------------
# Feature Selection & Filtering
# ------------------------------


def remove_nan_infs_columns(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that contain NaN or infinite values from a DataFrame.

    Parameters
    ----------
    dframe : pd.DataFrame
        The input DataFrame in which columns containing NaN or infinite values will be dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame excluding all columns that contained NaN or infinite values.
    """
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    logger.warning(f"Dropping {len(redlist)} NaN/INF features.")
    return dframe[[c for c in dframe.columns if c not in redlist]]


def select_variant_features(
    parquet_path: str, norm_stats_path: str, variant_feats_path: str
) -> None:
    """
    Filter out features that have zero MAD or an absolute coefficient of variation below 1e-3
    in any plate. Only keeps plates containing these "variant" features.

    Parameters
    ----------
    parquet_path : str
        The file path to the input parquet data.
    norm_stats_path : str
        The file path to the stats parquet file used for filtering (computed typically on negative controls).
    variant_feats_path : str
        The output parquet path of the DataFrame containing only variant features.

    Returns
    -------
    None
        Writes the filtered DataFrame to the specified parquet path, containing only
        variant features across all relevant plates.
    """
    dframe = pd.read_parquet(parquet_path)
    norm_stats = pd.read_parquet(norm_stats_path)

    # Remove NaN and Inf
    dframe = remove_nan_infs_columns(dframe)

    # Select variant_features
    norm_stats = norm_stats.query("mad!=0 and abs_coef_var>1e-3")
    groups = norm_stats.groupby("Metadata_Plate", observed=True)["feature"]
    variant_features = set.intersection(*groups.agg(set).tolist())

    # Select plates with variant features
    norm_stats = norm_stats.query("feature in @variant_features")
    dframe = dframe.query("Metadata_Plate in @norm_stats.Metadata_Plate")

    # Filter features
    variant_features = sorted(variant_features)
    meta = dframe[find_meta_cols(dframe)]
    vals = dframe[variant_features].values
    merge_parquet(meta, vals, variant_features, variant_feats_path)


# ------------------------------
# Metadata Operations
# ------------------------------


def add_metadata(stats: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Augment the stats DataFrame with metadata columns by
    - Mapping the source of each plate from the meta DataFrame.
    - Extracting compartment information from feature column names.

    Parameters
    ----------
    stats : pd.DataFrame
        The stats DataFrame produced by get_plate_stats or similar functions,
        containing "Metadata_Plate" and "feature" columns.
    meta : pd.DataFrame
        A DataFrame containing metadata columns, including "Metadata_Source" and
        "Metadata_Plate".

    Returns
    -------
    None
        This function modifies the stats DataFrame in place by adding new columns.
    """
    _validate_columns(stats, ["Metadata_Plate", "feature"])
    _validate_columns(meta, ["Metadata_Source", "Metadata_Plate"])

    source_map = meta[["Metadata_Source", "Metadata_Plate"]].drop_duplicates()
    source_map = source_map.set_index("Metadata_Plate").Metadata_Source
    stats["Metadata_Source"] = stats["Metadata_Plate"].map(source_map)
    parts = stats["feature"].str.split("_", expand=True)
    stats["compartment"] = parts[0].astype("category")


# stats['family'] = parts[range(3)].apply('_'.join, axis=1).astype('category')

from functools import partial
from itertools import chain
import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from tqdm.contrib.concurrent import thread_map
from preprocessing.io import merge_parquet

from .metadata import find_feat_cols, find_meta_cols, NEGCON_CODES

# logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s', level=logging.WARN)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


def get_feat_stats(dframe: pd.DataFrame, features=None):
    """Get statistics per each feature"""
    if features is None:
        features = find_feat_cols(dframe)
    desc = thread_map(lambda x: dframe[x].describe(), features, leave=False)
    desc = pd.DataFrame(desc)
    desc["iqr"] = desc["75%"] - desc["25%"]
    return desc


def get_plate_stats(dframe: pd.DataFrame):
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


def add_metadata(stats: pd.DataFrame, meta: pd.DataFrame):
    if "Metadata_Source" in meta.columns:
        source_map = meta[["Metadata_Source", "Metadata_Plate"]].drop_duplicates()
        source_map = source_map.set_index("Metadata_Plate").Metadata_Source
        stats["Metadata_Source"] = stats["Metadata_Plate"].map(source_map)
    parts = stats["feature"].str.split("_", expand=True)
    stats["compartment"] = parts[0].astype("category")


# stats['family'] = parts[range(3)].apply('_'.join, axis=1).astype('category')


def remove_nan_infs_columns(dframe: pd.DataFrame) -> pd.DataFrame:
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    logger.warning(f"Dropping {len(redlist)} NaN/INF features.")
    return dframe[[c for c in dframe.columns if c not in redlist]]


def compute_norm_stats(parquet_path, df_stats_path, use_negcon):
    """create platewise statistics for columns without nan/inf values only"""
    logger.info("Loading data")
    dframe = pd.read_parquet(parquet_path)
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


def select_variant_features(parquet_path, norm_stats_path, variant_feats_path):
    """
    Filtered out features that have mad == 0 or abs_coef_var>1e-3 in any plate.
    stats are computed using negative controls only
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


def compute_stats(parquet_path, stats_path):
    dframe = pd.read_parquet(parquet_path)
    fea_stats = get_feat_stats(dframe)
    fea_stats.to_parquet(stats_path)

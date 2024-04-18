import logging

import pandas as pd

from pycytominer.operations import correlation_threshold, variance_threshold

from .metadata import find_feat_cols

logger = logging.getLogger(__name__)


def select_features(dframe_path, feat_selected_path, keep_image_features):
    """Run feature selection"""
    dframe = pd.read_parquet(dframe_path)
    features = find_feat_cols(dframe.columns)
    low_variance = variance_threshold(dframe, features)
    features = [f for f in features if f not in low_variance]
    logger.info(f"{len(low_variance)} features removed by variance_threshold")
    high_corr = correlation_threshold(dframe, features)
    features = [f for f in features if f not in high_corr]
    logger.info(f"{len(high_corr)} features removed by correlation_threshold")

    dframe.drop(columns=low_variance + high_corr, inplace=True)

    cols = find_feat_cols(dframe.columns)
    with open("blocklist_features.txt", "r") as fpointer:
        blocklist = fpointer.read().splitlines()[1:]
    blocklist = [c for c in cols if c in blocklist]
    dframe.drop(columns=blocklist, inplace=True)
    logger.info(f"{len(blocklist)} features removed by blocklist")

    if not keep_image_features:
        cols = find_feat_cols(dframe.columns)
        img_features = [c for c in cols if c.startswith("Image")]
        dframe.drop(columns=img_features, inplace=True)
        logger.info(f"{len(img_features)} Image features removed")

    cols = find_feat_cols(dframe.columns)
    nan_cols = [c for c in cols if dframe[c].isna().any()]
    dframe.drop(columns=nan_cols, inplace=True)
    logger.info(f"{len(nan_cols)} features removed due to NaN values")

    dframe.reset_index(drop=True).to_parquet(feat_selected_path)

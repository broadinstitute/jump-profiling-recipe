import logging

import pandas as pd
from pycytominer.operations import correlation_threshold, variance_threshold
from pycytominer.feature_select import feature_select

from .metadata import find_feat_cols

logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


def select_features(dframe_path, feat_selected_path):
    '''Run feature selection'''
    dframe = pd.read_parquet(dframe_path)
    features = find_feat_cols(dframe.columns)
    low_variance = variance_threshold(dframe, features)
    features = [f for f in features if f not in low_variance]
    logger.info(f'{len(low_variance)} features removed by variance_threshold')
    high_corr = correlation_threshold(dframe, features)
    features = [f for f in features if f not in high_corr]
    logger.info(f'{len(high_corr)} features removed by correlation_threshold')

    dframe.drop(columns=low_variance + high_corr, inplace=True)

    cols = dframe.columns
    dframe = feature_select(dframe, operation='blocklist', image_features=False)
    diff_cols = [f for f in cols if f not in dframe.columns]
    logger.info(f'{len(diff_cols)} features removed by blocklist')

    cols = dframe.columns
    dframe = feature_select(dframe, operation='drop_na_columns', image_features=False)
    diff_cols = [f for f in cols if f not in dframe.columns]
    logger.info(f'{len(diff_cols)} features removed by drop_na_columns')

    dframe.reset_index(drop=True).to_parquet(feat_selected_path)

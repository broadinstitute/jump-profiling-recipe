import pandas as pd
from .metadata import find_feat_cols
import logging

# logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s', level=logging.WARN)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


def clip_features(dframe, threshold):
    """Clip feature values to a given magnitude"""
    feat_cols = find_feat_cols(dframe.columns)
    counts = (dframe.loc[:, feat_cols].abs() > threshold).sum()[lambda x: x > 0]
    if len(counts) > 0:
        logger.info(f"Clipping {counts.sum()} values in {len(counts)} columns")
        dframe.loc[:, feat_cols].clip(-threshold, threshold, inplace=True)
    return dframe


def drop_outlier_feats(dframe: pd.DataFrame, threshold: float):
    """Remove columns with 1 percentile of absolute values larger than threshold"""
    feat_cols = find_feat_cols(dframe.columns)
    large_feat = dframe[feat_cols].abs().quantile(0.99) > threshold
    large_feat = set(large_feat[large_feat].index)
    keep_cols = [c for c in dframe.columns if c not in large_feat]
    num_ignored = dframe.shape[1] - len(keep_cols)
    logger.info(f"{num_ignored} ignored columns due to large values")
    dframe = dframe[keep_cols]
    return dframe, num_ignored


def outlier_removal(input_path: str, output_path: str):
    """Remove outliers"""
    dframe = pd.read_parquet(input_path)
    dframe, _ = drop_outlier_feats(dframe, threshold=1e2)
    dframe = clip_features(dframe, threshold=1e2)
    dframe.to_parquet(output_path)

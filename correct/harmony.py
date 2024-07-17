"""Correction methods"""

import logging

from harmonypy import run_harmony

from preprocessing import io

# logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s', level=logging.WARN)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


def harmony(dframe_path, batch_key, output_path):
    """Harmony correction"""
    meta, feats, features = io.split_parquet(dframe_path)

    # Remove unused categories to avoid matmul dimension mismatch
    meta_ = meta[[batch_key]].copy()
    if meta_[batch_key].dtype == "category":
        meta_[batch_key] = meta_[batch_key].cat.remove_unused_categories()
    harmony_out = run_harmony(
        feats,
        meta,
        batch_key,
        max_iter_harmony=20,
        nclust=300,  # Number of compounds
    )

    feats = harmony_out.Z_corr.T
    features = [f"harmony_{i}" for i in range(feats.shape[1])]
    io.merge_parquet(meta, feats, features, output_path)

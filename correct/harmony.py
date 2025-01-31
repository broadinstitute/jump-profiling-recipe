"""
Functions for batch correction
"""

import logging
from harmonypy import run_harmony
from preprocessing import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_harmony_correction(dframe_path, batch_key, output_path):
    """Perform Harmony batch correction on feature data.

    Parameters
    ----------
    dframe_path : str
        Path to the input parquet file containing metadata and features.
    batch_key : str
        Column name in metadata that identifies the batch information.
    output_path : str
        Path where the corrected data will be saved as a parquet file.

    Returns
    -------
    None
        The corrected data is saved to the specified output path.

    Notes
    -----
    The function performs the following steps:
    1. Splits input parquet into metadata and features
    2. Removes unused categories from batch information
    3. Applies Harmony correction with 300 clusters and 20 iterations
    4. Saves the corrected features with original metadata to output path
    """
    meta, feats, features = io.split_parquet(dframe_path)

    # Remove unused categories to avoid matmul dimension mismatch
    meta_ = meta[[batch_key]].copy()
    if meta_[batch_key].dtype == "category":
        meta_[batch_key] = meta_[batch_key].cat.remove_unused_categories()
    harmony_out = run_harmony(
        feats,
        meta_,
        batch_key,
        max_iter_harmony=20,
        nclust=300,  # Number of compounds
    )

    feats = harmony_out.Z_corr.T
    features = [f"harmony_{i}" for i in range(feats.shape[1])]
    io.merge_parquet(meta, feats, features, output_path)

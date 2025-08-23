"""
Functions for batch correction
"""

import logging
from harmonypy import run_harmony
from ..preprocessing import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_harmony_correction(dframe_path, batch_key, use_gpu, output_path):
    """Perform Harmony batch correction on feature data.

    Parameters
    ----------
    dframe_path : str
        Path to the input parquet file containing metadata and features.
    batch_key : str
        Column name in metadata that identifies the batch information.
    use_gpu : bool
        Whether to use GPU acceleration if available.
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

    When use_gpu=True:
    - All matrix operations are performed on GPU using CuPy
    - Significant speedup for large datasets (>10,000 cells)
    - Falls back to CPU if GPU is unavailable
    """

    meta, feats, features = io.split_parquet(dframe_path)

    # Remove unused categories to avoid matmul dimension mismatch
    meta_ = meta[[batch_key]].copy()
    if meta_[batch_key].dtype == "category":
        meta_[batch_key] = meta_[batch_key].cat.remove_unused_categories()

    # Check GPU availability if requested
    if use_gpu:
        try:
            import cupy as cp

            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus > 0:
                logger.info(
                    f"Running Harmony with GPU acceleration ({n_gpus} device(s) available)"
                )
            else:
                logger.warning(
                    "GPU requested but no CUDA devices found. Falling back to CPU."
                )
                use_gpu = False
        except ImportError:
            logger.warning("GPU requested but CuPy not installed. Falling back to CPU.")
            use_gpu = False
        except Exception as e:
            logger.warning(
                f"GPU requested but CUDA error occurred: {e}. Falling back to CPU."
            )
            use_gpu = False

    # Run Harmony with appropriate backend
    logger.info(f"Running Harmony batch correction using {'GPU' if use_gpu else 'CPU'}")

    harmony_out = run_harmony(
        feats,
        meta_,
        batch_key,
        max_iter_harmony=20,
        nclust=300,  # Number of compounds
        use_cupy=use_gpu,
        dtype="float32" if use_gpu else None,  # Use float32 for GPU efficiency
        verbose=True,
    )

    feats = harmony_out.Z_corr.T
    features = [f"harmony_{i}" for i in range(feats.shape[1])]
    io.merge_parquet(meta, feats, features, output_path)

    logger.info(f"Harmony correction complete. Output saved to {output_path}")

"""
Functions for transforming feature data
"""

import numpy as np
import scipy.stats as ss
from tqdm.contrib.concurrent import thread_map

from .io import merge_parquet, split_parquet


def rank_int_array(
    array: np.ndarray, c: float = 3.0 / 8, stochastic: bool = True, seed: int = 0
) -> np.ndarray:
    """
    Perform rank-based inverse normal transformation on a 1D numpy array.

    Parameters
    ----------
    array : np.ndarray
        1-dimensional input array to be transformed.
    c : float, optional
        Blom's constant for quantile estimation, by default 3/8.
    stochastic : bool, optional
        If True, assigns random ranks to ties. If False, ties share the same value.
        Default is True.
    seed : int, optional
        Random seed when stochastic=True, by default 0.

    Returns
    -------
    np.ndarray
        Transformed array following standard normal distribution (mean=0, std=1).

    Notes
    -----
    Maps empirical distribution to standard normal while preserving rank order.
    Adapted from: https://github.com/edm1/rank-based-INT/blob/85cb37bb8e0d9e71bb9/rank_based_inverse_normal_transformation.py
    """
    rng = np.random.default_rng(seed=seed)

    if stochastic:
        # Shuffle
        ix = rng.permutation(len(array))
        rev_ix = np.argsort(ix)
        array = array[ix]
        # Get rank, ties are determined by their position(hence why we shuffle)
        rank = ss.rankdata(array, method="ordinal")
        rank = rank[rev_ix]
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(array, method="average")

    x = (rank - c) / (len(rank) - 2 * c + 1)
    return ss.norm.ppf(x)


def rank_int(normalized_path: str, rank_int_path: str) -> None:
    """
    Apply rank-based inverse normal transformation to data from a parquet file.

    Parameters
    ----------
    normalized_path : str
        Input parquet file path containing feature data.
    rank_int_path : str
        Output path for transformed data (overwrites existing file).

    Notes
    -----
    Processes feature columns in parallel using threads.
    """
    meta, vals, features = split_parquet(normalized_path)

    def to_normal(i):
        vals[:, i] = rank_int_array(vals[:, i]).astype(np.float32)

    thread_map(to_normal, range(len(features)), leave=False)
    merge_parquet(meta, vals, features, rank_int_path)

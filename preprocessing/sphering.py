"""
Functions for sphering

The sphering transformation is trained only on negative control samples (specified by
NEGCON_CODES) to ensure the transformation is learned from baseline cellular states.
The transformed data and sphering model are saved to disk for later use.
"""

import shutil

import numpy as np
import pandas as pd

from preprocessing.io import merge_parquet, split_parquet
from preprocessing.metadata import NEGCON_CODES
from pycytominer.operations import Spherize


def generate_log_uniform_samples(
    min_: float = -5.0,
    max_: float = 3.0,
    size: int = 25,
    seed: list[int] | int = (6, 12, 2022),
) -> np.ndarray:
    """
    Generate samples from a uniform distribution in log-space.

    Parameters
    ----------
    min_ : float, default=-5.0
        Lower bound exponent for the uniform distribution.
    max_ : float, default=3.0
        Upper bound exponent for the uniform distribution.
    size : int, default=25
        Number of samples to generate.
    seed : list of int or int, default=(6, 12, 2022)
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of generated samples in log-space.
    """
    rng = np.random.default_rng(seed)
    # Uniformly sample exponents in [min_, max_] and then take 10^(these exponents)
    return 10.0 ** rng.uniform(min_, max_, size=size)


def sphering(
    dframe_path: str,
    method: str,
    epsilon: float,
    sphered_path: str,
    spherer_path: str,
) -> None:
    """
    Apply sphering transformation to a parquet file.

    Parameters
    ----------
    dframe_path : str
        Path to the parquet file containing raw data.
    method : str
        Method to be used by the Spherize operation.
    epsilon : float
        Epsilon value for the Spherize operation.
    sphered_path : str
        Destination path for the transformed (sphered) parquet file.
    spherer_path : str
        Destination path for the compressed and saved spherer model.

    Returns
    -------
    None
    """
    # Instantiate the Spherize object with the provided method and epsilon
    spherer = Spherize(epsilon=epsilon, method=method)
    # Split the parquet data into metadata, values, and feature names
    meta, vals, features = split_parquet(dframe_path)
    # Identify which rows (samples) are negative controls
    train_ix = meta["Metadata_JCP2022"].isin(NEGCON_CODES).values
    # Fit the spherer on only the negative controls
    spherer.fit(vals[train_ix])
    # Transform the entire dataset using the fitted spherer
    vals = spherer.transform(vals).astype(np.float32)
    # Merge metadata, transformed values, and features back into a parquet file
    merge_parquet(meta, vals, features, sphered_path)
    # Save the fitted spherer object for later use
    np.savez_compressed(spherer_path, spherer=spherer)


def select_best(
    parquet_files: list[str],
    map_negcon_files: list[str],
    map_nonrep_files: list[str],
    ap_negcon_path: str,
    ap_nonrep_path: str,
    map_negcon_path: str,
    map_nonrep_path: str,
    parquet_path: str,
) -> None:
    """
    Among multiple processed files, select the best according to average precision
    scores and copy them to the given destinations.

    Parameters
    ----------
    parquet_files : list of str
        List of paths to parquet files.
    map_negcon_files : list of str
        List of paths to negative control MAP parquet files.
    map_nonrep_files : list of str
        List of paths to non-replicate MAP parquet files.
    ap_negcon_path : str
        Path to copy the chosen negative control AP parquet file.
    ap_nonrep_path : str
        Path to copy the chosen non-replicate AP parquet file.
    map_negcon_path : str
        Path to copy the chosen negative control MAP parquet file.
    map_nonrep_path : str
        Path to copy the chosen non-replicate MAP parquet file.
    parquet_path : str
        Path to copy the chosen parquet file.

    Returns
    -------
    None
    """
    scores = []
    # Iterate over all the candidate files to calculate and store their average
    # precision (AP) scores for both negative controls and non-replicate samples
    for negcon_file, nonrep_file, parquet_file in zip(
        map_negcon_files, map_nonrep_files, parquet_files
    ):
        # Read the negative control MAP parquet and compute mean AP
        negcon_score = (
            pd.read_parquet(negcon_file)["mean_average_precision"].dropna().mean()
        )
        # Read the non-replicate MAP parquet and compute mean AP
        nonrep_score = (
            pd.read_parquet(nonrep_file)["mean_average_precision"].dropna().mean()
        )
        # Store the average of both AP scores
        scores.append(
            {
                "parquet_file": parquet_file,
                "negcon_file": negcon_file,
                "nonrep_file": nonrep_file,
                "score": (negcon_score + nonrep_score) / 2,
            }
        )

    # Convert to a DataFrame for easier sorting and analysis
    scores = pd.DataFrame(scores)
    # Sort by the combined score and pick the highest-scoring entry
    best = scores.sort_values(by="score").iloc[-1]

    # Copy the best configuration
    shutil.copy(best["parquet_file"], parquet_path)
    shutil.copy(best["negcon_file"], map_negcon_path)
    shutil.copy(best["nonrep_file"], map_nonrep_path)

    # Derive the corresponding AP parquet file paths (negcon and nonrep)
    ap_negcon = best["negcon_file"].replace("_map_negcon.parquet", "_ap_negcon.parquet")
    ap_nonrep = best["nonrep_file"].replace("_map_nonrep.parquet", "_ap_nonrep.parquet")

    # Copy those as well
    shutil.copy(ap_negcon, ap_negcon_path)
    shutil.copy(ap_nonrep, ap_nonrep_path)

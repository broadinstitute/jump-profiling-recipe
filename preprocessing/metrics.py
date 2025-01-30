"""
Functions for computing metrics
"""

import copairs.map as copairs
import pandas as pd
import numpy as np
from typing import List, Optional

from preprocessing.io import split_parquet
from preprocessing.metadata import NEGCON_CODES
from preprocessing.io import _validate_columns


def _index(
    meta: pd.DataFrame,
    plate_types: List[str],
    ignore_codes: Optional[List[str]] = None,
    include_codes: Optional[List[str]] = None,
) -> np.ndarray:
    """Select samples to be used in mAP computation based on filtering criteria.

    Creates a boolean mask for samples that meet the following criteria:
    1. Belong to specified plate types
    2. Are not positive controls
    3. Have between 2 and 1000 replicates
    4. Are not compound JCP2022_033954 (excluded due to excessive replicates)
    5. Are not in ignore_codes (if specified)
    6. Are in include_codes (if specified)

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame containing sample information.
    plate_types : List[str]
        List of plate types to include in the analysis.
    ignore_codes : Optional[List[str]], optional
        List of perturbation codes to exclude, by default None.
    include_codes : Optional[List[str]], optional
        List of perturbation codes to forcibly include (bypassing replicate count
        requirements), by default None.

    Returns
    -------
    np.ndarray
        Boolean array indicating which samples to include.
    """
    required_cols = ["Metadata_PlateType", "Metadata_pert_type", "Metadata_JCP2022"]
    _validate_columns(meta, required_cols)

    index = meta["Metadata_PlateType"].isin(plate_types)
    index &= meta["Metadata_pert_type"] != "poscon"
    valid_cmpd = meta.loc[index, "Metadata_JCP2022"].value_counts()
    valid_cmpd = valid_cmpd[valid_cmpd.between(2, 1000)].index
    if include_codes:
        valid_cmpd = valid_cmpd.union(include_codes)
    index &= meta["Metadata_JCP2022"].isin(valid_cmpd)
    # TODO: This compound has many more replicates than any other. ignoring it
    # for now. This filter should be done early on.
    index &= meta["Metadata_JCP2022"] != "JCP2022_033954"
    if ignore_codes:
        index &= ~meta["Metadata_JCP2022"].isin(ignore_codes)
    return index.values


def _group_negcons(meta: pd.DataFrame) -> None:
    """Assign unique IDs to negative controls to prevent pair matching.

    This is a workaround to avoid mAP computation for negative controls by
    assigning a unique ID to each negative control sample, ensuring no pairs
    are found for such samples.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame containing sample information. Must have 'Metadata_JCP2022'
        column. Modified in-place to update negative control identifiers.
    """
    required_cols = ["Metadata_JCP2022"]
    _validate_columns(meta, required_cols)

    negcon_ix = meta["Metadata_JCP2022"].isin(NEGCON_CODES)
    n_negcon = negcon_ix.sum()
    negcon_ids = [f"negcon_{i}" for i in range(n_negcon)]
    pert_id = meta["Metadata_JCP2022"].astype("category").cat.add_categories(negcon_ids)
    pert_id[negcon_ix] = negcon_ids
    meta["Metadata_JCP2022"] = pert_id


def average_precision_negcon(
    parquet_path: str, ap_path: str, plate_types: List[str]
) -> None:
    """Calculate average precision with respect to negative controls.

    Parameters
    ----------
    parquet_path : str
        Path to input parquet file containing metadata and feature values.
        Must include columns: Metadata_JCP2022, Metadata_Plate, Metadata_pert_type
    ap_path : str
        Path where the average precision results will be saved.
    plate_types : List[str]
        List of plate types to include in the analysis.
    """
    meta, vals, _ = split_parquet(parquet_path)
    required_cols = [
        "Metadata_JCP2022",
        "Metadata_Plate",
        "Metadata_pert_type",
    ]
    _validate_columns(meta, required_cols)

    ix = _index(meta, plate_types, include_codes=NEGCON_CODES)
    meta = meta[ix].copy()
    vals = vals[ix]
    _group_negcons(meta)
    result = copairs.average_precision(
        meta,
        vals,
        pos_sameby=["Metadata_JCP2022"],
        # pos_diffby=['Metadata_Well'],
        pos_diffby=[],
        neg_sameby=["Metadata_Plate"],
        neg_diffby=["Metadata_pert_type", "Metadata_JCP2022"],
        batch_size=20000,
    )
    result = result.query('Metadata_pert_type!="negcon"')
    result.reset_index(drop=True).to_parquet(ap_path)


def average_precision_nonrep(
    parquet_path: str, ap_path: str, plate_types: List[str]
) -> None:
    """Calculate average precision with respect to non-replicate perturbations.

    Parameters
    ----------
    parquet_path : str
        Path to input parquet file containing metadata and feature values.
        Must include columns: Metadata_JCP2022, Metadata_Plate, Metadata_pert_type
    ap_path : str
        Path where the average precision results will be saved.
    plate_types : List[str]
        List of plate types to include in the analysis.
    """
    meta, vals, _ = split_parquet(parquet_path)
    required_cols = [
        "Metadata_JCP2022",
        "Metadata_Plate",
        "Metadata_pert_type",
    ]
    _validate_columns(meta, required_cols)

    ix = _index(meta, plate_types, ignore_codes=NEGCON_CODES)
    meta = meta[ix].copy()
    vals = vals[ix]
    result = copairs.average_precision(
        meta,
        vals,
        pos_sameby=["Metadata_JCP2022"],
        pos_diffby=[],
        neg_sameby=["Metadata_Plate"],
        neg_diffby=["Metadata_JCP2022"],
        batch_size=20000,
    )
    result.reset_index(drop=True).to_parquet(ap_path)


def mean_average_precision(
    ap_path: str, map_path: str, threshold: float = 0.05
) -> None:
    """Calculate mean average precision from average precision scores.

    Parameters
    ----------
    ap_path : str
        Path to input file containing average precision scores.
        Must include column: Metadata_JCP2022
    map_path : str
        Path where the mean average precision results will be saved.
    threshold : float, optional
        Threshold for significance testing, by default 0.05.
    """
    ap_scores = pd.read_parquet(ap_path)
    required_cols = ["Metadata_JCP2022"]
    _validate_columns(ap_scores, required_cols)

    map_scores = copairs.mean_average_precision(
        ap_scores, "Metadata_JCP2022", threshold=threshold, null_size=10000, seed=0
    )
    map_scores.to_parquet(map_path)

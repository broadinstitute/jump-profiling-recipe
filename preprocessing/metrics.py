import copairs.map as copairs
import pandas as pd

from preprocessing.io import split_parquet
from preprocessing.metadata import NEGCON_CODES


def _index(meta, plate_types, ignore_codes=None, include_codes=None):
    """Select samples to be used in mAP computation"""
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


def _group_negcons(meta: pd.DataFrame):
    """
    Hack to avoid mAP computation for negcons. Assign a unique id for every
    negcon so that no pairs are found for such samples.
    """
    negcon_ix = meta["Metadata_JCP2022"].isin(NEGCON_CODES)
    n_negcon = negcon_ix.sum()
    negcon_ids = [f"negcon_{i}" for i in range(n_negcon)]
    pert_id = meta["Metadata_JCP2022"].astype("category").cat.add_categories(negcon_ids)
    pert_id[negcon_ix] = negcon_ids
    meta["Metadata_JCP2022"] = pert_id


def average_precision_negcon(parquet_path, ap_path, plate_types):
    meta, vals, _ = split_parquet(parquet_path)
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


def average_precision_nonrep(parquet_path, ap_path, plate_types):
    meta, vals, _ = split_parquet(parquet_path)
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


def mean_average_precision(ap_path, map_path, threshold=0.05):
    ap_scores = pd.read_parquet(ap_path)

    map_scores = copairs.mean_average_precision(
        ap_scores, "Metadata_JCP2022", threshold=threshold, null_size=10000, seed=0
    )
    map_scores.to_parquet(map_path)

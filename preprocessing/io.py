import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
import logging

from .metadata import (
    build_path,
    load_metadata,
    MICRO_CONFIG,
    find_feat_cols,
    find_meta_cols,
    DMSO,
)

# logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_parquet(
    dframe_path, features=None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    dframe = pd.read_parquet(dframe_path)
    if features is None:
        features = find_feat_cols(dframe)
    vals = np.empty((len(dframe), len(features)), dtype=np.float32)
    for i, c in enumerate(features):
        vals[:, i] = dframe[c]
    meta = dframe[find_meta_cols(dframe)].copy()
    return meta, vals, features


def merge_parquet(meta, vals, features, output_path) -> None:
    """Save the data in a parquet file resetting the index"""
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    logger.info(f"Saving file {output_path.split('/')[-1]}")
    report_nan_infs_columns(dframe)
    dframe.to_parquet(output_path)


def get_num_rows(path) -> int:
    """Count the number of rows in a parquet file"""
    with pq.ParquetFile(path) as file:
        return file.metadata.num_rows


def prealloc_params(sources, plate_types):
    """
    Get a list of paths to the parquet files and the corresponding slices
    for further concatenation
    """
    meta = load_metadata(sources, plate_types)
    paths = (
        meta[["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]]
        .drop_duplicates()
        .apply(build_path, axis=1)
    ).values

    counts = thread_map(get_num_rows, paths, leave=False, desc="counts")
    slices = np.zeros((len(paths), 2), dtype=int)
    slices[:, 1] = np.cumsum(counts)
    slices[1:, 0] = slices[:-1, 1]
    return paths, slices


def load_data(sources, plate_types):
    """Load all plates given the params"""
    paths, slices = prealloc_params(sources, plate_types)
    total = slices[-1, 1]

    with pq.ParquetFile(paths[0]) as f:
        meta_cols = find_meta_cols(f.schema.names)
        feat_cols = find_feat_cols(f.schema.names)
    meta = np.empty([total, len(meta_cols)], dtype="|S128")
    feats = np.empty([total, len(feat_cols)], dtype=np.float32)

    def read_parquet(params):
        path, start, end = params
        df = pd.read_parquet(path)
        meta[start:end] = df[meta_cols].values
        feats[start:end] = df[feat_cols].values

    params = np.concatenate([paths[:, None], slices], axis=1)
    thread_map(read_parquet, params)

    meta = pd.DataFrame(data=meta.astype(str), columns=meta_cols, dtype="category")
    dframe = pd.DataFrame(columns=feat_cols, data=feats)
    for col in meta_cols:
        dframe[col] = meta[col]
    return dframe


def add_pert_type(
    meta: pd.DataFrame, col: str = "Metadata_pert_type", negcon_list: list = [DMSO]
):
    if not col in meta.columns:
        meta[col] = "trt"
        meta.loc[~meta["Metadata_JCP2022"].str.startswith("JCP"), col] = "poscon"
        meta.loc[meta["Metadata_JCP2022"].isin(negcon_list), col] = "negcon"
    meta[col] = meta[col].astype("category")


def add_row_col(meta: pd.DataFrame):
    """Add Metadata_Row and Metadata_Column to the DataFrame"""
    well_regex = r"^(?P<row>[a-zA-Z]{1,2})(?P<column>[0-9]{1,2})$"
    position = meta["Metadata_Well"].str.extract(well_regex)
    meta["Metadata_Row"] = position["row"].astype("category")
    meta["Metadata_Column"] = position["column"].astype("category")


def add_microscopy_info(meta: pd.DataFrame):
    configs = meta["Metadata_Source"].map(MICRO_CONFIG).astype("category")
    meta["Metadata_Microscope"] = configs


def write_parquet(sources, plate_types, output_file, negcon_list=[DMSO]):
    """Write the parquet dataset given the params"""
    dframe = load_data(sources, plate_types)
    # Efficient merge
    meta = load_metadata(sources, plate_types)
    add_pert_type(meta, negcon_list=negcon_list)
    add_row_col(meta)
    add_microscopy_info(meta)
    foreign_key = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    meta = dframe[foreign_key].merge(meta, on=foreign_key, how="left")
    assert dframe.shape[0] == meta.shape[0]
    for c in meta:
        dframe[c] = meta[c].astype("category")
    # Drop Image features
    image_col = [col for col in dframe.columns if "Image_" in col]
    dframe = dframe.drop(image_col, axis=1)
    # Dropping samples with no metadata
    dframe.dropna(subset=["Metadata_JCP2022"], inplace=True)

    # Dropping positive controls
    # dframe = dframe[dframe["Metadata_pert_type"] != "poscon"]

    dframe.reset_index(drop=True, inplace=True)
    dframe.to_parquet(output_file)


def report_nan_infs_columns(dframe: pd.DataFrame):
    logger.info("Checking for NaN and INF")
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    if withnan.shape[0] > 0:
        logger.info(f"Columns with NaN: {withnan}")
    if withinf.shape[0] > 0:
        logger.info(f"Columns with INF: {withinf}")
    if withninf.shape[0] > 0:
        logger.info(f"Columns with NINF: {withninf}")

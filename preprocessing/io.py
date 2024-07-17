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
    meta: pd.DataFrame, col: str = "Metadata_pert_type",poscon_list:list =["JCP2022_012818","JCP2022_050797","JCP2022_064022","JCP2022_035095" , "JCP2022_046054", "JCP2022_025848", "JCP2022_037716", "JCP2022_085227", "JCP2022_805264", "JCP2022_915132"], negcon_list: list = ["JCP2022_800001", "JCP2022_800002", "JCP2022_033924", "JCP2022_915131", "JCP2022_915130", "JCP2022_915129", "JCP2022_915128"]
):
    meta[col] = "trt"
    meta.loc[meta["Metadata_JCP2022"].isin(poscon_list), col] = "poscon"
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
    # Drop Image features
    image_col = [col for col in dframe.columns if "Image_" in col]
    dframe.drop(image_col, axis=1, inplace=True)

    # Get metadata
    meta = load_metadata(sources, plate_types)
    add_pert_type(meta, negcon_list=negcon_list)
    add_row_col(meta)
    add_microscopy_info(meta)
    foreign_key = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    meta = dframe[foreign_key].merge(meta, on=foreign_key, how="left")

    # Dropping samples with no metadata
    jcp_col = meta.pop("Metadata_JCP2022").astype("category")
    dframe["Metadata_JCP2022"] = jcp_col
    dframe.dropna(subset=["Metadata_JCP2022"], inplace=True)
    meta = meta[~jcp_col.isna()].copy()
    assert (meta.index == dframe.index).all()

    for c in meta:
        dframe[c] = meta[c].astype("category")

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

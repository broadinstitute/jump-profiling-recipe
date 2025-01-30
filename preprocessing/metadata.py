"""
Functions to load metadata information

File Structure:
- Constants & Configuration: Global constants and configuration variables
- Column Operations: Functions for identifying metadata and feature columns
- Metadata Loading & Filtering: Functions for loading and filtering metadata
- Plate & Well Operations: Functions for handling plate and well-specific data
"""

import logging
from collections.abc import Iterable
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------
# Constants & Configuration
# ------------------------------

UNTREATED = "JCP2022_999999"
UNKNOWN = "JCP2022_UNKNOWN"
BAD_CONSTRUCT = "JCP2022_900001"

MICRO_CONFIG = pd.read_csv(
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/181fa0dc96b0d68511b437cf75a712ec782576aa/metadata/microscope_config.csv"
)
MICRO_CONFIG["Metadata_Source"] = "source_" + MICRO_CONFIG["Metadata_Source"].astype(
    str
)
MICRO_CONFIG = MICRO_CONFIG.set_index("Metadata_Source")["Metadata_Microscope_Name"]

POSCON_CODES = [
    "JCP2022_012818",
    "JCP2022_050797",
    "JCP2022_064022",
    "JCP2022_035095",
    "JCP2022_046054",
    "JCP2022_025848",
    "JCP2022_037716",
    "JCP2022_085227",
    "JCP2022_805264",
    "JCP2022_915132",
]

NEGCON_CODES = [
    "JCP2022_800001",
    "JCP2022_800002",
    "JCP2022_033924",
    "JCP2022_915131",
    "JCP2022_915130",
    "JCP2022_915129",
    "JCP2022_915128",
]

SOURCE3_BATCH_REDLIST = {
    "CP_32_all_Phenix1",
    "CP_33_all_Phenix1",
    "CP_34_mix_Phenix1",
    "CP_35_all_Phenix1",
    "CP_36_all_Phenix1",
    "CP59",
    "CP60",
}

# ------------------------------
# Column Operations
# ------------------------------


def find_feat_cols(cols: Iterable[str]) -> list[str]:
    """Find column names for features.

    Parameters
    ----------
    cols : Iterable[str]
        Collection of column names to search through.

    Returns
    -------
    list[str]
        List of feature column names (those not starting with 'Meta').
    """
    feat_cols = [c for c in cols if not c.startswith("Meta")]
    return feat_cols


def find_meta_cols(cols: Iterable[str]) -> list[str]:
    """Find column names for metadata.

    Parameters
    ----------
    cols : Iterable[str]
        Collection of column names to search through.

    Returns
    -------
    list[str]
        List of metadata column names (those starting with 'Meta').
    """
    meta_cols = [c for c in cols if c.startswith("Meta")]
    return meta_cols


# ------------------------------
# Metadata Loading & Filtering
# ------------------------------


def load_metadata(sources: list[str], plate_types: list[str]) -> pd.DataFrame:
    """Load and merge plate and well metadata.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to include.
    plate_types : list[str]
        List of plate types to include.

    Returns
    -------
    pd.DataFrame
        Merged metadata DataFrame containing both plate and well information,
        filtered according to the specified sources and plate types.
    """
    plate = get_plate_metadata(sources, plate_types)
    well = get_well_metadata(plate_types)
    meta = well.merge(plate, on=["Metadata_Source", "Metadata_Plate"])
    return meta


def get_orf_plate_redlist(plate_types: list[str]) -> set[str]:
    """Get set of plate_id's that should not be considered in the analysis.

    Parameters
    ----------
    plate_types : list[str]
        List of plate types to consider.

    Returns
    -------
    set[str]
        Set of plate IDs to exclude from analysis.

    Notes
    -----
    Excludes:
    - Low concentration plates
    - Plates from specific batches (e.g., Batch12)
    """
    # https://github.com/jump-cellpainting/jump-orf-analysis/issues/1#issuecomment-921888625
    # Low concentration plates
    redlist = set(["BR00127147", "BR00127148", "BR00127145", "BR00127146"])
    # https://github.com/jump-cellpainting/aws/issues/70#issuecomment-1182444836
    redlist.add("BR00123528A")

    # filter ORF plates.
    metadata = pd.read_csv("inputs/experiment-metadata.tsv", sep="\t")
    query = 'Batch=="Batch12"'
    bad_plates = set(metadata.query(query).Assay_Plate_Barcode)
    redlist |= bad_plates
    return redlist


# ------------------------------
# Plate & Well Operations
# ------------------------------


def build_path(row: pd.Series) -> str:
    """Create the path to the parquet file.

    Parameters
    ----------
    row : pd.Series
        Row containing metadata information with required fields:
        Metadata_Source, Metadata_Batch, Metadata_Plate.

    Returns
    -------
    str
        Formatted path string to the parquet file.
    """
    required_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
    missing_cols = [col for col in required_cols if col not in row.index]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    template = (
        "./inputs/{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )
    return template.format(**row.to_dict())


def get_plate_metadata(sources: list[str], plate_types: list[str]) -> pd.DataFrame:
    """Create filtered metadata DataFrame.

    Parameters
    ----------
    sources : list[str]
        List of source identifiers to include.
    plate_types : list[str]
        List of plate types to include.

    Returns
    -------
    pd.DataFrame
        Filtered plate metadata containing only specified sources and plate types,
        with certain plates excluded based on redlist criteria.
    """
    plate_metadata = pd.read_csv("./inputs/metadata/plate.csv.gz")

    required_cols = [
        "Metadata_Source",
        "Metadata_Plate",
        "Metadata_PlateType",
        "Metadata_Batch",
    ]
    missing_cols = [col for col in required_cols if col not in plate_metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in plate metadata: {missing_cols}")

    # Filter plates from source_4
    if "ORF" in plate_types:
        redlist = get_orf_plate_redlist(plate_types)
        plate_metadata = plate_metadata[~plate_metadata["Metadata_Plate"].isin(redlist)]

    # Filter plates from source_3 batches without DMSO
    plate_metadata = plate_metadata[
        (~plate_metadata["Metadata_Batch"].isin(SOURCE3_BATCH_REDLIST))
        | (plate_metadata["Metadata_PlateType"] == "TARGET2")
    ]

    plate_metadata = plate_metadata[plate_metadata["Metadata_Source"].isin(sources)]
    plate_metadata = plate_metadata[
        plate_metadata["Metadata_PlateType"].isin(plate_types)
    ]
    return plate_metadata


def get_well_metadata(plate_types: list[str]) -> pd.DataFrame:
    """Load well metadata.

    Parameters
    ----------
    plate_types : list[str]
        List of plate types to process. Special handling for 'ORF' and 'CRISPR' types.

    Returns
    -------
    pd.DataFrame
        Well metadata DataFrame, optionally merged with ORF or CRISPR metadata
        depending on plate_types. Excludes certain predefined JCP2022 codes.
    """
    well_metadata = pd.read_csv("./inputs/metadata/well.csv.gz")

    required_cols = ["Metadata_JCP2022"]
    missing_cols = [col for col in required_cols if col not in well_metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in well metadata: {missing_cols}")

    if "ORF" in plate_types:
        orf_metadata = pd.read_csv("./inputs/metadata/orf.csv.gz")
        well_metadata = well_metadata.merge(
            orf_metadata, how="left", on="Metadata_JCP2022"
        )
        # well_metadata = well_metadata[well_metadata['Metadata_pert_type']!='poscon']
    if "CRISPR" in plate_types:
        crispr_metadata = pd.read_csv("./inputs/metadata/crispr.csv.gz")
        well_metadata = well_metadata.merge(
            crispr_metadata, how="left", on="Metadata_JCP2022"
        )
    # Filter out wells

    well_metadata = well_metadata[
        ~well_metadata["Metadata_JCP2022"].isin([UNTREATED, UNKNOWN, BAD_CONSTRUCT])
    ]

    return well_metadata

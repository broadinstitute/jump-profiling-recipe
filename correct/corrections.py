"""Functions for data preprocessing, annotation, and correction.

File Structure:
- DataFrame Column Operations: Utilities for handling metadata and feature columns
- NA Value Handling: Functions for cleaning and handling missing values
- Data Annotation: Gene and chromosome annotation functions
- Data Transformation & Correction: Well position, PCA, and arm correction methods
- Cell Count Operations: Cell count regression and related utilities
"""

import sys

sys.path.append("..")
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)


# ------------------------------
# DataFrame Column Operations
# ------------------------------


def get_meta_cols(df):
    """Get metadata columns from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.Index
        List of metadata column names
    """
    return df.filter(regex="^(Metadata_)").columns


def get_feature_cols(df):
    """Get feature columns from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.Index
        List of feature column names
    """
    return df.filter(regex="^(?!Metadata_)").columns


def get_metadata(df):
    """Get metadata columns subset from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.DataFrame
        Dataframe containing only metadata columns
    """
    return df[get_meta_cols(df)]


def get_featuredata(df):
    """Get feature columns subset from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.DataFrame
        Dataframe containing only feature columns
    """
    return df[get_feature_cols(df)]


# ------------------------------
# NA Value Handling
# ------------------------------


def drop_rows_with_na_features(ann_dframe: pd.DataFrame) -> pd.DataFrame:
    """Drop rows containing NA values in feature columns.

    Parameters
    ----------
    ann_dframe : pandas.DataFrame
        Input dataframe to process

    Returns
    -------
    pandas.DataFrame
        If fewer than 100 rows would be dropped:
            Returns cleaned dataframe with rows containing NA values in feature columns removed
        Otherwise:
            Returns original dataframe unchanged
    """
    org_shape = ann_dframe.shape[0]
    ann_dframe_clean = ann_dframe[
        ~ann_dframe[get_feature_cols(ann_dframe)].isnull().T.any()
    ]
    ann_dframe_clean.reset_index(drop=True, inplace=True)
    if org_shape - ann_dframe_clean.shape[0] < 100:
        return ann_dframe_clean
    return ann_dframe


def drop_features_with_na(df):
    """Remove features containing NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.DataFrame
        Dataframe with NaN-containing features removed
    """
    _, c = np.where(df.isna())
    features_to_remove = [
        _ for _ in list(df.columns[list(set(c))]) if not _.startswith("Metadata_")
    ]
    logger.info(f"Removing {len(features_to_remove)} features containing NaN values")
    return df.drop(features_to_remove, axis=1)


# ------------------------------
# Data Annotation
# ------------------------------


def annotate_gene(df, df_meta):
    """Annotate dataframe with gene symbols.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    df_meta : pandas.DataFrame
        Metadata dataframe containing gene symbol information

    Returns
    -------
    pandas.DataFrame
        Annotated dataframe with gene symbols

    Raises
    ------
    ValueError
        If required columns are missing in df_meta
        If merge results in data loss
    """
    # Check required columns exist in metadata
    required_cols = ["Metadata_JCP2022", "Metadata_Symbol"]
    if not all(col in df_meta.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_meta.columns]
        raise ValueError(f"Missing required columns in df_meta: {missing}")

    if "Metadata_Symbol" not in df.columns:
        # Store original row count
        original_rows = len(df)

        # Perform merge
        df = df.merge(
            df_meta[["Metadata_JCP2022", "Metadata_Symbol"]],
            on="Metadata_JCP2022",
            how="left",
        )

        # Check for data loss
        if len(df) != original_rows:
            raise ValueError(
                f"Merge resulted in row count change: {original_rows} -> {len(df)}"
            )

        # Check for null values after merge
        null_count = df["Metadata_Symbol"].isnull().sum()
        if null_count > 0:
            logger.warning(
                f"Merge resulted in {null_count} null values in Metadata_Symbol"
            )

    return df


def annotate_chromosome(df, df_meta):
    """Annotate dataframe with chromosome information.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    df_meta : pandas.DataFrame
        Metadata dataframe containing chromosome information

    Returns
    -------
    pandas.DataFrame
        Annotated dataframe with chromosome information
    """

    def split_arm(locus):
        return (
            f"{locus.split('p')[0]}p"
            if "p" in locus
            else f"{locus.split('q')[0]}q"
            if "q" in locus
            else np.nan
        )

    if "Metadata_Locus" not in df.columns:
        df_meta_copy = df_meta.drop_duplicates(subset=["Approved_symbol"]).reset_index(
            drop=True
        )
        df_meta_copy.columns = [
            "Metadata_" + col if not col.startswith("Metadata_") else col
            for col in df_meta_copy.columns
        ]

        df = df.merge(
            df_meta_copy,
            how="left",
            left_on="Metadata_Symbol",
            right_on="Metadata_Approved_symbol",
        ).reset_index(drop=True)

    if "Metadata_arm" not in df.columns:
        df["Metadata_arm"] = df["Metadata_Locus"].apply(lambda x: split_arm(str(x)))

    if "Metadata_Chromosome" not in df.columns:
        df["Metadata_Chromosome"] = df["Metadata_Chromosome"].apply(
            lambda x: "12" if x == "12 alternate reference locus" else x
        )

    return df


def annotate_dataframe(
    df_path: str,
    output_path: str,
    df_gene_path: str,
    df_chrom_path: str,
):
    """Annotate dataframe with gene and chromosome information.

    Parameters
    ----------
    df_path : str
        Path to input parquet file containing profiles
    output_path : str
        Path where annotated dataframe will be saved
    df_gene_path : str
        Path to CSV file containing gene annotation data
    df_chrom_path : str
        Path to TSV file containing chromosome annotation data

    Returns
    -------
    None
        Saves annotated dataframe to output_path with added gene symbols
        and chromosome location information
    """
    df = pd.read_parquet(df_path)
    df_gene = pd.read_csv(df_gene_path)
    df_chrom = pd.read_csv(df_chrom_path, sep="\t", dtype=str)

    df = drop_features_with_na(df)
    df = annotate_gene(df, df_gene)
    df = annotate_chromosome(df, df_chrom)
    df.to_parquet(output_path)


# ------------------------------
# Data Transformation & Correction
# ------------------------------


def subtract_well_mean(input_path: str, output_path: str):
    """Subtract the mean of each feature per each well.

    Parameters
    ----------
    input_path : str
        Path to input parquet file containing the dataframe
    output_path : str
        Path where the processed dataframe will be saved as parquet

    Returns
    -------
    None
        Saves processed dataframe to output_path
    """
    df = pd.read_parquet(input_path)
    df = drop_rows_with_na_features(df)

    feature_cols = get_feature_cols(df)
    mean_ = df.groupby("Metadata_Well")[feature_cols].transform("mean").values
    df[feature_cols] = df[feature_cols].values - mean_
    df.to_parquet(output_path, index=False)


def transform_data(input_path: str, output_path: str, variance=0.98):
    """Transform data by applying PCA.

    Parameters
    ----------
    input_path : str
        Path to input dataframe
    output_path : str
        Path to output dataframe
    variance : float, optional
        Variance to keep after PCA, by default 0.98

    Returns
    -------
    None
        Saves transformed dataframe to output_path with PCA-transformed features
        and original metadata columns
    """
    df = pd.read_parquet(input_path)

    metadata = df[get_meta_cols(df)]
    features = df[get_feature_cols(df)]

    features = pd.DataFrame(PCA(variance).fit_transform(features))

    df_new = pd.concat([metadata, features], axis=1)
    df_new.columns = df_new.columns.astype(str)
    df_new.to_parquet(output_path, index=False)


def arm_correction(
    crispr_profile_path: str, output_path: str, gene_expression_file: str
):
    """Perform chromosome arm correction on CRISPR profiles.

    This function corrects for chromosome arm effects by:
    1. Separating CRISPR and non-CRISPR profiles
    2. Identifying unexpressed genes (zfpkm < -3)
    3. Computing mean profiles for unexpressed genes by chromosome arm
    4. Subtracting arm-specific background from CRISPR profiles
    5. Recombining corrected profiles

    Parameters
    ----------
    crispr_profile_path : str
        Path to parquet file containing CRISPR profiles
    output_path : str
        Path where corrected profiles will be saved
    gene_expression_file : str
        Path to CSV file containing gene expression data (with zfpkm values)

    Returns
    -------
    None
        Saves chromosome arm-corrected profiles to output_path
    """
    df_exp = pd.read_csv(gene_expression_file)
    df = pd.read_parquet(crispr_profile_path)
    crispr_ix = df["Metadata_PlateType"] == "CRISPR"
    df_crispr = df[crispr_ix].copy()
    df = df[~crispr_ix].copy()

    df_no_arm = df_crispr[df_crispr["Metadata_Chromosome"].isna()].reset_index(
        drop=True
    )
    df_crispr = df_crispr[~df_crispr["Metadata_Chromosome"].isna()].reset_index(
        drop=True
    )

    df_exp = df_exp.assign(
        Metadata_arm=df_exp.gene.map(
            dict(zip(df_crispr.Metadata_Symbol, df_crispr.Metadata_arm))
        )
    )
    df_exp = df_exp.dropna(subset="Metadata_arm")

    unexp_genes = df_exp[df_exp["zfpkm"] < -3]["gene"].unique()
    arm_include = (
        df_exp[df_exp["zfpkm"] < -3].groupby("Metadata_arm")["gene"].nunique() > 20
    )

    feature_cols = get_feature_cols(df_crispr)
    feature_cols_unexpressed = [feat + "_unexpressed" for feat in feature_cols]

    df_unexp_mean = (
        df_crispr[df_crispr["Metadata_Symbol"].isin(unexp_genes)]
        .groupby("Metadata_arm")[feature_cols]
        .mean()[arm_include]  # filter for arms to include
    )

    df_unexp_mean = df_crispr.merge(
        df_unexp_mean, on="Metadata_arm", how="left", suffixes=("", "_unexpressed")
    )[feature_cols_unexpressed]

    df_unexp_mean = df_unexp_mean.fillna(0)
    df_unexp_mean.columns = feature_cols

    df_crispr[feature_cols] = df_crispr[feature_cols] - df_unexp_mean[feature_cols]

    col = df_no_arm.columns
    df_crispr = pd.concat([df_no_arm, df_crispr[col]], axis=0, ignore_index=True)

    df = pd.concat([df, df_crispr])
    df.to_parquet(output_path, index=False)


# ------------------------------
# Cell Count Operations
# ------------------------------


def merge_cell_counts(df, cc_path):
    """Merge cell count data with input dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    cc_path : str
        Path to cell count CSV file

    Returns
    -------
    pandas.DataFrame
        Merged dataframe with cell count information
    """
    df_cc = pd.read_csv(
        cc_path,
        low_memory=False,
        dtype={
            "Metadata_Plate": str,
            "Metadata_Well": str,
            "Metadata_Count_Cells": int,
        },
    )
    df_cc.rename(columns={"Metadata_Count_Cells": "Cells_Count_Count"}, inplace=True)
    df = df.merge(
        df_cc[["Metadata_Well", "Metadata_Plate", "Cells_Count_Count"]],
        on=["Metadata_Well", "Metadata_Plate"],
        how="left",
    ).reset_index(drop=True)
    return df


def regress_out_cell_counts_parallel(
    input_path: str,
    output_path: str,
    cc_path: str,
    cc_col: str = "Cells_Count_Count",
    min_unique: int = 100,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Regress out cell counts from all features in a dataframe in parallel.

    Parameters
    ----------
    ann_df : pandas.core.frame.DataFrame
        DataFrame of annotated profiles.
    cc_col : str
        Name of column containing cell counts.
    min_unique : int, optional
        Minimum number of unique feature values to perform regression.
    cc_rename : str, optional
        Name to rename cell count column to.
    inplace : bool, optional
        Whether to perform operation in place.

    Returns
    -------
    df : pandas.core.frame.DataFrame
    """
    ann_df = pd.read_parquet(input_path)
    df = ann_df if inplace else ann_df.copy()
    df = merge_cell_counts(df, cc_path)

    feature_cols = df.filter(regex="^(?!Metadata_)").columns.to_list()
    feature_cols.remove(cc_col)
    feature_cols = [
        feature for feature in feature_cols if df[feature].nunique() > min_unique
    ]

    logger.info(
        f"Performing cell count regression on {len(feature_cols)} features (requiring >{min_unique} unique values per feature)"
    )
    resid = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    for i, feature in tqdm(
        enumerate(feature_cols), leave=False, total=len(feature_cols)
    ):
        model = ols(f"{feature} ~ {cc_col}", data=df).fit()
        resid[model.resid.index, i] = model.resid.values
    logger.info("Replacing NaN regression residuals with original feature values")
    mask = np.isnan(resid)
    vals = df[feature_cols].values
    vals = mask * vals + (1 - mask) * resid
    logger.info("Converting regression residuals to dataframe format")
    df_res = pd.DataFrame(index=df.index, columns=feature_cols, data=vals)
    logger.info("Adding non-feature columns (metadata) to dataframe")
    for c in df:
        if c not in df_res:
            df_res[c] = df[c].values
    logger.info("Removing features that contain NaN values after regression")
    df_res = drop_features_with_na(df_res)
    logger.info(f"Writing cell count corrected profiles to {output_path}")
    df_res.to_parquet(output_path, index=False)

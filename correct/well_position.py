"""Functions to perform well position correction, chromosome arm correction, and PCA"""
from concurrent import futures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


def get_meta_cols(df):
    """return a list of metadata columns"""
    return df.filter(regex="^(Metadata_)").columns


def get_feature_cols(df):
    """returna  list of featuredata columns"""
    return df.filter(regex="^(?!Metadata_)").columns


def subtract_well_mean_parallel(input_path: str, output_path: str):
    """Subtract the mean of each feature per each well in parallel."""
    df = pd.read_parquet(input_path)
    feature_cols = get_feature_cols(df)

    # rewrite main loop to parallelize it
    def subtract_well_mean_parallel_helper(feature):
        return {feature: df[feature] - df.groupby("Metadata_Well")[feature].mean()}

    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(subtract_well_mean_parallel_helper, feature_cols)

    for res in results:
        df.update(pd.DataFrame(res))

    df.to_parquet(output_path, index=False)


def transform_data(input_path: str, output_path: str, variance=0.98):
    """Transform data by scaling and applying PCA. Data is scaled by plate
    before and after PCA is applied. The experimental replicates are averaged
    together by taking the mean.

    Parameters
    ----------
    input_path : str
        Path to input dataframe
    output_path : str
        Path to output dataframe
    variance : float, optional
        Variance to keep after PCA, by default 0.98
    """
    df = pd.read_parquet(input_path)

    metadata = df[get_meta_cols(df)]
    features = df[get_feature_cols(df)]

    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    features = pd.DataFrame(PCA(variance).fit_transform(features))

    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    df_new = pd.concat([metadata, features], axis=1)
    df_new.columns = df_new.columns.astype(str)
    df_new.to_parquet(output_path, index=False)


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_meta_cols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_feature_cols(df)]


def remove_nan_features(df):
    """remove nan features"""
    _, c = np.where(df.isna())
    features_to_remove = [
        _ for _ in list(df.columns[list(set(c))]) if not _.startswith("Metadata_")
    ]
    print(f"Removed nan features: {features_to_remove}")
    return df.drop(features_to_remove, axis=1)


def annotate_gene(df, df_meta):
    """annotate genes names"""
    if "Metadata_Symbol" not in df.columns:
        df = df.merge(
            df_meta[["Metadata_JCP2022", "Metadata_Symbol"]],
            on="Metadata_JCP2022",
            how="inner",
        )
    return df


def annotate_chromosome(df, df_meta):
    """annotate chromosome locus"""
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

    df["Metadata_Chromosome"] = df["Metadata_Chromosome"].apply(
        lambda x: "12" if x == "12 alternate reference locus" else x
    )

    return df


def split_arm(locus):
    """helper function to split p and q arms"""
    if "p" in locus:
        return locus.split("p")[0] + "p"
    if "q" in locus:
        return locus.split("q")[0] + "q"
    return np.nan


def annotate_dataframe(
    df_path: str,
    output_path: str,
    df_gene_path: str,
    df_chrom_path: str,
):
    """Annotate gene and chromosome name and sort genes based on
    chromosome location"""
    df = pd.read_parquet(df_path)
    df_gene = pd.read_csv(df_gene_path)
    df_chrom = pd.read_csv(df_chrom_path, sep="\t", dtype=str)

    df = remove_nan_features(df)
    df = annotate_gene(df, df_gene)
    df = annotate_chromosome(df, df_chrom)
    df.to_parquet(output_path)


def arm_correction(
    gene_expression_file: str, crispr_profile_path: str, output_path: str
):
    """Perform chromosome arm correction"""
    df_exp = pd.read_csv(gene_expression_file)
    df_crispr = pd.read_parquet(crispr_profile_path)

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

    df_crispr.to_parquet(output_path, index=False)

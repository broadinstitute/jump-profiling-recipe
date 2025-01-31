wildcard_constraints:
    pipeline=r"[_a-zA-Z.~0-9\-]*",
    scenario=r"[_a-zA-Z.~0-9\-]*",


import correct
import preprocessing as pp


include: "rules/sphering.smk"
include: "rules/map.smk"


rule all:
    input:
        f"outputs/{config['scenario']}/reformat.done",


rule reformat:
    input:
        f"outputs/{config['scenario']}/{config['pipeline']}.parquet",
    output:
        touch("outputs/{scenario}/reformat.done"),
    params:
        profile_dir=lambda w: f"outputs/{w.scenario}/",
    run:
        correct.format_check.run_format_check(params.profile_dir)


rule write_parquet:
    output:
        "outputs/{scenario}/profiles.parquet",
    run:
        pp.io.write_parquet(
            config["sources"],
            config["plate_types"],
            *output,
        )


rule compute_norm_stats:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/norm_stats/{pipeline}.parquet",
    params:
        use_negcon=config["use_mad_negcon"],
    run:
        pp.stats.compute_norm_stats(*input, *output, **params)


rule select_variant_feats:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
        "outputs/{scenario}/norm_stats/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_var.parquet",
    run:
        pp.stats.select_variant_features(*input, *output)


rule mad_normalize:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
        "outputs/{scenario}/norm_stats/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_mad.parquet",
    run:
        pp.normalize.mad(*input, *output)


rule INT:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_int.parquet",
    run:
        pp.transform.rank_int(*input, *output)


rule well_correct:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_wellpos.parquet",
    run:
        correct.corrections.subtract_well_mean(*input, *output)


rule cc_regress:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_cc.parquet",
    params:
        cc_path=config.get("cc_path"),
    run:
        correct.corrections.regress_out_cell_counts_parallel(
            *input, *output, params.cc_path
        )


rule remove_outliers:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_outlier.parquet",
    run:
        pp.clean.remove_outliers(*input, *output)


rule annotate_genes:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_annotated.parquet",
    params:
        df_gene_path="inputs/metadata/crispr.csv.gz",
        df_chrom_path="inputs/metadata/gene_chromosome_map.tsv",
    run:
        correct.corrections.annotate_dataframe(
            *input, *output, params.df_gene_path, params.df_chrom_path
        )


rule pca_transform:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_PCA.parquet",
    run:
        correct.corrections.transform_data(*input, *output)


rule correct_arm:
    input:
        "outputs/{scenario}/{pipeline}_annotated.parquet",
    output:
        "outputs/{scenario}/{pipeline}_corrected.parquet",
    params:
        gene_expression_path="inputs/metadata/Recursion_U2OS_expression_data.csv.gz",
    run:
        correct.corrections.arm_correction(*input, *output, params.gene_expression_path)


rule featselect:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_featselect.parquet",
    params:
        keep_image_features=config["keep_image_features"],
    run:
        pp.select_features(*input, *output, *params)


rule harmony:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_harmony.parquet",
    params:
        batch_key=config["batch_key"],
        thread_config = {
            "OPENBLAS_NUM_THREADS": "128",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        },
    run:
        correct.apply_harmony_correction(*input, *params, *output)

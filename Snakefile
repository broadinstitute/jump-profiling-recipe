wildcard_constraints:
    pipeline=r"[_a-zA-Z.~0-9\-]*",
    scenario=r"[_a-zA-Z.~0-9\-]*",

import logging

logging.basicConfig(level=logging.INFO)

import correct
import preprocessing as pp


include: "rules/sphering.smk"
include: "rules/map.smk"


rule all:
    input:
        f"outputs/{config['scenario']}/reformat.done",
        ap_negcon_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_ap_nonrep.parquet",
        map_negcon_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_map_nonrep.parquet",

rule reformat:
    input:
        f"outputs/{config['scenario']}/{config['pipeline']}.parquet",
    output:
        touch("outputs/{scenario}/reformat.done"),
    params:
        profile_dir=lambda w: f"outputs/{w.scenario}/",
        meta_col_new=config.get("meta_col_new", None),
    run:
        if "meta_col_new" in config:
            correct.format_check.run_format_check(params.profile_dir, params.meta_col_new)
        else:
            correct.format_check.run_format_check(params.profile_dir)


rule write_parquet:
    params:
        sources=config.get("sources", None),
        plate_types=config.get("plate_types", None),
        existing_profile_file=config.get("existing_profile_file", None),
    output:
        "outputs/{scenario}/profiles.parquet",
    run:
        if "existing_profile_file" in config:
            shell("mkdir -p $(dirname {output}) && cp {input} {output}".format(input=params.existing_profile_file, output=output))
        else:
            pp.io.write_parquet(
                params.sources,
                params.plate_types,
                *output
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


rule outlier_removal:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_outlier.parquet",
    run:
        pp.clean.outlier_removal(*input, *output)


rule annotate_genes:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_annotated.parquet",
    params:
        df_gene_path="inputs/crispr.csv.gz",
        df_chrom_path="inputs/gene_chromosome_map.tsv",
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
        gene_expression_path="inputs/Recursion_U2OS_expression_data.csv.gz",
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
    run:
        correct.harmony(*input, *params, *output)

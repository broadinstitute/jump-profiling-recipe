wildcard_constraints:
    pipeline=r"[_a-zA-Z.~0-9\-]*",
    scenario=r"[_a-zA-Z.~0-9\-]*",


import jump_profiling_recipe.correct as correct
import jump_profiling_recipe.preprocessing as pp


include: "rules/sphering.smk"
include: "rules/map.smk"


rule all:
    input:
        f"outputs/{config['scenario']}/reformat.done",
        # ap_negcon_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_ap_negcon.parquet",
        # map_negcon_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_map_negcon.parquet",
        ap_nonrep_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_ap_nonrep.parquet",
        map_nonrep_path=f"outputs/{config['scenario']}/metrics/{config['pipeline']}_map_nonrep.parquet",

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
    output:
        "outputs/{scenario}/profiles.parquet",
    params:
        existing_profile_file=config.get("existing_profile_file", None),
    benchmark:
        "benchmarks/{scenario}/write_parquet.txt"
    run:
        if "existing_profile_file" in config:
            shell("mkdir -p $(dirname {output}) && cp {input} {output}".format(input=params.existing_profile_file, output=output))
        else:
            pp.io.write_parquet(
                config["sources"],
                config["plate_types"],
                output[0],
                profile_type=config.get("profile_type"),
                search_additional_metadata=config.get("search_additional_metadata", False)
            )


rule compute_norm_stats:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/norm_stats/{pipeline}.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_normstats.txt"
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
    benchmark:
        "benchmarks/{scenario}/{pipeline}_var.txt"
    run:
        pp.stats.select_variant_features(*input, *output)


rule mad_normalize:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
        "outputs/{scenario}/norm_stats/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_mad.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_mad.txt"
    run:
        pp.normalize.mad(*input, *output)


rule INT:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_int.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_int.txt"
    run:
        pp.transform.rank_int(*input, *output)


rule well_correct:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_wellpos.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_wellpos.txt"
    run:
        correct.corrections.subtract_well_mean(*input, *output)


rule cc_regress:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_cc.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_cc.txt"
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
    benchmark:
        "benchmarks/{scenario}/{pipeline}_outlier.txt"
    run:
        pp.clean.remove_outliers(*input, *output)


rule drop_na_rows:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_dropna.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_dropna.txt"
    params:
        na_threshold=config.get("na_threshold", 0.1),
        max_rows_to_drop=config.get("max_rows_to_drop", 100),
    run:
        correct.corrections.remove_na_rows(
            *input,
            *output,
            na_threshold=params.na_threshold,
            max_rows_to_drop=params.max_rows_to_drop
        )


rule annotate_genes:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_annotated.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_annotated.txt"
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
    benchmark:
        "benchmarks/{scenario}/{pipeline}_PCA.txt"
    run:
        correct.corrections.transform_data(*input, *output)


rule correct_arm:
    input:
        "outputs/{scenario}/{pipeline}_annotated.parquet",
    output:
        "outputs/{scenario}/{pipeline}_corrected.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_corrected.txt"
    params:
        gene_expression_path="inputs/metadata/Recursion_U2OS_expression_data.csv.gz",
    run:
        correct.corrections.arm_correction(*input, *output, params.gene_expression_path)


rule featselect:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_featselect.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_featselect.txt"
    params:
        keep_image_features=config["keep_image_features"],
    run:
        pp.select_features(*input, *output, *params)


rule harmony:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_harmony.parquet",
    benchmark:
        "benchmarks/{scenario}/{pipeline}_harmony.txt"
    params:
        batch_key=config["batch_key"],
        thread_config = {
            "OPENBLAS_NUM_THREADS": "128",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        },
    run:
        correct.apply_harmony_correction(*input, *params, *output)

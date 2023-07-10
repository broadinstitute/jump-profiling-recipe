configfile: "./inputs/compound.json"


import correct
import preprocessing as pp


rule all:
    input:
        "outputs/jump_dataset/mad_int_featselect_harmony.parquet",


rule write_parquet:
    output:
        "outputs/{scenario}/raw.parquet",
    run:
        pp.io.write_parquet(config["sources"], config["plate_types"], *output)


rule compute_negcon_stats:
    input:
        "outputs/{scenario}/raw.parquet",
    output:
        "outputs/{scenario}/neg_stats.parquet",
    run:
        pp.stats.compute_negcon_stats(*input, *output)


rule select_variant_feats:
    input:
        "outputs/{scenario}/raw.parquet",
        "outputs/{scenario}/neg_stats.parquet",
    output:
        "outputs/{scenario}/variant_feats.parquet",
    run:
        pp.stats.select_variant_features(*input, *output)


rule mad_normalize:
    input:
        "outputs/{scenario}/variant_feats.parquet",
        "outputs/{scenario}/neg_stats.parquet",
    output:
        "outputs/{scenario}/mad.parquet",
    run:
        pp.normalize.mad(*input, *output)


rule compute_norm_stats:
    input:
        "outputs/{scenario}/mad.parquet",
    output:
        "outputs/{scenario}/norm_stats.parquet",
    run:
        pp.stats.compute_stats(*input, *output)


rule INT:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_int.parquet",
    run:
        pp.transform.rank_int(*input, *output)


rule featselect:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_featselect.parquet",
    run:
        pp.select_features(*input, *output)


rule harmony:
    input:
        "outputs/{scenario}/{pipeline}.parquet",
    output:
        "outputs/{scenario}/{pipeline}_harmony.parquet",
    params:
        batch_key=config["batch_key"],
    run:
        correct.harmony(*input, *params, *output)

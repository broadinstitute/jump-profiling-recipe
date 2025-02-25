rule average_precision_negcon:
    input:
        "outputs/{prefix}/{pipeline}.parquet",
    output:
        "outputs/{prefix}/metrics/{pipeline}_ap_negcon.parquet",
    benchmark:
        "benchmarks/{prefix}/{pipeline}_ap_negcon.txt"
    params:
        plate_types=config["plate_types"],
    run:
        pp.metrics.average_precision_negcon(*input, *output, **params)


rule average_precision_nonrep:
    input:
        "outputs/{prefix}/{pipeline}.parquet",
    output:
        "outputs/{prefix}/metrics/{pipeline}_ap_nonrep.parquet",
    benchmark:
        "benchmarks/{prefix}/{pipeline}_ap_nonrep.txt"
    params:
        plate_types=config["plate_types"],
    run:
        pp.metrics.average_precision_nonrep(*input, *output, **params)


rule mean_average_precision:
    input:
        "outputs/{prefix}/metrics/{pipeline}_ap_{reftype}.parquet",
    output:
        "outputs/{prefix}/metrics/{pipeline}_map_{reftype}.parquet",
    benchmark:
        "benchmarks/{prefix}/{pipeline}_map_{reftype}.txt"
    run:
        pp.metrics.mean_average_precision(*input, *output)

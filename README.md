# Workflow for JUMP dataset

This repository contains the source code to reproduce the preprocessing workflow
for COMPOUND, CRISPR and ORF data from the JUMP dataset.

## Installation

We suggest [Mamba](https://github.com/conda-forge/miniforge#mambaforge) for
environment management. The following commands create the environment from
scratch and install the required packages.

```bash
mamba env create --file environment.yaml
mamba activate jump_recipe
```

## Get data

Download profiles and metadata for `compound` (`crispr` or `orf`):

```bash
source download_data.sh compound
```

## Run workflow

```bash
snakemake -c1 --configfile inputs/compound.json
```

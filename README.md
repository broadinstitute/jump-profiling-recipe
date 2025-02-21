# Workflow for JUMP dataset

This repository contains the source code to reproduce the preprocessing workflow
for COMPOUND, CRISPR and ORF data from the JUMP dataset.

## Installation

We suggest [uv](https://docs.astral.sh/uv/) for environment management. The
following commands create the environment from scratch and install the required
packages.

```bash
uv sync
uv pip install -e .
source .venv/bin/activate
```

## Get data

Download profiles and metadata for `compound` (`crispr` or `orf`):

```bash
source download_data.sh compound
```

## Run workflow

```bash
snakemake -c1 --configfile inputs/config/compound.json
```

## Testing

To run the tests, first set your PYTHONPATH to include the repository root:

```bash
export PYTHONPATH=$(pwd)
pytest
```

The test suite includes an integration test to verify the pipeline's functionality using a minimal dataset.

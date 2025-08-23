# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JUMP Profiling Recipe is a Snakemake workflow for preprocessing and analyzing JUMP Cell Painting dataset profiles. The pipeline supports multiple perturbation types (compound, CRISPR, ORF) and includes steps for normalization, batch correction, feature selection, and metrics calculation.

## Commands

### Environment Setup
```bash
# Install dependencies using pixi
pixi install
pixi shell

# For development environment with additional tools
pixi install -e dev
pixi shell -e dev

# For GPU support (Linux only)
pixi install -e gpu
pixi shell -e gpu

# Test CUDA availability (GPU environment)
pixi run cuda-smoke
```

### Running the Workflow
```bash
# Run with default configuration (4 cores)
snakemake -c4 --configfile inputs/config/compound.json

# Dry run to preview operations
snakemake -n --configfile inputs/config/compound.json

# Run specific target
snakemake -c1 outputs/compound/profiles_var_mad_int_featselect_harmony.parquet --configfile inputs/config/compound.json

# Verbose execution for debugging
snakemake -c1 --configfile inputs/config/compound.json --verbose
```

### Testing
```bash
# Run tests using pixi
pixi run test

# Run tests with coverage
pytest --cov=jump_profiling_recipe tests/

# Run specific test file
pytest tests/test_pipeline_integration.py
```

### Linting and Code Quality
```bash
# Run ruff linter
ruff check src/ --config .ruff.toml

# Format code with ruff
ruff format src/ --config .ruff.toml

# Run pre-commit hooks
pre-commit run --all-files
```

### Data Download
```bash
# Download JUMP data for specific perturbation type
source download_data.sh compound  # or crispr, orf
```

## Architecture

### Core Components

**Snakefile** (`/Snakefile`): Main workflow definition that orchestrates all processing steps through rule dependencies. Pipeline string (e.g., `profiles_var_mad_int_featselect_harmony`) determines which rules execute.

**Configuration System** (`inputs/config/`): JSON files control workflow parameters. Key fields:
- `pipeline`: String encoding processing steps (e.g., `_var` for variant selection, `_mad` for MAD normalization)
- `scenario`: Output directory name for organizing results
- `sources`: Data generating centers to include
- `batch_key`: Column for Harmony batch correction

**Python Package** (`src/jump_profiling_recipe/`):
- `preprocessing/`: Feature selection, normalization, transformations
- `correct/`: Batch correction including Harmony GPU implementation (`harmony_gpu.py`)
- `cli/converter.py`: Tool for converting non-JUMP data to JUMP format

**Rules** (`rules/`):
- `sphering.smk`: Sphering transformation and parameter selection
- `map.smk`: Mean Average Precision metrics calculation

### Processing Pipeline Flow

The pipeline string determines processing order through Snakemake's dependency resolution:
1. **Base profiles** loaded from parquet files
2. **Preprocessing**: variant selection, MAD normalization, INT transformation
3. **Corrections**: well position, batch effects (Harmony), outlier removal
4. **Feature selection**: Removes non-informative features
5. **Metrics**: AP/MAP calculation for quality assessment

Each step creates intermediate files with suffixes matching the pipeline string components.

### Key Technical Details

- **Harmony Batch Correction**: Uses custom GPU implementation when available (`correct/harmony_gpu.py`)
- **Parallel Processing**: Rules support multi-core execution via Snakemake's `-c` flag
- **Feature Columns**: Controlled by `inputs/metadata/cpg0016_mandatory_feature_columns*.txt`
- **Metadata Integration**: Automatic gene annotation for CRISPR data, chromosome arm correction support

### Data Formats

- Input/Output: Parquet files for efficient columnar storage
- Metadata columns: Prefixed with `Metadata_`
- Feature columns: Morphological measurements from Cell Painting
- Key identifier: `Metadata_JCP2022` for perturbation tracking

## Important Notes

- Always check if test/lint commands exist before suggesting to run them
- The pipeline string in config files determines the entire processing flow
- GPU support requires Linux and CUDA-capable hardware
- Harmony correction is computationally intensive; GPU acceleration recommended for large datasets
- When modifying the pipeline, ensure intermediate file naming follows the suffix pattern

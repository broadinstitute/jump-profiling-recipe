#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Error: Missing argument."
  echo "Usage: $0 (orf|crispr|compound)"
  exit 1
fi

# Validate argument value
pert="$1"
if [[ ! "$pert" =~ ^(orf|crispr|compound)$ ]]; then
  echo "Error: Invalid argument. Please provide 'orf', 'crispr', or 'compound'."
  echo "Usage: $0 (orf|crispr|compound)"
  exit 1
fi

configfile="inputs/config/$pert.json"
BASEPATH="s3://cellpainting-gallery/cpg0016-jump"

readarray -t sources < <(jq -r '.sources[]' "$configfile")

mkdir -p inputs/metadata
for source_id in "${sources[@]}";
do
    aws s3 sync --no-sign-request "${BASEPATH}/${source_id}/workspace/profiles" inputs/${source_id}/workspace/profiles
done

wget https://raw.githubusercontent.com/jump-cellpainting/datasets/refs/tags/v0.9.0/metadata/plate.csv.gz -O inputs/metadata/plate.csv.gz
wget https://raw.githubusercontent.com/jump-cellpainting/datasets/refs/tags/v0.9.0/metadata/well.csv.gz -O inputs/metadata/well.csv.gz
wget https://raw.githubusercontent.com/jump-cellpainting/datasets/refs/tags/v0.9.0/metadata/${pert}.csv.gz -O inputs/metadata/${pert}.csv.gz

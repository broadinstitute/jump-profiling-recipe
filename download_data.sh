#!/bin/bash

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: $0 (orf|crispr|compound) [embedding_id]"
  exit 1
fi

# Validate argument value
pert="$1"
if [[ ! "$pert" =~ ^(orf|crispr|compound)$ ]]; then
  echo "Error: Invalid argument. Please provide 'orf', 'crispr', or 'compound'."
  echo "Usage: $0 (orf|crispr|compound) [embedding_id]"
  exit 1
fi

configfile="inputs/config/$pert.json"
BASEPATH="s3://cellpainting-gallery/cpg0016-jump"

readarray -t sources < <(jq -r '.sources[]' "$configfile")

# Set embedding_id if provided as second argument
embedding_id=""
if [ $# -eq 2 ]; then
    embedding_id="$2" # e.g. cpcnn_zenodo_7114558
fi

for source_id in "${sources[@]}";
do
    if [ -n "$embedding_id" ]; then
        aws s3 sync --no-sign-request "${BASEPATH}/${source_id}/workspace_dl/profiles/${embedding_id}" inputs/profiles_${embedding_id}/${source_id}/workspace/profiles
    else
        aws s3 sync --no-sign-request "${BASEPATH}/${source_id}/workspace/profiles" inputs/profiles/${source_id}/workspace/profiles
    fi
done

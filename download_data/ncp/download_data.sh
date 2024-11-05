#!/bin/bash

# Set variables for generalization
REPO_URL="https://github.com/broadinstitute/NeuroPainting.git"
REPO_VERSION="v0.0.1"
TARGET_DIR="inputs/workspace/profiles/ncp"
SOURCE_FILE="5.reanalysis/output/processed/augmented/combined.parquet"
REPO_DIR="$HOME/Downloads/NeuroPainting"

# Clone the repository if it doesn't already exist
if [ ! -d "${REPO_DIR}" ]; then
  git clone --branch "${REPO_VERSION}" "${REPO_URL}" "${REPO_DIR}"
fi

# Create necessary directories
mkdir -p "${TARGET_DIR}"

# Copy the specified file to the target location
cp "${REPO_DIR}/${SOURCE_FILE}" "${TARGET_DIR}/profiles.parquet"

echo "Repository cloned at version ${REPO_VERSION}, and file copied to ${TARGET_DIR}."
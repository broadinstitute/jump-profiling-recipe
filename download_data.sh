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

configfile="inputs/$pert.json"
BASEPATH="s3://cellpainting-gallery/cpg0016-jump"

readarray -t sources < <(jq -r '.sources[]' "$configfile")

mkdir -p inputs/metadata
for source_id in "${sources[@]}";
do
    aws s3 sync --no-sign-request "$BASEPATH/$source_id/workspace/profiles" inputs/$source_id/workspace/profiles
done

wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/plate.csv.gz?raw=true -O inputs/metadata/plate.csv.gz
wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/well.csv.gz?raw=true -O inputs/metadata/well.csv.gz
wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/$pert.csv.gz?raw=true -O inputs/metadata/$pert.csv.gz

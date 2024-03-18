pert="orf"

configfile="inputs/$pert.json"
BASEPATH="s3://cellpainting-gallery/cpg0016-jump"
# To download all of the sources in the JUMP dataset
# sources=`aws s3 ls --no-sign-request "$BASEPATH/" | awk '{print substr($2, 1, length($2)-1)}'`
# sources="source_2 source_3 source_6 source_8 source_10"

readarray -t sources < <(jq -r '.sources[]' "$configfile")

mkdir -p inputs/metadata
for source_id in "${sources[@]}";
do
    aws s3 cp --recursive --no-sign-request "$BASEPATH/$source_id/workspace/profiles" inputs/$source_id/workspace/profiles
done

wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/plate.csv.gz?raw=true -O inputs/metadata/plate.csv.gz
wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/well.csv.gz?raw=true -O inputs/metadata/well.csv.gz
wget https://github.com/jump-cellpainting/datasets/blob/main/metadata/$pert.csv.gz?raw=true -O inputs/metadata/$pert.csv.gz

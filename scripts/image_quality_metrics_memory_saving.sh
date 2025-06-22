#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompt_INFO.csv"
export SYNTHETIC_IMG_DIR="assets/synthetic_images/" # Assuming the synthetic images from your T2I model are in this folder. You can give a different folder path as well!

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="<NAME-OF-THE-T2I-MODEL>" # e.g. SD-V1-4, SD-V1-5, etc
export EXPERIMENT_TYPE="regular"            # Don't Change!

export RESULTS_SAVEDIR="Results/"
export SHARDS_DIR="Results/saved_shards"
mkdir -p  $SHARDS_DIR

export NUM_SHARDS=4
export SHARD=0

export BATCH_SIZE=64
export NUM_WORKERS=4

echo "Calculating regular metrics for $EXTRA_INFO"

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    export SHARD=$shard
    echo "Calculating FID, KID, IS for shard $SHARD ..."
    ./scripts/fid.sh
done

# Combine the shards here
python tools/combine_shards.py --shards_dir=$SHARDS_DIR --extra_info=$EXTRA_INFO --output_dir=$RESULTS_SAVEDIR --delete_after_combining 

# Calculate img-text alignment scores
echo "Calculating Image Text Alignment Scores ..."
./scripts/img_text_alignment.sh

# Calculate the FRD metric
./scripts/frd.sh
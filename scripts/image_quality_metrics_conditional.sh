#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompt_INFO.csv"
export SYNTHETIC_IMG_DIR="assets/synthetic_images/"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="<NAME-OF-THE-T2I-MODEL>" # e.g. SD-V1-4, SD-V1-5, etc
export EXPERIMENT_TYPE="conditional"        # Don't Change!

export NUM_SHARDS=-1
export SHARD=-1

export BATCH_SIZE=128
export NUM_WORKERS=4

MIMIC_PATHOLOGIES=("Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Enlarged Cardiomediastinum" "Fracture" "Lung Lesion" "Lung Opacity" "No Finding" "Pleural Effusion" "Pleural Other" "Pneumonia" "Pneumothorax" "Support Devices")

for pathology in "${MIMIC_PATHOLOGIES[@]}"; do
    export EXPERIMENT_TYPE="conditional"
    export PATHOLOGY=$pathology
    echo "Conditional Experiment for: '$PATHOLOGY'"
    echo "Calculating FID, KID, IS ..."
    ./scripts/fid.sh

    # echo "Calculating Image Text Alignment Scores ..."
    # ./scripts/img_text_alignment.sh
done
#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompt_INFO.csv"
export SYNTHETIC_IMG_DIR="assets/synthetic_images/" # Assuming the synthetic images from your T2I model are in this folder. You can give a different folder path as well!

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="<NAME-OF-THE-T2I-MODEL>" # e.g. SD-V1-4, SD-V1-5, etc
export EXPERIMENT_TYPE="regular"            # Don't Change!

export BATCH_SIZE=64
export NUM_WORKERS=4

echo "Calculating FID, KID, IS ..."
./scripts/fid.sh

echo "Calculating Image Text Alignment Scores ..."
./scripts/img_text_alignment.sh

echo "Calculating the FRD Metric ..."
./scripts/frd.sh
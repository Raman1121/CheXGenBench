#!/bin/bash

export REID_CKPT="assets/checkpoints/ResNet-50_epoch11_data_handling_RPN.pth"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export REAL_IMG_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"

export GEN_SAVEDIR="<PATH-TO-save-generated-images-across-seeds>" # Optional

export MODEL_PATH="<PATH-TO-trained-model-pipeline>"
export MODEL_NAME="<MODEL-NAME>" # e.g. SD-V1-5, SD-V1-4, etc  (Check the 'SUPPORTED_MODELS' constant in the metrics/privacy_metrics.py file)
export EXTRA_INFO="SD-V1-4"
export PROMPT_COL="annotated_prompt" # Don't Change!

export RESULTS_SAVEDIR="Results/"   

export SUBSET=2000

python metrics/privacy_metrics.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --reid_ckpt=$REID_CKPT \
    --real_csv=$REAL_CSV \
    --real_img_dir=$REAL_IMG_DIR \
    --gen_savedir=$GEN_SAVEDIR \
    --results_savedir=$RESULTS_SAVEDIR \
    --subset=$SUBSET \
    --extra_info=$EXTRA_INFO \
    --prompt_col=$PROMPT_COL
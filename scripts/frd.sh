#!/bin/bash

export SYNTHETIC_CSV=""
export SYNTHETIC_IMG_DIR=""
export SYNTHETIC_IMG_COL="synthetic_filename"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXPERIMENT_TYPE="regular"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python metrics/frd.py \
                                            --synthetic_csv=$SYNTHETIC_CSV \
                                            --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
                                            --synthetic_img_col=$SYNTHETIC_IMG_COL \
                                            --real_csv=$REAL_CSV \
                                            --real_img_dir=$REAL_IMG_DIR \
                                            --experiment_type=$EXPERIMENT_TYPE \
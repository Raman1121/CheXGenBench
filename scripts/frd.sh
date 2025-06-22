#!/bin/bash

echo "SYNTHETIC CSV CSV: $SYNTHETIC_CSV"
echo "SYNTHETIC IMG DIR: $SYNTHETIC_IMG_DIR"
echo "SYNTHETIC IMG COL: $SYNTHETIC_IMG_COL"
echo "REAL CSV: $REAL_CSV"
echo "REAL IMG DIR: $REAL_IMG_DIR"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python metrics/frd.py \
                                            --synthetic_csv=$SYNTHETIC_CSV \
                                            --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
                                            --synthetic_img_col=$SYNTHETIC_IMG_COL \
                                            --real_csv=$REAL_CSV \
                                            --real_img_dir=$REAL_IMG_DIR \
                                            --experiment_type=$EXPERIMENT_TYPE \
                                            --results_savedir=$RESULTS_SAVEDIR \
                                            --extra_info=$EXTRA_INFO \
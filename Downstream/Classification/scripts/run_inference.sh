#!/bin/bash

echo "Loading from checkpoint: $CHECKPOINT"

python Downstream/Classification/downstream_inference.py \
        --model_name=$MODEL \
        --checkpoint=$CHECKPOINT \
        --test_csv=$TEST_CSV \
        --real_image_dir=$REAL_IMAGE_DIR \
        --synthetic_image_dir=$SYNTHETIC_IMG_DIR \
        --save_predictions \
        --extra_info=$EXTRA_INFO \
        --t2i_model=$T2I_MODEL
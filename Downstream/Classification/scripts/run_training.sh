#!/bin/bash

python Downstream/Classification/train_downstream_classification.py \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --model_name=$MODEL \
        --train_csv=$TRAIN_CSV \
        --real_image_dir=$REAL_IMAGE_DIR \
        --synthetic_image_dir=$SYNTHETIC_IMG_DIR \
        --extra_info=$EXTRA_INFO \
        --training_setting=$TRAINING_SETTING \
        --t2i_model=$T2I_MODEL \
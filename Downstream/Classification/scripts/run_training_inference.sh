#!/bin/bash

########################################### Training on 20K Synthetic samples ###########################################
echo "RUNNING EXPERIMENT: ALL SYNTHETIC DATA"
export MODEL="resnet50" 
export T2I_MODEL="SD-V1-4" # Example
export BATCH_SIZE=768
export TRAIN_CSV="<PATH-TO>/generations_with_metadata.csv"
export TEST_CSV="<PATH-TO>/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMAGE_DIR="<PATH-TO>/physionet.org/files/mimic-cxr-jpg/2.0.0"
export SYNTHETIC_IMG_DIR="<PATH-TO-SYNTHETIC-IMAGES>"
export TRAINING_SETTING="all_synthetic"
export export EXTRA_INFO=$TRAINING_SETTING

export EPOCHS=20

export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}_${T2I_MODEL}.pth"

echo "Training the model..."
chmod +x Downstream/Classification/scripts/run_training.sh
./Downstream_Training/run_training.sh > logs_ds_training_synthetic_${MODEL}_${T2I_MODEL}.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
chmod +x Downstream/Classification/scripts/run_inference.sh
./Downstream_Training/run_inference.sh > logs_ds_inference_synthetic_${MODEL}_${T2I_MODEL}.txt 2>&1
echo "Inference Finished!!"
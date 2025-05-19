#!/bin/bash

########################################### 1. Training on 20K REAL samples (baseline) ###########################################
echo "RUNNING EXPERIMENT: ALL REAL DATA"
export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export T2I_MODEL="sana"
export BATCH_SIZE=768
export TRAIN_CSV="/pvc/SYNTHETIC_IMAGES_NEW/sana/generations_with_metadata.csv"
export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES_NEW/sana/generations_with_metadata.csv"
export TRAINING_SETTING="all_original"
export EXTRA_INFO=$TRAINING_SETTING
export EPOCHS=20

export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}_${T2I_MODEL}.pth"

echo "Training the model..."
chmod +x Downstream_Training/run_training.sh
./Downstream_Training/run_training.sh > logs_ds_training_original_${MODEL}.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
chmod +x Downstream_Training/run_inference.sh
./Downstream_Training/run_inference.sh > logs_ds_inference_original_${MODEL}.txt 2>&1
echo "Inference Finished!!"
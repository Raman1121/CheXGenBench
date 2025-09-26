#!/bin/bash

# Make sure the environment 'myenv' is activated!

echo "SYNTHETIC CSV CSV: $SYNTHETIC_CSV"
echo "SYNTHETIC IMG DIR: $SYNTHETIC_IMG_DIR"
echo "SYNTHETIC PROMPT COL: $SYNTHETIC_PROMPT_COL"
echo "REAL CSV: $REAL_CSV"
echo "REAL IMG DIR: $REAL_IMG_DIR"
echo "Saving Results at: $RESULTS_SAVEDIR"

python metrics/fid.py --synthetic_csv=$SYNTHETIC_CSV \
                        --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
                        --synthetic_prompt_col=$SYNTHETIC_PROMPT_COL \
                        --real_csv=$REAL_CSV \
                        --real_img_dir=$REAL_IMG_DIR \
                        --results_savedir=$RESULTS_SAVEDIR \
                        --batch_size=$BATCH_SIZE \
                        --num_workers=$NUM_WORKERS \
                        --extra_info=$EXTRA_INFO \
                        --experiment_type=$EXPERIMENT_TYPE \
                        --pathology="$PATHOLOGY" \
                        --num_shards=$NUM_SHARDS \
                        --shard=$SHARD \
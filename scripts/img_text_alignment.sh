#!/bin/bash

# Make sure a different environment is activated!
# conda activate himl

echo "SYNTHETIC CSV CSV: $SYNTHETIC_CSV"
echo "REAL PROMPT COL: $SYNTHETIC_PROMPT_COL"
echo "SYNTHETIC IMG DIR: $SYNTHETIC_IMG_DIR"
echo "Saving Results at: $RESULTS_SAVEDIR"

python metrics/img_text_alignment_scores.py \
    --real_csv=$REAL_CSV \
    --synthetic_csv=$SYNTHETIC_CSV \
    --real_caption_col=$SYNTHETIC_PROMPT_COL \
    --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
    --results_savedir=$RESULTS_SAVEDIR \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --extra_info=$EXTRA_INFO \
    --experiment_type=$EXPERIMENT_TYPE \
    --pathology="$PATHOLOGY" \
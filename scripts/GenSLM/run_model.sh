#!/bin/bash

# script used to score GenSLM model
# the directory containing the GenSLM checkpoints
# downloaded from https://github.com/ramanathanlab/genslm?tab=readme-ov-file#usage
# should contain a file named 2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
export checkpoint_dir="/home/ma_run_ze/lzm/rnagym/fitness/baselines/GenSLM/checkpoints/"
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv.old"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/old_data/scores/genslm/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/old_data/fitness_processed_assays"

# Get the current index from the array (0-31)

for DMS_index in $(seq 0 31); do
    # Run the scoring script with the array task ID
    python compute_fitness.py \
        --reference_sheet "$reference_sheet" \
        --task_id "$DMS_index" \
        --checkpoint_dir "$checkpoint_dir" \
        --dms_directory "$dms_data_dir" \
        --output_directory "$output_scores_dir"
    echo "Scoring for DMS index $DMS_index completed."
done

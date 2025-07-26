#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/enformer/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
touch "$output_scores_dir/correlation_summary.txt"
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=0

    # Run the scoring script with the array task ID
    python run.py \
        --task_id 45 \
        --reference_sheet "$reference_sheet" \
        --dms_directory "$dms_data_dir" \
        --output_directory "$output_scores_dir" 
    echo "Scoring for DMS index $DMS_index completed."


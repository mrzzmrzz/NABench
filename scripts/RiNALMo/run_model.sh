#!/bin/bash

# script used to score RiNALMo model
conda activate rinalmo

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/rinalmo/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
export CUDA_VISIBLE_DEVICES=6  # Set to the GPU you want to use
# Get the current index from the array (0-31)
DMS_index=$((44))  # Change this to the desired DMS index

    # Run the scoring script with the array task ID
    python compute_fitness.py \
        --reference_sheet "$reference_sheet" \
        --task_id "$DMS_index"
        
    echo "Scoring for DMS index $DMS_index completed."


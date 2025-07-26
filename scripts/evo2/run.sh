#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)
export model_name="evo2_40b"

export reference_sheet="/home/zhangzy/evo2/reference_sheet_final.csv"
export output_scores_dir="/home/zhangzy/evo2/scores-40B"
export dms_data_dir="/home/zhangzy/evo2/fitness_processed_assays"
export CUDA_VISIBLE_DEVICES=5,6,7
mkdir -p "$output_scores_dir"
rm -f $output_scores_dir/*.csv
rm -f $output_scores_dir/correlation_summary.txt
touch "$output_scores_dir/correlation_summary.txt"
# Get the current index from the array (0-31)


    # Run the scoring script with the array task ID
    python run.py \
        --row_id 44 \
        --ref_sheet "$reference_sheet" \
        --dms_dir_path "$dms_data_dir" \
        --output_dir_path "$output_scores_dir" \
        --model_name "$model_name"



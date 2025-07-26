#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)
export model_name="evo-1.5-8k-base"

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/evo/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
rm -f $output_scores_dir/*.csv
rm -f $output_scores_dir/correlation_summary.txt
touch "$output_scores_dir/correlation_summary.txt"
# Get the current index from the array (0-31)


    # Run the scoring script with the array task ID
    python score_evo_single_dms.py \
        --row_id 44 \
        --ref_sheet "$reference_sheet" \
        --dms_dir_path "$dms_data_dir" \
        --output_dir_path "$output_scores_dir" \
        --model_name "$model_name"
    echo "Scoring for DMS index $DMS_index completed."


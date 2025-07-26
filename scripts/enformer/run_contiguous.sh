#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/enformer/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export cache_dir="/data4/ma_run_ze/enformer/"
mkdir -p "$output_scores_dir"
touch "$output_scores_dir/correlation_summary.txt"
export HF_ENDPOINT=https://hf-mirror.com
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=6

    # Run the scoring script with the array task ID
    python run_contiguous.py \
        --ref_sheet "$reference_sheet" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --cache_dir "/data4/ma_run_ze/enformer/" \
        --n_folds 5 \
        --max_test_frac 0.20
    echo "Scoring for DMS index $DMS_index completed."


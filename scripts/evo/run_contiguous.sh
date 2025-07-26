#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)
export model_name="evo-1.5-8k-base"

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/evo/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export cache_dir="/data3/marunze/evo/cache/"
mkdir -p "$cache_dir"
# Get the current index from the array (0-31)


    # Run the scoring script with the array task ID
    python run_contiguous.py \
        --ref_sheet "$reference_sheet" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --model_name "$model_name" \
        --cache_dir "$cache_dir" \
        --n_folds 5 \
        --max_test_frac 0.20\
        --batch_size 64
    echo "Scoring for DMS index $DMS_index completed."


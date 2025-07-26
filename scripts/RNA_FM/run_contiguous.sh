#!/bin/bash

# the directory containing RNA-FM model
# downloaded fromc 
export model_location="/home/ma_run_ze/lzm/rnagym/fitness/baselines/RNA_FM/RNA-FM"

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/RNA-FM"
export CUDA_VISIBLE_DEVICES=7
export cache_dir="/data3/marunze/rna_fm/cache/"
mkdir -p "$cache_dir"
mkdir -p "$output_scores_dir"
# for DMS_index in $(seq 0 31); do
    python run_contiguous.py  \
    --model_location "$model_location" \
    --reference_sequences "$reference_sheet" \
    --dms_directory "$dms_data_dir" \
    --output_directory "$output_scores_dir"\
    --cache_dir "$cache_dir" \
    --n_folds 5 \
    --max_test_frac 0.20 
# done
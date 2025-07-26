#!/bin/bash

# RNAErnie model
export model_checkpoint="src/"
export vocab_path="src/vocab_1MER.txt"
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays/"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/RNAErnie/"
export CUDA_VISIBLE_DEVICES=7
export cache_dir="/data3/marunze/rna_ernie/cache/"
mkdir -p "$output_scores_dir"


python run_contiguous.py  \
    --model_checkpoint "$model_checkpoint" \
    --reference_sequences "$reference_sheet" \
    --dms_directory "$dms_data_dir" \
    --output_directory "$output_scores_dir" \
    --vocab_path "$vocab_path" \
    --n_folds 5 \
    --max_test_frac 0.20 

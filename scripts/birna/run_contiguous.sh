#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/birna/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export cache_dir="/data3/marunze/birna/cache/"
mkdir -p "$output_scores_dir"
# https://huggingface.co/AIRI-Institute/bert-large-t2t
export model_location="buetnlpbio/birna-bert"
DMS_index=$((44))  # Change this to the desired DMS index
export HF_ENDPOINT=https://hf-mirror.com
    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=6  # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python run_contiguous.py \
        --ref_sheet "$reference_sheet" \
        --model_location "$model_location" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --cache_dir  "$cache_dir" \
        --n_folds 5 \
        --max_test_frac 0.20

    echo "Scoring for DMS index $DMS_index completed."

# DMS_index="10"
# # Run the scoring script with the array task ID
#     python compute_fitness.py \
#         --reference_sheet "$reference_sheet" \
#         --task_id "$DMS_index" \
#         --model_location "$model_location" \
#         --dms_directory "$dms_data_dir" \
#         --output_directory "$output_scores_dir"
#     echo "Scoring for DMS index $DMS_index completed."
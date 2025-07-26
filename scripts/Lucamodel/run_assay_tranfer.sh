#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_trans/LucaOne_assay/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export llm_dir="/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/lucamodel/lucaone/checkpoint-step36000000/"
export cache_dir="/data4/marunze/lucaone/cache"
# Clear the cache directory
mkdir -p "$cache_dir"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
DMS_index=$((44))  # Change this to the desired DMS index

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=5  # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python assay_transfer_learning.py \
        --ref_sheet "$reference_sheet" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --llm_dir "$llm_dir" \
        --cache_dir "$cache_dir" \

echo "Scoring for DMS index $DMS_index completed."
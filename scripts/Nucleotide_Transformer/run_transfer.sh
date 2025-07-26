#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_transfer/nucleotide_transformer/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export cache_dir="/data4/ma_run_ze/nucleotide_transformer/"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
export model_location="InstaDeepAI/nucleotide-transformer-500m-human-ref"
DMS_index=$((45))  # Change this to the desired DMS index
export HF_ENDPOINT=https://hf-mirror.com
    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=7  # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python transfer_learning.py \
        --ref_sheet "$reference_sheet" \
        --model_location "$model_location" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --cache_dir "$cache_dir" \
        --batch_size 16 


    echo "Scoring for DMS index $DMS_index completed."

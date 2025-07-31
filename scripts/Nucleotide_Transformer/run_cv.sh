#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv/nucleotide_transformer-v2/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
export model_location="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DMS_index=$((44))  # Change this to the desired DMS index
export HF_ENDPOINT=https://hf-mirror.com
    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=2 # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python compute_fitness.py \
        --reference_sheet "$reference_sheet" \
        --task_id "$DMS_index" \
        --model_location "$model_location" \
        --dms_directory "$dms_data_dir" \
        --output_directory "$output_scores_dir"\
        --cv

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
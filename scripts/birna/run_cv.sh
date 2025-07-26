#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/birna_cv/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
DMS_index=$((44))  # Change this to the desired DMS index

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=7  # Set to the GPU you want to use

  
python run_cv.py \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path  "$dms_data_dir" \
    --output_dir_path  "$output_scores_dir" \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --folds 3 \

#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv/LucaVirus/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export llm_dir="/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/lucamodel/lucavirus/"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
DMS_index=$((44))  # Change this to the desired DMS index

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=0  # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python run_virus.py \
        --ref_sheet "$reference_sheet" \
        --dms_dir "$dms_data_dir" \
        --output_dir "$output_scores_dir" \
        --llm_dir "$llm_dir" \
        --cv 
echo "Scoring for DMS index $DMS_index completed."
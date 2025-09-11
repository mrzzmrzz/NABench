#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
DMS_index=$((44))  # Change this to the desired DMS index
export HF_ENDPOINT=https://hf-mirror.com
source /data4/marunze/aido/bin/activate

export CUDA_VISIBLE_DEVICES=1  # Set to the GPU you want to use
    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)




export output_dir="/data_share/marunze/lzm/rnagym/fitness/embeddings/aido/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_dir"
# Get the current index from the array (0-31)

    # Run the scoring script with the array task ID
    python seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \


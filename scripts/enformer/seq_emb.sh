#!/bin/bash

# script used to score RiNALMo model
conda activate enformer

export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_dir="/data_share/marunze/lzm/rnagym/fitness/embeddings/enformer/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES=3  # Set to the GPU you want to use
# Get the current index from the array (0-31)
export HF_ENDPOINT=https://hf-mirror.com
    # Run the scoring script with the array task ID
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \


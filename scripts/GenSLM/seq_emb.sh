#!/bin/bash


export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_dir="/data_share/marunze/lzm/rnagym/fitness/embeddings/genslm/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
conda activate genslm
mkdir -p "$output_dir"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1  # Set to the GPU you want to use
# Get the current index from the array (0-31)
    # Run the scoring script with the array task ID
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --model_cache_dir "/data4/marunze/GenSLM/"

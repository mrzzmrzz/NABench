#!/bin/bash


export reference_sheet=""
export output_dir=""
export dms_data_dir=""
conda activate genslm
mkdir -p "$output_dir"
export model_location=export model_location="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --model_location "$model_location"

#!/bin/bash
conda activate nt
export reference_sheet=""
export output_dir=""
export dms_data_dir=""
mkdir -p "$output_dir"
export model_location=export model="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --model_location "$model"

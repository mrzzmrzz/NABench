#!/bin/bash
conda activate rnaernie
export model_location=""
export reference_sheet=""
export output_dir=""
export dms_data_dir=""
export model_checkpoint=""
export vocab_path=""
mkdir -p "$output_dir"
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --model_checkpoint "$model_checkpoint" \
    --vocab_path "$vocab_path" 

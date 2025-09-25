#!/bin/bash
conda activate enformer
export reference_sheet=""
export output_dir=""
export dms_data_dir=""
mkdir -p "$output_dir"
    python3 seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" 

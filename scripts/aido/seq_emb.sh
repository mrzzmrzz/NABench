#!/bin/bash


export reference_sheet=""
export dms_data_dir=""
mkdir -p "$output_scores_dir"
DMS_index=$((44)) 
source /data4/marunze/aido/bin/activate


export output_dir=""
export dms_data_dir=""
mkdir -p "$output_dir"
    python seq_emb.py \
    --output_dir_path "$output_dir" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \

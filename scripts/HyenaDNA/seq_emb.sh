#!/bin/bash


export reference_sheet=""
export output_scores_dir=""
export dms_data_dir=""
mkdir -p "$output_scores_dir"
DMS_index=$((44))
conda activate hyena-dna
    python seq_emb.py \
        --ref_sheet "$reference_sheet" \
        --dms_dir_path "$dms_data_dir" \
        --output_dir_path "$output_scores_dir"
    echo "Scoring for DMS index $DMS_index completed."

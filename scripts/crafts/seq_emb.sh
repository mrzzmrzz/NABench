#!/bin/bash


export reference_sheet=""
export output_scores_dir=""
export dms_data_dir=""
mkdir -p "$output_scores_dir"
conda activate crafts

python seq_emb.py \
  --model_name_or_path  \
  --ref_sheet "$reference_sheet" \
  --dms_dir_path "$dms_data_dir" \
  --output_dir_path "$output_scores_dir" \
  --batch_size 64

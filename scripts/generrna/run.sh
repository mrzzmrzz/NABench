#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/GENERNA/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=6  # Set to the GPU you want to use


python run_inference.py \
  --model_ckpt /home/ma_run_ze/lzm/rnagym/fitness/baselines/generrna/GenerRNA/model_updated.pt \
  --tokenizer_dir "GenerRNA/tokenizer" \
  --ref_sheet "$reference_sheet" \
  --dms_dir_path "$dms_data_dir" \
  --output_dir_path "$output_scores_dir" \
  --dms_column "DMS_score" \
  --sequence_column "mutant" \
  --batch_size 32
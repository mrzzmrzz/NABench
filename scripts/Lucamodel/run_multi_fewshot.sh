#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_few_shots/lucaone_multi/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=7  # Set to the GPU you want to use
export HF_ENDPOINT=https://hf-mirror.com

python run_multi_fewshot.py \
  --k_list 10 20 30 40 50 100 150 200 250 300\
  --few_shot_repeat 20 \
  --out_root "$output_scores_dir" \


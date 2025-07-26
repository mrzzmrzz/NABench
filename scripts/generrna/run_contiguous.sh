#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/GENERNA/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export cache_dir="/data4/marunze/generrna/cache/"
mkdir -p "$cache_dir"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=7  # Set to the GPU you want to use
export HF_ENDPOINT=https://hf-mirror.com

python run_contiguous.py \
  --model_ckpt /data4/marunze/generrna/GenerRNA/model_updated.pt \
  --tokenizer_dir "/data4/marunze/generrna/GenerRNA/tokenizer" \
  --ref_sheet "$reference_sheet" \
  --dms_dir "$dms_data_dir" \
  --output_dir "$output_scores_dir" \
  --cache_dir "$cache_dir" \
  --n_folds 5 \
  --max_test_frac 0.20 
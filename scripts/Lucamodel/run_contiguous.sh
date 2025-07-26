#!/bin/bash
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores_cv_con/LucaOne_contiguous/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export llm_dir="/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/lucamodel/lucaone/checkpoint-step36000000/"
export cache_dir="/data4/marunze/lucaone/cache/"

mkdir -p "$output_scores_dir"
export CUDA_VISIBLE_DEVICES=5

python lucaone_eval_contiguous.py \
  --ref_sheet "$reference_sheet" \
  --dms_dir    "$dms_data_dir" \
  --output_dir "$output_scores_dir" \
  --llm_dir    "$llm_dir"  \
  --cache_dir  "$cache_dir" \
  --embedding_type vector 
echo "Scoring completed for LucaOne contiguous model."
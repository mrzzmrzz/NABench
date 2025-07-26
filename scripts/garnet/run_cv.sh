#!/bin/bash

# script used to score Nucleotide Transformer model
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/garnet-cv/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
export checkpoint_path="/home/ma_run_ze/lzm/rnagym/fitness/baselines/garnet/checkpoints/231RNAs_Thermo_triples_finetune_0_18_6_300_rot_flash.pt"
mkdir -p "$output_scores_dir"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
DMS_index=$((44))  # Change this to the desired DMS index

    # Remove existing score files and create a new summary file
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=1  # Set to the GPU you want to use

    # Run the scoring script with the array task ID
    python ../runcv.py \
        --model_checkpoint  "$checkpoint_path" \
        --ref_sheet "$reference_sheet" \
        --dms_dir_path "$dms_data_dir" \
        --tokenization_type triples \
        --output_dir "$output_scores_dir" \
        --epochs 3 \
        --lr 1e-5 \
        --folds 3 \
        --batch_size 64 
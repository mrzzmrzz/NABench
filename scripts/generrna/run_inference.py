#!/usr/bin/env python3
"""
Script to run GenerRNA model inference on all Deep Mutational Scanning (DMS) 
datasets listed in a reference sheet.

This script loads a GenerRNA checkpoint, then iterates through each assay
in the reference sheet, processes the corresponding RNA sequences, and evaluates
the model by correlating its log-likelihood scores with experimental data.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

# Add the GenerRNA directory to the Python path to import its modules
# Assumes this script is in the same parent directory as the GenerRNA folder
sys.path.append(str(Path(__file__).parent.resolve()))

from GenerRNA.model import GPT, GPTConfig
from transformers import AutoTokenizer

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run GenerRNA model inference on all DMS datasets in a reference sheet."
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to the GenerRNA model checkpoint file (ckpt.pt)."
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="GenerRNA/tokenizer",
        help="Path to the directory containing GenerRNA tokenizer files."
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to the DMS experiment reference sheet (CSV) listing all assays to process."
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Path to the directory containing all DMS dataset CSV files."
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="./outputs",
        help="Directory path to save the output results."
    )
    parser.add_argument(
        "--dms_column",
        type=str,
        default="DMS_score",
        help="Column name in the DMS file containing experimental scores."
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="mutated_sequence",
        help="Column name in the DMS file containing mutated sequences."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use for inference."
    )
    return parser.parse_args()

def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses an RNA sequence:
    - Converts to uppercase
    - Strips leading/trailing whitespace
    
    Args:
        sequence: The input RNA sequence.
        
    Returns:
        str: The preprocessed RNA sequence.
    """
    return sequence.strip().upper()

def get_log_likelihood(model, tokens, device):
    """
    Calculate the log-likelihood of a given sequence using the GenerRNA model.
    
    Args:
        model: The GenerRNA model.
        tokens (list): A list of token IDs for the sequence.
        device: 'cuda' or 'cpu'.
        
    Returns:
        float: The log-likelihood of the entire sequence.
    """
    if not tokens or len(tokens) < 2:
        return -np.inf # Cannot compute likelihood

    input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits, loss = model(input_ids, targets)
    
    log_likelihood = -loss.item() * (len(tokens) - 1)
    return log_likelihood

def main():
    """Main execution function"""
    args = parse_args()
    
    # --- 1. Setup Environment and Load Model (once) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    args.output_dir_path = Path(args.output_dir_path)
    args.output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading GenerRNA model from '{args.model_ckpt}'...")
    if not Path(args.model_ckpt).exists():
        print(f"Error: Model checkpoint file not found: {args.model_ckpt}")
        return
        
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    print(f"Loading tokenizer from '{args.tokenizer_dir}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    except Exception as e:
        print(f"Error: Failed to load tokenizer: {e}")
        return

    # --- 2. Load Reference Sheet and Loop Through All Assays ---
    ref_sheet = pd.read_csv(args.ref_sheet)
    print(f"\nFound {len(ref_sheet)} assays to process in '{args.ref_sheet}'.")

    summary_filepath = args.output_dir_path / "summary_results.csv"
    if summary_filepath.exists():
        print(f"Summary file found at {summary_filepath}. Appending results.")

    for row_id, assay_info in ref_sheet.iterrows():
        print(f"\n{'='*20} Processing row {row_id}: Target '{assay_info.get('Target', 'N/A')}' {'='*20}")
        
        try:
            # --- Per-assay Data Loading ---
            dms_filename = assay_info['DMS_ID'] + '.csv'
            dms_file = Path(args.dms_dir_path) / dms_filename
            if not dms_file.exists():
                print(f"Warning: DMS file '{dms_filename}' not found for row {row_id}. Skipping.")
                continue

            print(f"Loading DMS data from: {dms_file}")
            dms_df = pd.read_csv(dms_file)

            if 'RAW_CONSTRUCT_SEQ' not in assay_info or pd.isna(assay_info['RAW_CONSTRUCT_SEQ']):
                print(f"Warning: Wild-type sequence (wt_seq) not provided for row {row_id}. Skipping.")
                continue
            wt_sequence = preprocess_sequence(assay_info['RAW_CONSTRUCT_SEQ'])

            # --- Inference Logic ---
            wt_tokens = tokenizer.encode(wt_sequence)
            log_p_wt = get_log_likelihood(model, wt_tokens, device)

            all_scores = []
            sequences_to_process = dms_df[args.sequence_column].tolist()
            
            print(f"Starting inference on {len(sequences_to_process)} mutant sequences...")
            for seq_str in tqdm(sequences_to_process, desc=f"Evaluating mutants for row {row_id}", leave=False):
                mut_sequence = preprocess_sequence(str(seq_str))
                mut_tokens = tokenizer.encode(mut_sequence)
                log_p_mut = get_log_likelihood(model, mut_tokens, device)
                score = log_p_mut - log_p_wt
                all_scores.append(score)

            dms_df['model_score'] = all_scores
            
            # --- Calculate Correlation and Save Results ---
            final_df = dms_df.dropna(subset=[args.dms_column, 'model_score'])
            
            if len(final_df) > 1:
                correlation, p_value = spearmanr(final_df[args.dms_column], final_df['model_score'])
                print(f"Spearman's correlation (ρ): {correlation:.4f} (p-value: {p_value:.4g})")
            else:
                correlation = np.nan
                print("Not enough data to calculate correlation.")

            output_filename = f"generrna_results_row_{row_id}_{dms_filename.replace('.csv', '')}.csv"
            output_filepath = args.output_dir_path / output_filename
            dms_df.to_csv(output_filepath, index=False)
            print(f"Individual results saved to: {output_filepath}")

            summary_data = {
                'row_id': row_id,
                'target': assay_info.get('Target', 'N/A'),
                'spearman_correlation': correlation,
                'num_sequences': len(final_df),
                'model_ckpt': Path(args.model_ckpt).name,
                'dms_file': dms_filename
            }
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(summary_filepath, mode='a', header=not summary_filepath.exists(), index=False)

        except Exception as e:
            print(f"An unexpected error occurred while processing row {row_id}: {e}. Skipping.")
            raise e

    print(f"\n{'='*20} All assays processed. {'='*20}")
    print(f"Summary of all results saved to: {summary_filepath}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluate a trained GenerRNA model on DMS assay sequences.

This script is self-contained and does NOT require a meta.pkl file.
It generates the tokenizer on-the-fly based on the --tokenization_type argument.
It is CRITICAL that the selected tokenization type EXACTLY matches the one
used to train the model checkpoint.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm

# Add LM_Model directory to path to import model components
sys.path.append('./LM_Model')
try:
    from model_RNA_rot import GPTConfig, GPT
except ImportError:
    print("Error: Could not import GPTConfig or GPT from 'LM_Model/model_RNA_rot.py'.")
    print("Please ensure you are running this script from the root of the GARNET_DL directory.")
    sys.exit(1)

# --- Sequence Cleaning Function (from prepare_RNA_nonoverlapping_val.py) ---
def rna_AUGCgap_only(string):
    """
    Cleans up RNA fasta formatting. Leaves '-' for MSA.
    Replaces non-standard nucleotides with standard ones.
    """
    for char in ['R', 'N', 'W', 'V', 'D', 'Z', 'n', 'a']:
        string = string.replace(char, 'A')
    for char in ['Y', 'K', 'B', 'H', 'u', 'T']:
        string = string.replace(char, 'U')
    for char in ['M', 'S', 'c']:
        string = string.replace(char, 'C')
    string = string.replace('g', 'G')
    for char in ['\n', ' ', '.']:
        string = string.replace(char, '')
    return string

# --- Vocabulary Generation Functions ---
def get_vocab(tokenization_type):
    """Generates stoi and itos dictionaries based on the tokenization type."""
    nucleotides = ['A', 'U', 'G', 'C']
    if tokenization_type == 'single':
        chars = nucleotides + ['5', '3', '-']
    elif tokenization_type == 'pairs':
        chars = [a + b for a in nucleotides for b in nucleotides]
        chars.extend(['5' + b for b in nucleotides])
        chars.extend([a + '3' for a in nucleotides])
        chars += ['--']
    elif tokenization_type == 'triples':
        chars = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides]
        chars.extend(['5' + b + c for b in nucleotides for c in nucleotides])
        chars.extend([a + b + '3' for a in nucleotides for b in nucleotides])
        chars += ['---']
    elif tokenization_type == 'quadruples':
        chars = [a + b + c + d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
        chars.extend(['5' + b + c + d for b in nucleotides for c in nucleotides for d in nucleotides])
        chars.extend([a + b + c + '3' for a in nucleotides for b in nucleotides for c in nucleotides])
        chars += ['----']
    elif tokenization_type == 'triples_MSA':
        # Base triples
        chars = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides]
        chars.extend(['5' + b + c for b in nucleotides for c in nucleotides])
        chars.extend([a + b + '3' for a in nucleotides for b in nucleotides])
        chars += ['---']
        # MSA additions
        chars.extend(['-' + b + c for b in nucleotides for c in nucleotides])
        chars.extend([a + '-' + c for a in nucleotides for c in nucleotides])
        chars.extend([a + b + '-' for a in nucleotides for b in nucleotides])
        chars.extend(['--' + c for c in nucleotides])
        chars.extend(['-' + b + '-' for b in nucleotides])
        chars.extend([a + '--' for a in nucleotides])
        chars.extend(['5-' + c for c in nucleotides])
        chars.extend(['5' + b + '-' for b in nucleotides])
        chars.extend([a + '-3' for a in nucleotides])
        chars.extend(['-' + b + '3' for b in nucleotides])
        chars += ['5--', '--3']
    else:
        raise ValueError(f"Unknown tokenization_type: {tokenization_type}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

# --- Tokenization Functions ---
def tokenize_sequence(sequence, stoi, token_size):
    """Tokenizes a sequence based on token size and a provided stoi map."""
    tokens = []
    padding = '-' * token_size
    for i in range(len(sequence) - token_size + 1):
        token_str = sequence[i:i + token_size]
        tokens.append(stoi.get(token_str, stoi.get(padding)))
    return tokens

# --- Main Logic ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GenerRNA model on DMS data with on-the-fly tokenization.")
    parser.add_argument("--row_id", type=int, default=0, help="Row ID in the reference sheet to process.")
    parser.add_argument("--ref_sheet", type=str, required=True, help="Path to the reference sheet CSV file.")
    parser.add_argument("--dms_dir_path", type=str, required=True, help="Path to the directory containing DMS data files.")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Path to the directory to save output files.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the GenerRNA model checkpoint (.pt file).")
    parser.add_argument("--tokenization_type", type=str, required=True, choices=['single', 'pairs', 'triples', 'quadruples', 'triples_MSA'],
                        help="MUST match the tokenization used for model training.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing sequences.")
    return parser.parse_args()

def calculate_score(model, mut_seq, wt_seq, wt_score, mutations, token_size, stoi, device):
    mut_tokens = tokenize_sequence(mut_seq, stoi, token_size)
    mut_tensor = torch.tensor([mut_tokens], dtype=torch.long, device=device)
    mut_score = model.sequence_probability(mut_tensor)
    mut_score = mut_score.squeeze().cpu().numpy()
    return (mut_score - wt_score).item()

def calculate_batch_scores(model, mut_seqs, wt_score, token_size, stoi, device):
    """
    Calculate scores for a batch of mutant sequences.
    
    Args:
        model: The GenerRNA model
        mut_seqs: List of mutant sequences
        wt_score: Wild-type sequence score
        token_size: Token size for tokenization
        stoi: String-to-index mapping
        device: Device to run inference on
    
    Returns:
        List of scores for each mutant sequence
    """
    # Tokenize all sequences in the batch
    batch_tokens = []
    for seq in mut_seqs:
        tokens = tokenize_sequence(seq, stoi, token_size)
        batch_tokens.append(tokens)
    
    # Convert to tensor
    batch_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
    
    # Get batch scores
    batch_scores = model.sequence_probability(batch_tensor)
    batch_scores = batch_scores.cpu().numpy()
    
    # Calculate difference from wild-type
    scores = [(score - wt_score).item() for score in batch_scores]
    return scores

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")

    # --- Generate Tokenizer On-the-Fly ---
    print(f"Generating tokenizer for type: '{args.tokenization_type}'")
    stoi, itos = get_vocab(args.tokenization_type)
    token_size_map = {'single': 1, 'pairs': 2, 'triples': 3, 'quadruples': 4, 'triples_MSA': 3}
    token_size = token_size_map[args.tokenization_type]
    print(f"Vocabulary size: {len(stoi)}")

    # --- Load Model ---
    print(f"Loading model from {args.model_checkpoint}")
    try:
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])

        # --- Sanity Check ---
        if gptconf.vocab_size != len(stoi):
            print("\n" + "="*80)
            print("FATAL ERROR: VOCABULARY SIZE MISMATCH!")
            print(f"The model was trained with a vocabulary size of {gptconf.vocab_size}.")
            print(f"The generated tokenizer for '--tokenization_type {args.tokenization_type}' has a size of {len(stoi)}.")
            print("These values MUST be identical. Please ensure you are using the correct tokenization type.")
            print("="*80 + "\n")
            sys.exit(1)

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
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_checkpoint}")
        sys.exit(1)
    for curr_row in range(0, args.row_id):
        # --- Load and Process Data ---
        ref_df = pd.read_csv(args.ref_sheet)
        row = ref_df.iloc[curr_row]
        print(f"Processing row {curr_row}: {row['DMS_ID']}")
        wt_seq_cleaned = rna_AUGCgap_only(row['RAW_CONSTRUCT_SEQ'])
        dms_file_path = args.dms_dir_path +"/"+ row['DMS_ID'] + '.csv'
        dms_df = pd.read_csv(dms_file_path)
        wt_score = model.sequence_probability(
            torch.tensor([tokenize_sequence(wt_seq_cleaned, stoi, token_size)], dtype=torch.long, device=device)
        ).squeeze().cpu().numpy()
        print(f"Wild-type sequence score: {wt_score}")


        model_scores = []
        with torch.no_grad():
            # Process sequences in batches for better speed
            for i in tqdm(range(0, len(dms_df), args.batch_size), desc="Scoring mutants in batches"):
                batch_end = min(i + args.batch_size, len(dms_df))
                batch_rows = dms_df.iloc[i:batch_end]
                
                # Clean sequences for the batch
                mut_seqs_cleaned = [rna_AUGCgap_only(row['sequence']) for _, row in batch_rows.iterrows()]
                
                # Calculate batch scores
                batch_scores = calculate_batch_scores(model, mut_seqs_cleaned, wt_score, token_size, stoi, device)
                model_scores.extend(batch_scores)

        dms_df['model_score'] = model_scores
        
        # --- Evaluate and Save ---
        results_df = dms_df.dropna(subset=['DMS_score', 'model_score'])
        if len(results_df) > 1:
            spearman_corr, p_value = spearmanr(results_df['DMS_score'], results_df['model_score'])
            print(f"Spearman Correlation: {spearman_corr:.4f}, P-value: {p_value:.4g}")
            output_dir = Path(args.output_dir_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            corr_path = output_dir / f"{Path(row['DMS_ID']).stem}_correlation.csv"
            pd.DataFrame([{'spearman_correlation': spearman_corr, 'p_value': p_value}]).to_csv(corr_path, index=False)
        else:
            print("Not enough valid data to calculate correlation.")
        summary_path = output_dir / f"correlation_summary.csv"
        if not summary_path.exists():
            with open(summary_path, 'w') as f:
                f.write("DMS_ID,spearman_correlation,p_value\n")
        with open(summary_path, 'a') as f:
            f.write(f"{row['DMS_ID']},{spearman_corr:.4f},{p_value:.4g}\n")
        scores_path = Path(args.output_dir_path) / f"{Path(row['DMS_ID']).stem}_scores.csv"
        dms_df.to_csv(scores_path, index=False)
        print(f"Results saved to {args.output_dir_path}")

if __name__ == "__main__":
    main()
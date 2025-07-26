#!/usr/bin/env python3
"""
Fine-tune and evaluate a trained GenerRNA model on DMS assay sequences
using K-fold cross-validation.

This script is adapted from a zero-shot evaluation script, preserving its
on-the-fly tokenization and sequence probability scoring, while adding a
fine-tuning loop for domain adaptation.

In each fold:
1. The pre-trained model is loaded.
2. It's fine-tuned on the training split using a Causal LM objective.
3. The fine-tuned model scores the validation split using the original
   sequence_probability method.
4. An overall Spearman correlation is computed on out-of-fold predictions.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Add LM_Model directory to path to import model components
# This assumes the script is run from the root of the GARNET_DL repository
sys.path.append('./LM_Model')
try:
    from model_RNA_rot import GPTConfig, GPT
except ImportError:
    print("Error: Could not import GPTConfig or GPT from 'LM_Model/model_RNA_rot.py'.")
    print("Please ensure you are running this script from the root of the GARNET_DL directory.")
    sys.exit(1)


# --- Reused Functions from Original Script ---

def rna_AUGCgap_only(string):
    """Cleans up RNA fasta formatting."""
    if not isinstance(string, str):
        return ""
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
    else: # Fallback for other k-mers or MSA variants
        # This part can be extended based on the full logic from the original script if needed
        base_chars = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides]
        start_chars = ['5' + b + c for b in nucleotides for c in nucleotides]
        end_chars = [a + b + '3' for a in nucleotides for b in nucleotides]
        msa_chars = ['---']
        chars = base_chars + start_chars + end_chars + msa_chars
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def tokenize_sequence(sequence, stoi, token_size):
    """Tokenizes a sequence based on token size and a provided stoi map."""
    tokens = []
    # Use a generic padding token for unseen k-mers
    padding_token = '-' * token_size
    padding_idx = stoi.get(padding_token)

    for i in range(len(sequence) - token_size + 1):
        token_str = sequence[i:i + token_size]
        tokens.append(stoi.get(token_str, padding_idx))
    return tokens


# --- New/Adapted Components for Fine-tuning ---

class FineTuneDataset(Dataset):
    """Dataset for fine-tuning, returns tokenized sequences."""
    def __init__(self, sequences, stoi, token_size, max_size=384):
        self.sequences = sequences
        self.stoi = stoi
        self.token_size = token_size
        self.max_size = max_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = tokenize_sequence(seq, self.stoi, self.token_size)
        # Pad or truncate to max_size
        if len(tokens) > self.max_size:
            tokens = tokens[:self.max_size]

        return torch.tensor(tokens, dtype=torch.long)

def run_finetuning_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Runs a single epoch of Causal LM fine-tuning."""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Fine-tuning epoch"):
        batch = batch.to(device)            # shape [B, T]
        print(f"Batch shape: {batch.shape}")
        print("Type of batch:", batch.dtype)
        logits, loss = model(batch, targets=batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def score_sequences(model, sequences, token_size, stoi, batch_size, device):
    """
    Scores a list of sequences using the model's sequence_probability method.
    This function is a wrapper around the original scoring logic.
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring validation set"):
            batch_seqs = sequences[i:i + batch_size]
            
            batch_tokens = [tokenize_sequence(seq, stoi, token_size) for seq in batch_seqs]
            
            # Handle empty token lists which can result from short sequences
            if not all(batch_tokens):
                print(f"Warning: Skipping some sequences in batch {i//batch_size} due to being shorter than token size.")
                batch_tokens = [tokens for tokens in batch_tokens if tokens]
                if not batch_tokens: continue

            batch_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            
            batch_scores = model.sequence_probability(batch_tensor)
            scores.extend(batch_scores.cpu().numpy())
            
    return np.array(scores)


# --- Main Logic ---

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate GenerRNA model on DMS data.")
    # Paths
    parser.add_argument("--row_id", type=int, default=None, help="Row ID in the reference sheet to process.")
    parser.add_argument("--ref_sheet", type=str, required=True, help="Path to the reference sheet CSV file.")
    parser.add_argument("--dms_dir_path", type=str, required=True, help="Path to the directory containing DMS data files.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the GenerRNA model checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to save output files.")
    # Tokenization
    parser.add_argument("--tokenization_type", type=str, required=True, choices=['single', 'pairs', 'triples', 'quadruples', 'triples_MSA'],
                        help="MUST match the tokenization used for model training.")
    # Fine-tuning params
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs per fold.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for scoring.")
    # System
    parser.add_argument("--device", type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. Setup Tokenizer (Reused from original script) ---
    print(f"Generating tokenizer for type: '{args.tokenization_type}'")
    stoi, itos = get_vocab(args.tokenization_type)
    token_size_map = {'single': 1, 'pairs': 2, 'triples': 3, 'quadruples': 4, 'triples_MSA': 3}
    token_size = token_size_map[args.tokenization_type]
    print(f"Vocabulary size: {len(stoi)}, Token size: {token_size}")
    if args.row_id:
        run_list =[args.row_id]
    else:
        # If no row_id is specified, process all rows in the reference sheet
        ref_sheet = pd.read_csv(args.ref_sheet)
        run_list = range(ref_sheet.index.tolist()[-1])
    print(f"Total rows to process: {run_list}")
    for curr_row_id in run_list:
        # --- 2. Load Data ---



        ref_df = pd.read_csv(args.ref_sheet)
        row = ref_df.iloc[curr_row_id]
        print(f"Processing row {curr_row_id}: {row['DMS_ID']}")

        dms_file_path = args.dms_dir_path +"/"+ row['DMS_ID'] + '.csv'
        dms_df = pd.read_csv(dms_file_path)


        sequences= [rna_AUGCgap_only(row['sequence']) for _, row in dms_df.iterrows()]
        # --- 3. Cross-Validation Loop ---
        oof_preds = np.zeros(len(sequences))
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
            print(f"\n----- Processing Fold {fold + 1}/{args.folds} -----")
            
            # --- Load Model for each fold ---
            checkpoint = torch.load(args.model_checkpoint, map_location=device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            if gptconf.vocab_size != len(stoi):
                raise ValueError(f"Vocab size mismatch! Model: {gptconf.vocab_size}, Tokenizer: {len(stoi)}")
            
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            model.to(device)

            # --- Fine-tuning Phase ---
            train_seqs = [sequences[i] for i in train_idx]
            train_dataset = FineTuneDataset(train_seqs, stoi, token_size)
            # Filter out any empty token lists from the dataset before creating loader
            train_dataset.sequences = [s for s in train_dataset.sequences if len(s) >= token_size]
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                print(f"--- Fine-tuning Epoch {epoch + 1}/{args.epochs} ---")
                run_finetuning_epoch(model, train_loader, optimizer, device)

            # --- Evaluation Phase ---
            val_seqs = [sequences[i] for i in val_idx]
            # Use the fine-tuned model to score validation set
            val_scores = score_sequences(model, val_seqs, token_size, stoi, args.eval_batch_size, device)
            oof_preds[val_idx] = val_scores

        # --- 4. Final Analysis & Saving ---
        dms_df['model_score'] = oof_preds
        results_df = dms_df.dropna(subset=['DMS_score', 'model_score'])

        print("\n----- Cross-Validation Complete -----")
        if len(results_df) > 1:
            spearman_corr, p_value = spearmanr(results_df['DMS_score'], results_df['model_score'])
            print(f"Overall Spearman Correlation: {spearman_corr:.4f}, P-value: {p_value:.4g}")
        else:
            print("Not enough valid data to calculate final correlation.")
            spearman_corr, p_value = np.nan, np.nan

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scores_path = output_dir / f"{row['DMS_ID']}_scores.csv"
        dms_df.to_csv(scores_path, index=False)
        print(f"Saved detailed scores to {scores_path}")

        summary_path = output_dir / "correlation_summary.csv"
        if not summary_path.exists():
            with open(summary_path, 'w') as f:
                f.write("DMS_ID,spearman_correlation,p_value\n")
        with open(summary_path, 'a') as f:
            f.write(f"{row['DMS_ID']},{spearman_corr:.4f},{p_value:.4g}\n")
        print(f"Appended correlation summary to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fine-tuning script for the GARNET Language Model (LM) on DMS datasets.

This script implements K-fold cross-validation to fine-tune the custom
Transformer-based Language Model from the GARNET_DL repository.

In each fold:
1. The pre-trained LM is loaded.
2. It is fine-tuned on the training split using a Causal Language Modeling objective
   (next-token prediction).
3. The fine-tuned model scores the validation split by computing sequence
   log-likelihoods.
4. Finally, an overall Spearman correlation is computed between the out-of-fold
   predictions and the experimental DMS scores.
"""

import argparse
import os
import sys
from pathlib import Path
import random
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# --- Dynamically add GARNET_DL to Python Path ---
script_dir = Path(__file__).resolve().parent
garnet_dir = script_dir / 'GARNET_DL'
if garnet_dir.exists() and str(script_dir) not in sys.path:
    # We add the parent of GARNET_DL to resolve imports like `LM_Model.model_RNA_rot`
    sys.path.insert(0, str(script_dir))
    print(f"Added {script_dir} to Python path for GARNET_DL imports.")

# --- Import from GARNET_DL codebase ---
sys.path.append('./LM_Model')
try:
    from LM_Model.model_RNA_rot import Model_RNA_rot
    from LM_Model.configurator import configurator
except ImportError as e:
    print("Failed to import modules from GARNET_DL/LM_Model.")
    print("Please ensure GARNET_DL is in your PYTHONPATH or this script is in its parent directory.")
    sys.exit(f"ImportError: {e}")


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DMSSequenceDataset(Dataset):
    """Custom PyTorch Dataset for DMS sequences."""
    def __init__(self, sequences, tokenizer_map):
        self.sequences = sequences
        self.tokenizer_map = tokenizer_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Tokenize sequence manually based on the provided map
        tokens = [self.tokenizer_map.get(char, self.tokenizer_map.get('N', 4)) for char in seq]
        return torch.tensor(tokens, dtype=torch.long)


def run_finetuning_epoch(model, dataloader, optimizer, criterion, device, grad_clip):
    """Runs a single epoch of Causal LM fine-tuning."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Fine-tuning epoch"):
        # Causal LM: input is sequence, target is sequence shifted by 1
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(inputs)
        # Reshape for CrossEntropyLoss: (batch_size * seq_len, num_classes)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def score_sequences_with_lm(model, dataloader, criterion, device):
    """Scores sequences using log-likelihood with the fine-tuned LM."""
    model.eval()
    log_likelihoods = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring validation set"):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            logits, _ = model(inputs)
            
            # Calculate loss (negative log likelihood) for each item in the batch
            # We need to calculate it per sequence, so we set reduction='none'
            per_token_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Reshape back to (batch_size, seq_len) and sum to get per-sequence NLL
            per_sequence_nll = per_token_loss.view(logits.size(0), -1).sum(dim=1)
            
            # Log likelihood is the negative of the NLL
            log_likelihoods.extend(-per_sequence_nll.cpu().numpy())
            
    return np.array(log_likelihoods)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune GARNET LM model.")
    # --- Paths ---
    parser.add_argument("--dms_csv", type=str, required=True, help="Path to the DMS CSV file with 'sequence' and 'dms_score' columns.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the pre-trained LM checkpoint (.pt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuning results.")
    # --- Fine-tuning Parameters ---
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs per fold.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    # --- CV & Model Parameters ---
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dms_id = Path(args.dms_csv).stem
    
    print("Loading DMS data...")
    df = pd.read_csv(args.dms_csv)
    df.columns = df.columns.str.lower()
    df.dropna(subset=['dms_score', 'sequence'], inplace=True)
    
    sequences = [seq.strip().upper().replace('U', 'T') for seq in df['sequence']]
    labels = df['dms_score'].values

    # Tokenizer map from GARNET code
    tokenizer_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'M': 5}
    
    # Use configurator to get model parameters
    config = configurator(args.model_checkpoint)
    
    oof_preds = np.zeros(len(df))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n----- Processing Fold {fold + 1}/{args.folds} -----")
        
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]

        print("Loading pre-trained model checkpoint...")
        model = Model_RNA_rot(config)
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # --- Fine-tuning Phase ---
        train_dataset = DMSSequenceDataset(train_seqs, tokenizer_map)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            print(f"--- Epoch {epoch+1}/{args.epochs} ---")
            avg_loss = run_finetuning_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
            print(f"Fold {fold+1}, Epoch {epoch+1} Average NLL Loss: {avg_loss:.4f}")

        # --- Evaluation Phase ---
        val_dataset = DMSSequenceDataset(val_seqs, tokenizer_map)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        # Use a criterion with no reduction for scoring
        scoring_criterion = nn.CrossEntropyLoss(reduction='none')
        
        val_scores = score_sequences_with_lm(model, val_loader, scoring_criterion, device)
        oof_preds[val_idx] = val_scores

    # --- Final Analysis ---
    print("\n----- Cross-Validation Summary -----")
    df['model_score'] = oof_preds
    final_df = df.dropna(subset=['dms_score', 'model_score'])
    
    overall_correlation, pvalue = spearmanr(final_df['dms_score'], final_df['model_score'])
    print(f"Overall Spearman Correlation for {dms_id}: {overall_correlation:.4f} (p-value: {pvalue:.2e})")

    results_file = output_dir / f"{dms_id}_predictions.csv"
    df.to_csv(results_file, index=False)
    print(f"Saved predictions to: {results_file}")

    summary_file = output_dir / "correlation_summary.csv"
    is_new_file = not summary_file.exists()
    with open(summary_file, 'a') as f:
        if is_new_file:
            f.write("DMS_ID,Spearman_Correlation\n")
        f.write(f"{dms_id},{overall_correlation:.4f}\n")


if __name__ == "__main__":
    main()
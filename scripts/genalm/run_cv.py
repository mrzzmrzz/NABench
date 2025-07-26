#!/usr/bin/env python3
"""
Script to fine-tune and evaluate Gena-LM (BERT-style) models on DMS assays.
Fine-tuning is performed using a Masked Language Modeling (MLM) objective.
Evaluation is done by scoring sequences based on pseudo-log-likelihood and
correlating with DMS scores, all within a cross-validation framework.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate Gena-LM on DMS assays using MLM."
    )
    # --- Data and Path Arguments ---
    parser.add_argument(
        "--row_id",
        type=int,
        default=None,
        help="Row ID in the reference sheet to process. If not provided, all rows will be processed."
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to reference sheet containing DMS_ID and sequence columns."
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing DMS CSV files with mutation and score data."
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save fine-tuning outputs (checkpoints, logs, results)."
    )
    parser.add_argument(
        "--model_location",
        type=str,
        required=True,
        help="Hugging Face model identifier, e.g., 'AIRI-Institute/gena-lm-bert-large-t2t'."
    )
    # --- Training Hyperparameters ---
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run training on."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for scoring/evaluation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs for fine-tuning. For MLM, 1 is often sufficient."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate for AdamW optimizer."
    )
    # --- Acceleration and CV Arguments ---
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients."
    )
    return parser.parse_args()

def load_data(ref_sheet_path, dms_dir_path, row_id):
    """Loads reference data and corresponding DMS file."""
    try:
        ref_df = pd.read_csv(ref_sheet_path, encoding='latin-1')
        if row_id >= len(ref_df):
            raise ValueError(f"Row ID {row_id} is out of bounds.")
        
        assay_info = ref_df.iloc[row_id]
        # Standardize column names for robustness
        assay_info.rename({assay_info.index[0]: 'DMS_ID'}, inplace=True)
        dms_id = assay_info['DMS_ID']

        dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
        if not dms_file.exists():
            raise FileNotFoundError(f"DMS file not found: {dms_file}")
            
        dms_df = pd.read_csv(dms_file)
        dms_df.columns = dms_df.columns.str.lower()
        dms_df.dropna(subset=['dms_score', 'sequence'], inplace=True)
        return dms_id, dms_df
    except Exception as e:
        print(f"Error loading data for row {row_id}: {e}")
        raise

class SequenceTextDataset(Dataset):
    """Simple dataset that returns a list of sequences for MLM."""
    def __init__(self, sequences, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        encoding = self.tokenizer(
            self.sequences[i],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze()
        }

def score_sequences_with_model(model, tokenizer, sequences, batch_size, device):
    """Scores a list of sequences using pseudo-log-likelihood."""
    model.eval()
    model.to(device)
    
    scores = []
    max_length = model.config.max_position_embeddings

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring validation set"):
            batch = sequences[i:i+batch_size]
            
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True
            ).to(device)

            logits = model(**tokens).logits
            log_probs_full = F.log_softmax(logits, dim=-1)

            for j in range(log_probs_full.shape[0]):
                input_ids = tokens['input_ids'][j]
                valid_len = (input_ids != tokenizer.pad_token_id).sum()
                
                # Gather the log-probabilities of the actual tokens
                log_probs_for_seq = log_probs_full[j, torch.arange(valid_len), input_ids[:valid_len]]
                
                avg_log_prob = log_probs_for_seq.mean().item()
                scores.append(avg_log_prob)
    
    return np.array(scores)

def run_cross_validation(args, dms_id, dms_df, tokenizer):
    """Manages the n-fold cross-validation process for a single DMS assay."""
    output_dir = Path(args.output_dir_path)
    
    sequences = [seq.strip().upper().replace('U', 'T') for seq in dms_df["sequence"]]
    labels = dms_df["dms_score"].values

    oof_preds = np.zeros(len(dms_df)) * np.nan
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    max_len = AutoConfig.from_pretrained(args.model_location).max_position_embeddings

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n----- Processing Fold {fold + 1}/{args.folds} for DMS ID: {dms_id} -----")
        fold_output_dir = output_dir / dms_id / f"fold_{fold+1}"

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        
        train_dataset = SequenceTextDataset(train_seqs, tokenizer, max_len)
        
        # We don't need a formal eval dataset for MLM trainer, loss on train is enough
        # The true evaluation happens via scoring after training.
        
        model = AutoModel.from_pretrained(args.model_location,trust_remote_code=True)
        
        # Data collator will take care of randomly masking tokens
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="no", # We don't need to save checkpoints unless for resume
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        print(f"Starting MLM fine-tuning for fold {fold + 1}...")
        trainer.train()
        
        print("Fine-tuning complete. Scoring validation sequences...")
        # After training, use the fine-tuned model for scoring
        fold_scores = score_sequences_with_model(
            model, tokenizer, val_seqs, args.eval_batch_size, args.device
        )
        oof_preds[val_idx] = fold_scores

    print(f"\n----- Cross-Validation Summary for {dms_id} -----")
    dms_df['model_score'] = oof_preds
    final_df = dms_df.dropna(subset=['dms_score', 'model_score'])
    
    if len(final_df) < 2:
        print("Not enough valid predictions to compute final correlation.")
        overall_correlation = 0.0
    else:
        overall_correlation, _ = spearmanr(final_df['dms_score'], final_df['model_score'])
        print(f"Overall Spearman Correlation: {overall_correlation:.4f}")

    results_file = output_dir / f"{dms_id}_predictions.csv"
    dms_df.to_csv(results_file, index=False)
    print(f"Saved predictions to: {results_file}")
    return overall_correlation

def main():
    """Main execution function."""
    args = parse_args()
    
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "correlation_summary.csv"
    if not summary_file.exists():
        with open(summary_file, 'w') as f:
            f.write("DMS_ID,Spearman_Correlation\n")
    
    print(f"Loading tokenizer from {args.model_location}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_location)

    if args.row_id is not None:
        rows_to_process = [args.row_id]
    else:
        print("No specific --row_id. Processing all rows in the reference sheet.")
        total_rows = len(pd.read_csv(args.ref_sheet, encoding='latin-1'))
        rows_to_process = range(total_rows)

    for row_id in rows_to_process:
        dms_id = ""
        try:
            dms_id, dms_df = load_data(args.ref_sheet, args.dms_dir_path, row_id)
            print(f"\n{'='*60}\nProcessing Assay: {dms_id} (Row ID: {row_id})\n{'='*60}")
            
            if len(dms_df) < args.folds:
                print(f"Skipping {dms_id}: Not enough data ({len(dms_df)}) for {args.folds}-fold CV.")
                continue

            correlation = run_cross_validation(args, dms_id, dms_df, tokenizer)

            with open(summary_file, 'a') as f:
                f.write(f"{dms_id},{correlation:.4f}\n")

        except Exception as e:
            print(f"FATAL: Failed to process row {row_id} (DMS_ID: {dms_id or 'N/A'}). Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

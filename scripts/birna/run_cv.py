#!/usr/bin/env python3
"""
Script to fine-tune and evaluate BiRNA-BERT model on DMS assay sequences
for fitness prediction using 5-fold cross-validation.
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, EvalPrediction
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate BiRNA-BERT on DMS assay sequences."
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
        default="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv",
        help="Path to reference sheet containing DMS_ID column."
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        default="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays",
        help="Directory containing DMS CSV files."
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save fine-tuning outputs (checkpoints, logs, results)."
    )
    # --- Training Hyperparameters ---
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run training on (default: cuda:0 if available, else cpu)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation (default: 16)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for fine-tuning (default: 3)."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW optimizer (default: 2e-5)."
    )
    # --- Acceleration and CV Arguments ---
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation. A common choice for speed is 3. (default: 5)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing an optimization step. (default: 1)"
    )
    parser.add_argument(
        "--torch_compile",
        action='store_true',
        help="Enable torch.compile() for model optimization (requires PyTorch 2.0+)."
    )
    return parser.parse_args()

def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """Loads DMS_ID for a given row from the reference sheet."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if row_id >= len(ref_df):
            raise ValueError(f"Row ID {row_id} is out of bounds for the reference sheet.")
        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}.")
        return str(dms_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    except KeyError:
        raise KeyError("Reference sheet must contain a 'DMS_ID' column.")

def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """Loads and validates the DMS data for a given DMS_ID."""
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")
    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DMS file must contain columns: {required_cols}")
    df.dropna(subset=['DMS_score', 'sequence'], inplace=True)
    return df

def preprocess_sequence(sequence: str) -> str:
    """Cleans and standardizes a DNA/RNA sequence."""
    return sequence.strip().upper().replace('U', 'T')

class RegressionDataset(Dataset):
    """
    PyTorch Dataset for sequence regression tasks.
    Takes sequences, labels, and a tokenizer.
    """
    def __init__(self, sequences, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(p: EvalPrediction) -> dict:
    """
    Computes Spearman correlation for regression tasks.
    This function is passed to the Trainer.
    """
    preds = p.predictions.flatten()
    labels = p.label_ids.flatten()
    
    valid_indices = ~np.isnan(preds) & ~np.isnan(labels)
    preds = preds[valid_indices]
    labels = labels[valid_indices]

    if len(preds) < 2:
        return {"spearmanr": 0.0}

    spearman_corr, _ = spearmanr(labels, preds)
    return {"spearmanr": spearman_corr if not np.isnan(spearman_corr) else 0.0}


def run_cross_validation(args, dms_id, dms_df, tokenizer):
    """
    Manages the n-fold cross-validation process for a single DMS assay.
    """
    output_dir = Path(args.output_dir_path)
    
    sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"]]
    labels = dms_df["DMS_score"].values.astype(np.float32)

    oof_preds = np.zeros(len(dms_df)) * np.nan
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    model_name = "buetnlpbio/birna-bert"
    config = AutoConfig.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=False
        )
    model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=False
    )
    for fold, (train_index, val_index) in enumerate(kf.split(sequences)):
        print(f"\n----- Processing Fold {fold + 1}/{args.folds} for DMS ID: {dms_id} -----")
        

        train_seqs = [sequences[i] for i in train_index]
        train_labels = labels[train_index]
        val_seqs = [sequences[i] for i in val_index]
        val_labels = labels[val_index]

        
        if args.torch_compile:
            print("Enabling torch.compile() for model acceleration.")
            if hasattr(torch, "compile"):
                model = torch.compile(model)
            else:
                print("Warning: torch.compile() is not available in this PyTorch version. Continuing without it.")

        model.to(args.device)

        max_len = config.max_position_embeddings
        print(f"Using model's max sequence length: {max_len}")
        
        train_dataset = RegressionDataset(train_seqs, train_labels, tokenizer, max_length=max_len)
        val_dataset = RegressionDataset(val_seqs, val_labels, tokenizer, max_length=max_len)
        
        steps_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size

        training_args = TrainingArguments(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
            # Compatibility for older transformers versions
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="spearmanr",
            greater_is_better=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        print(f"Starting fine-tuning for fold {fold + 1}...")
        trainer.train()
        
        print(f"Evaluating on validation set for fold {fold + 1}...")
        predictions = trainer.predict(val_dataset)
        oof_preds[val_index] = predictions.predictions.flatten()

    print(f"\n----- Cross-Validation Summary for {dms_id} -----")
    dms_df['model_score'] = oof_preds
    
    final_df = dms_df.dropna(subset=['DMS_score', 'model_score'])
    
    if len(final_df) < 2:
        print("Not enough valid predictions to compute final correlation.")
        overall_correlation = 0.0
    else:
        overall_correlation, pvalue = spearmanr(final_df['DMS_score'], final_df['model_score'])
        print(f"Overall Spearman Correlation: {overall_correlation:.4f} (p-value: {pvalue:.2e})")

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
    
    print("Loading BiRNA-BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")

    if args.row_id is not None:
        rows_to_process = [args.row_id]
    else:
        print("No specific --row_id provided. Processing all rows in the reference sheet.")
        ref_df = pd.read_csv(args.ref_sheet)
        rows_to_process = range(len(ref_df))

    for row_id in rows_to_process:
        dms_id = ""
        try:
            dms_id = load_reference_data(args.ref_sheet, row_id)
            print(f"\n{'='*60}\nProcessing Assay: {dms_id} (Row ID: {row_id})\n{'='*60}")
            
            dms_df = load_dms_data(args.dms_dir_path, dms_id)

            # Check if there are enough samples for cross-validation
            if len(dms_df) < 200:
                print(f"Skipping {dms_id}: Not enough data points ({len(dms_df)}) for {args.folds}-fold CV.")
                continue

            correlation = run_cross_validation(args, dms_id, dms_df, tokenizer)

            with open(summary_file, 'a') as f:
                f.write(f"{dms_id},{correlation:.4f}\n")

        except Exception as e:
            print(f"Failed to process row {row_id} (DMS_ID: {dms_id or 'N/A'}). Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

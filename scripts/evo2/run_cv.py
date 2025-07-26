#!/usr/bin/env python3
"""
Script to fine-tune and evaluate an Evo2 model on a single DMS assay
using 5-fold cross-validation.
"""
import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# Add bionemo path to sys.path if it's not already there.
# This is required to import the model architecture for loading checkpoints.
# You may need to adjust this path depending on your environment.
BIONEMO_PATH = os.environ.get("BIONEMO_PATH", "/workspaces/bionemo-framework/3rdparty/NeMo")
if BIONEMO_PATH not in sys.path:
    sys.path.append(BIONEMO_PATH)

from nemo.collections.llm.gpt.model.megatron_gpt_model import MegatronGPTModel
from evo2.scoring import prepare_batch, logits_to_logprobs
from evo2 import Evo2 as Evo

# --- Argument Parsing ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate Evo2 model on a DMS assay using K-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input/Output Arguments ---
    parser.add_argument("--dms_path", type=str, required=True, help="Path to the DMS CSV file to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all outputs, models, and logs.")
    
    # --- Model Arguments ---
    parser.add_argument("--model_name", type=str, default="evo-1-8k-base", help="Name of the base Evo model to fine-tune.")
    parser.add_argument("--model_size", type=str, default="1b", help="Size of the model, e.g., '1b'. Must match the model being loaded.")
    
    # --- Fine-Tuning Hyperparameters ---
    parser.add_argument("--max_steps", type=int, default=100, help="Number of training steps for each fold.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size per GPU.")
    
    # --- Computational Arguments ---
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPU devices to use for training.")
    parser.add_argument("--batch_size_scoring", type=int, default=128, help="Batch size for scoring the test set.")
    parser.add_argument("--device_scoring", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run scoring on.")

    return parser.parse_args()

# --- Helper Functions ---
def run_command(command: list, error_msg: str):
    """Executes a shell command and exits on failure."""
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"{error_msg}\nError: {e}", file=sys.stderr)
        # print(f"Stdout: {e.stdout}", file=sys.stderr)
        # print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)

def df_to_fasta(df: pd.DataFrame, fasta_path: Path):
    """Converts a DataFrame with a 'sequence' column to a FASTA file."""
    with open(fasta_path, 'w') as f:
        for i, row in df.iterrows():
            # Sequence headers must be unique, simple integers are fine
            f.write(f">{i}\n")
            # Preprocess sequence: uppercase, no whitespace, U->T
            sequence = str(row['sequence']).strip().upper().replace('U', 'T')
            f.write(f"{sequence}\n")

# --- Core Workflow Functions ---

def convert_base_model(model_name: str, model_size: str, nemo_model_dir: Path):
    """Converts a HuggingFace model to NeMo format if it doesn't already exist."""
    if nemo_model_dir.exists() and any(nemo_model_dir.iterdir()):
        print(f"NeMo-converted model already found at {nemo_model_dir}. Skipping conversion.")
        return
        
    print(f"Converting base model {model_name} to NeMo format at {nemo_model_dir}...")
    nemo_model_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "evo2_convert_to_nemo2",
        "--model-path", f"hf://arcinstitute/{model_name}",
        "--model-size", model_size,
        "--output-dir", str(nemo_model_dir)
    ]
    run_command(command, "Failed to convert base model to NeMo format.")

def preprocess_data_for_fold(fold_dir: Path, data_dir: Path, fasta_path: Path):
    """Generates config and runs preprocess_evo2 for a fold's training data."""
    print(f"Preprocessing FASTA file: {fasta_path}")
    
    preprocess_config = {
        'datapaths': [str(fasta_path.resolve())],
        'output_dir': str((data_dir / "preprocessed_data").resolve()),
        'output_prefix': 'fold_train_data',
        'train_split': 0.9,
        'valid_split': 0.1,
        'test_split': 0.0,
        'overwrite': True,
        'indexed_dataset_dtype': "uint8",
        'tokenizer_type': "Byte-Level",
        'workers': os.cpu_count() or 1,
        'append_eod': True
    }
    
    config_path = data_dir / "preprocess_config.yaml"
    with open(config_path, 'w') as f:
        # Use a list format for YAML as in the tutorial
        yaml.dump([preprocess_config], f)

    command = ["preprocess_evo2", "--config", str(config_path)]
    run_command(command, f"Failed to preprocess data for fold {fold_dir.name}.")
    
    # --- Create training data config ---
    preprocessed_pfx = preprocess_config['output_dir'] + '/' + preprocess_config['output_prefix'] + '_byte-level'
    
    training_data_config = [
        {'dataset_prefix': f"{preprocessed_pfx}_train", 'dataset_split': 'train', 'dataset_weight': 1.0},
        {'dataset_prefix': f"{preprocessed_pfx}_val", 'dataset_split': 'validation', 'dataset_weight': 1.0}
    ]
    
    training_config_path = data_dir / "training_data_config.yaml"
    with open(training_config_path, 'w') as f:
        yaml.dump(training_data_config, f)
        
    return training_config_path

def run_finetuning_for_fold(args, fold_dir: Path, data_dir: Path, training_config_path: Path, base_nemo_model_dir: Path):
    """Runs the train_evo2 command for a fold."""
    experiment_dir = fold_dir / "fine_tuned_model"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting fine-tuning for {fold_dir.name}...")

    command = [
        "train_evo2",
        "-d", str(training_config_path.resolve()),
        "--dataset-dir", str(data_dir.resolve()),
        "--experiment-dir", str(experiment_dir.resolve()),
        "--model-size", args.model_size,
        "--devices", str(args.devices),
        "--num-nodes", "1",
        "--seq-length", "1024", # Default from tutorial
        "--micro-batch-size", str(args.micro_batch_size),
        "--lr", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--max-steps", str(args.max_steps),
        "--ckpt-dir", str(base_nemo_model_dir.resolve()),
        "--clip-grad", "1.0",
        "--wd", "0.01",
        "--activation-checkpoint-recompute-num-layers", "1",
        "--val-check-interval", str(args.max_steps // 2), # Validate once halfway
        "--ckpt-async-save",
        "--no-wandb"
    ]
    
    run_command(command, f"Fine-tuning failed for fold {fold_dir.name}.")
    
    # Find the resulting checkpoint file
    try:
        # The checkpoint is saved inside a 'default' subdirectory
        ckpt_dir = experiment_dir / "default"
        # Find the '-last.ckpt' file which is saved at the end of training
        checkpoint_file = next(ckpt_dir.glob("*--last.ckpt"))
    except (StopIteration, FileNotFoundError):
        print(f"Could not find fine-tuned checkpoint in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)
        
    return checkpoint_file

def score_test_set(test_df: pd.DataFrame, tokenizer, fine_tuned_ckpt: Path, args: argparse.Namespace) -> np.ndarray:
    """Loads a fine-tuned NeMo checkpoint and scores sequences."""
    print(f"Loading fine-tuned model from checkpoint: {fine_tuned_ckpt}")
    
    # Instantiate the model architecture and load the fine-tuned weights
    # The trainer configuration is saved in the checkpoint, making this straightforward.
    model = MegatronGPTModel.load_from_checkpoint(checkpoint_path=str(fine_tuned_ckpt))
    model.to(args.device_scoring)
    model.eval()
    
    sequences = [str(seq).strip().upper().replace('U', 'T') for seq in test_df['sequence'].tolist()]
    
    dataset = torch.utils.data.TensorDataset(torch.arange(len(sequences)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_scoring)
    
    all_logprobs = []
    
    pbar = tqdm(total=len(sequences), desc=f"Scoring test set for {fine_tuned_ckpt.parent.parent.parent.name}", unit="seq")
    
    for (indices,) in dataloader:
        batch_seqs = [sequences[i] for i in indices]
        
        input_ids, seq_lengths = prepare_batch(
            batch_seqs,
            tokenizer,
            prepend_bos=True,
            device=args.device_scoring,
        )
        
        with torch.inference_mode():
            logits, *_ = model(input_ids)
            logprobs = logits_to_logprobs(logits, input_ids)
            
        # Sum log-probabilities for each sequence and normalize by length
        # This gives the average log-probability, a common sequence score
        batch_scores = []
        for i in range(len(batch_seqs)):
            # Sum up to the actual length of the sequence, excluding padding
            score = logprobs[i, :seq_lengths[i]-1].sum().item()
            batch_scores.append(score)
            
        all_logprobs.extend(batch_scores)
        pbar.update(len(batch_seqs))
        
    pbar.close()
    
    return np.array(all_logprobs)
    
# --- Main Execution ---
def main():
    args = parse_args()
    
    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dms_id = Path(args.dms_path).stem
    print(f"Starting {args.num_folds}-fold CV for DMS assay: {dms_id}")

    # --- Load Data ---
    try:
        dms_df = pd.read_csv(args.dms_path)
        required_cols = ["sequence", "DMS_score"]
        if not all(col in dms_df.columns for col in required_cols):
            raise ValueError(f"DMS file must contain 'sequence' and 'DMS_score' columns.")
        # Drop rows with missing values in essential columns
        dms_df.dropna(subset=required_cols, inplace=True)
        dms_df.reset_index(drop=True, inplace=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Convert Base Model (once) ---
    base_nemo_model_dir = output_dir / f"{args.model_name}_nemo"
    convert_base_model(args.model_name, args.model_size, base_nemo_model_dir)

    # --- Load Tokenizer (once) ---
    print("Loading tokenizer...")
    evo_model = Evo(args.model_name)
    tokenizer = evo_model.tokenizer

    # --- K-Fold Cross-Validation Loop ---
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    fold_correlations = []

    for i, (train_idx, test_idx) in enumerate(kf.split(dms_df)):
        fold_num = i + 1
        print("\n" + "="*50)
        print(f"Processing Fold {fold_num}/{args.num_folds}")
        print("="*50)
        
        fold_dir = output_dir / dms_id / f"fold_{fold_num}"
        data_dir = fold_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_df, test_df = dms_df.iloc[train_idx], dms_df.iloc[test_idx]
        
        # 1. Prepare data for the current fold
        train_fasta_path = data_dir / "train.fa"
        df_to_fasta(train_df, train_fasta_path)
        
        # 2. Preprocess data using NeMo's script
        training_config_path = preprocess_data_for_fold(fold_dir, data_dir, train_fasta_path)
        
        # 3. Run fine-tuning
        fine_tuned_ckpt = run_finetuning_for_fold(
            args, fold_dir, data_dir, training_config_path, base_nemo_model_dir
        )
        
        # 4. Score the hold-out test set
        predicted_scores = score_test_set(test_df, tokenizer, fine_tuned_ckpt, args)
        
        # 5. Evaluate and store results
        true_scores = test_df['DMS_score'].values
        
        # Ensure no NaNs are passed to spearmanr
        valid_indices = ~np.isnan(predicted_scores) & ~np.isnan(true_scores)
        correlation, p_value = spearmanr(predicted_scores[valid_indices], true_scores[valid_indices])
        fold_correlations.append(correlation)
        
        print(f"\nFold {fold_num} Evaluation:")
        print(f"  - Test Set Size: {len(test_df)}")
        print(f"  - Spearman Correlation: {correlation:.4f} (p-value: {p_value:.2e})")
        
        # Save fold results
        test_df.loc[:, f"{args.model_name}_finetuned_score"] = predicted_scores
        test_df.to_csv(fold_dir / "test_set_predictions.csv", index=False)

    # --- Final Summary ---
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    mean_corr = np.mean(fold_correlations)
    std_corr = np.std(fold_correlations)
    
    print(f"DMS Assay: {dms_id}")
    print(f"Correlations per fold: {[f'{c:.4f}' for c in fold_correlations]}")
    print(f"Mean Spearman Correlation: {mean_corr:.4f}")
    print(f"Std Dev of Correlation: {std_corr:.4f}")

    # Save summary results
    summary_path = output_dir / f"{dms_id}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"DMS Assay: {dms_id}\n")
        f.write(f"Mean Spearman Correlation ({args.num_folds}-fold CV): {mean_corr:.4f}\n")
        f.write(f"Std Dev of Correlation: {std_corr:.4f}\n")
        f.write(f"Individual Fold Correlations: {', '.join([f'{c:.4f}' for c in fold_correlations])}\n")

    print(f"\nProcess complete. All results saved in {output_dir / dms_id}")


if __name__ == "__main__":
    main()
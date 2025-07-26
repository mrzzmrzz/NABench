#!/usr/bin/env python3
"""
Script to run Evo model inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.
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
from transformers import PreTrainedModel
import re
from hyena.standalone_hyenadna import HyenaDNAModel
from hyena.standalone_hyenadna import CharacterTokenizer
from hyena.huggingface import HyenaDNAPreTrainedModel, inject_substring, load_weights
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader

def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess RNA sequence for DNA model:
    - Convert RNA (U) to DNA (T)
    - Convert to uppercase
    - Remove any whitespace
    
    Args:
        sequence: Input RNA or DNA sequence
        
    Returns:
        str: Preprocessed DNA sequence
    """
    return sequence.strip().upper().replace('U', 'T')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo model inference on DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,

        default=0,
        help="Row ID in the reference sheet to process"
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to reference sheet containing DMS_ID column"
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing DMS CSV files"
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available, else cpu)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hyenadna-large-1m-seqlen",
        help="Name of the Evo model to use (default: evo-1-8k-base)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for processing sequences (default: 128)"
    )
    return parser.parse_args()

def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load reference sheet and get DMS_ID for specified row.
    
    Args:
        ref_sheet_path: Path to reference sheet
        row_id: Row ID to process
        
    Returns:
        str: DMS_ID for specified row
    
    Raises:
        ValueError: If row_id is not found or DMS_ID is missing
    """
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if row_id >= len(ref_df):
            raise ValueError(f"Row ID {row_id} exceeds number of rows in reference sheet")
        
        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}")
            
        return str(dms_id)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    except KeyError:
        raise KeyError("Reference sheet must contain 'DMS_ID' column")

def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """
    Load DMS data for specified DMS_ID.
    
    Args:
        dms_dir_path: Directory containing DMS files
        dms_id: DMS ID to process
        
    Returns:
        pd.DataFrame: DataFrame containing sequences to process
        
    Raises:
        FileNotFoundError: If DMS file is not found
    """
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")
        
    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")
        
    return df

class SequenceDataset(Dataset):
    """Dataset for processing sequences in batches."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def run_inference(model, tokenizer, sequences: list, device: str, batch_size: int) -> np.ndarray:
    """
    Run Evo model inference on sequences in batches.
    
    Args:
        model: Loaded Evo model
        tokenizer: Evo tokenizer
        sequences: List of sequences to process
        device: Device to run inference on
        batch_size: Number of sequences to process in each batch
        
    Returns:
        np.ndarray: Log probabilities array
    """
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    all_logprobs = []
    
    # Create progress bar
    total_batches = len(dataloader)
    pbar = tqdm(
        total=len(sequences),
        desc="Scoring sequences",
        unit="seq"
    )
    for i in range(0,len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        tok_seq = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tok_seq = tok_seq["input_ids"]  # grab ids
        # place on device, convert to tensor
        tok_seq_input = torch.LongTensor(tok_seq)  # unsqueeze for batch dim
        tok_seq_input = tok_seq_input.to(device)
        tok_seq = tok_seq.to(device)  # move to device

        model.to(device)
        with torch.no_grad():
            logits = model(tok_seq_input)

            logprobs = F.log_softmax(logits, dim=-1)

            true_logprobs = torch.gather(logprobs, dim=-1, index=tok_seq.unsqueeze(-1)).squeeze(-1)

            mask = (tok_seq != tokenizer.pad_token_id).float()
            true_logprobs = true_logprobs * mask

            scores = true_logprobs.sum(dim=1) / mask.sum(dim=1)
        all_logprobs.append(scores.cpu().numpy())
    return np.concatenate(all_logprobs)
    

def main(args):

    

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_file = output_dir / f"correlation_summary.txt"

    
    
    # Load DMS ID from reference sheet
    dms_id = load_reference_data(args.ref_sheet, args.row_id)
    print(f"Processing DMS ID: {dms_id}")
    
    # Load DMS data
    dms_df = load_dms_data(args.dms_dir_path, dms_id)
    
    # Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [preprocess_sequence(seq) for seq in tqdm(
        dms_df["sequence"].tolist(),
        desc="Preprocessing",
        unit="seq"
    )]
    print(f"Preprocessed {len(sequences)} sequences")
    
    # Initialize model
    
        
    # Run inference in batches
    print(f"Running inference with batch size {args.batch_size}...")
    logprobs = run_inference(
        model, 
        tokenizer, 
        sequences, 
        args.device, 
        args.batch_size
    )
    
    # Calculate sequence scores (mean log probability)
    sequence_scores = logprobs
    
    # Add scores to DataFrame
    score_column = f"{args.model_name.replace('-', '_')}_score"
    dms_df[score_column] = sequence_scores
    
    # Calculate Spearman correlation
    correlation, pvalue = spearmanr(dms_df['DMS_score'], dms_df[score_column])
    
    # Save results
    output_file = output_dir / f"{dms_id}.csv"
    dms_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Spearman correlation with DMS scores: {correlation:.3f} (p-value: {pvalue:.2e})")

    print(f"Output saved to: {output_file}")

    #Save the Spearman correlation
    correlation_file = output_dir / f"correlation_summary.txt"
    if not correlation_file.exists():
        with open(correlation_file, 'w') as f:
            f.write("DMS_ID,Spearman_Correlation\n")
    # Append the correlation result
    with open(correlation_file, 'a') as f:
        f.write(f"{dms_id},{correlation:.3f}\n")



if __name__ == "__main__":
    args = parse_args()
    reference_sheet_path = Path(args.ref_sheet)
    df = pd.read_csv(reference_sheet_path)
    total_rows = len(df)

    print("Initializing Hyena model...")
    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = 1000000  # auto selects
    model_name = args.model_name
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )
    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if model_name in ['hyenadna-tiny-1k-seqlen',
                                'hyenadna-small-32k-seqlen',
                                'hyenadna-medium-160k-seqlen',
                                'hyenadna-medium-450k-seqlen',
                                'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    elif model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    for row_id in range(total_rows):
        args.row_id = row_id
        print(f"Processing row {row_id + 1}/{total_rows}...")
        args.ref_sheet = reference_sheet_path
        args.dms_dir_path = Path(args.dms_dir_path)
        args.output_dir_path = Path(args.output_dir_path)
        args.model_name = "hyenadna-large-1m-seqlen"
        main(args)
# python score_evo_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path evo_output --model_name evo-1-8k-base
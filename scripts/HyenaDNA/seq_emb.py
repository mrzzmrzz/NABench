import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# HyenaDNA-specific imports
from hyena.standalone_hyenadna import CharacterTokenizer
from hyena.huggingface import HyenaDNAPreTrainedModel


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence by converting it to uppercase, stripping whitespace, and U->T.
    """
    return str(sequence).strip().upper().replace("U", "T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HyenaDNA model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_name", type=str, default="hyenadna-large-1m-seqlen",
        help="Name of the HyenaDNA model to use from the checkpoint directory."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory where pretrained model checkpoints are stored."
    )
    parser.add_argument(
        "--row_id", type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed."
    )
    parser.add_argument(
        "--ref_sheet", type=str, required=True,
        help="Path to the reference sheet CSV containing DMS_ID column."
    )
    parser.add_argument(
        "--dms_dir_path", type=str, required=True,
        help="Directory containing the DMS assay CSV files."
    )
    parser.add_argument(
        "--output_dir_path", type=str, required=True,
        help="Directory where the output .npy files will be saved."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda:0' or 'cpu')."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Number of sequences to process in each batch."
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """Load the reference sheet and retrieve the DMS_ID for a specific row."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if not (0 <= row_id < len(ref_df)):
            raise ValueError(f"Row ID {row_id} is out of bounds for the reference sheet.")
        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}.")
        return str(dms_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found at: {ref_sheet_path}")
    except KeyError:
        raise KeyError("The reference sheet must contain a 'DMS_ID' column.")


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """Load DMS data for a specific DMS_ID."""
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")
    df = pd.read_csv(dms_file)
    df.columns = [col.lower() for col in df.columns]
    return df


@dataclass
class InferenceConfig:
    """Configuration for the feature extraction process."""
    model_name: str
    checkpoint_dir: str
    device: str = "cuda:0"
    batch_size: int = 32


def extract_features(
    model: HyenaDNAPreTrainedModel,
    tokenizer: CharacterTokenizer,
    sequences: list[str],
    config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the last hidden state of the HyenaDNA model.

    Args:
        model: The pre-trained HyenaDNA model.
        tokenizer: The corresponding character tokenizer.
        sequences: A list of preprocessed DNA sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of mean-pooled sequence embeddings.
    """
    model.to(config.device)
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), config.batch_size), desc="Extracting Features", unit="batch"):
            batch_sequences = sequences[i : i + config.batch_size]
            if not batch_sequences:
                continue
            
            while True:
                try:
                    tokens = tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    
                    input_ids = tokens['input_ids'].to(config.device)
                    
                    outputs = model(input_ids)
                    cls_embeddings = outputs[:, 0, :]  # CLS token embedding
                    pool_embeddings = torch.mean(outputs, dim=1)  # Mean pooling
                    seq_emb = torch.cat([cls_embeddings, pool_embeddings], dim=1)  # Concatenate

                    all_embeddings.append(seq_emb.cpu().numpy())
                    break  # Success, exit while loop

                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA Out of Memory on batch starting at index {i}. Retrying...")
                    torch.cuda.empty_cache()
                    continue # Retry while loop

    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)


def main(model: HyenaDNAPreTrainedModel, tokenizer: CharacterTokenizer, config: InferenceConfig, args: argparse.Namespace, row_id: int):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dms_id = load_reference_data(args.ref_sheet, row_id)
    print(f"Processing DMS ID: {dms_id}")

    # Your workflow improvement: skip if the output file already exists
    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return

    dms_df = load_dms_data(args.dms_dir_path, dms_id)
    
    sequence_col = next((col for col in dms_df.columns if "sequence" in col), None)
    if not sequence_col:
        print(f"Error: No sequence column found in {dms_id}.csv. Skipping.")
        return
    sequences = [preprocess_sequence(seq) for seq in dms_df[sequence_col]]
    
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)}).")
        return

    score_col = next((col for col in dms_df.columns if "dms_score" in col), None)
    if not score_col:
        print(f"Error: No DMS score column found in {dms_id}.csv. Skipping.")
        return
    true_labels = dms_df[score_col].values
    
    print(f"Embeddings shape: {embeddings.shape}, Labels shape: {true_labels.shape}")

    valid_mask = np.isfinite(embeddings).all(axis=1) & ~np.isnan(true_labels)
    filtered_embeddings = embeddings[valid_mask]
    filtered_labels = true_labels[valid_mask]

    if filtered_embeddings.shape[0] == 0:
        print("No valid data remaining after filtering. Skipping save.")
        return

    result_array = np.concatenate([filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1)
    
    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    config = InferenceConfig(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size
    )

    # --- Load HyenaDNA Model and Tokenizer Once ---
    print(f"Initializing HyenaDNA model: {config.model_name}...")
    try:
        # Create tokenizer
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=1_000_002,  # A high value to accommodate large models
            add_special_tokens=False,
            padding_side='left',
        )

        # Instantiate the model from pretrained
        model = HyenaDNAPreTrainedModel.from_pretrained(
            config.checkpoint_dir,
            config.model_name,
            download=True,
            device=config.device,
            use_head=False, # We want the backbone embeddings, not classification logits
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Determine which rows to process
    if args.row_id is not None:
        rows_to_process = [args.row_id]
    else:
        ref_df = pd.read_csv(args.ref_sheet)
        rows_to_process = range(len(ref_df))

    # Loop through and process each dataset
    for i, row_id in enumerate(rows_to_process):
        print("-" * 50)
        print(f"Processing row {row_id} ({i + 1}/{len(rows_to_process)})...")
        main(model, tokenizer, config, args, row_id)
    
    print("-" * 50)
    print("All tasks completed.")
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence by converting to uppercase and stripping whitespace.
    It also handles RNA-to-DNA conversion.
    """
    return str(sequence).strip().upper().replace("U", "T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GENERator model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERator-eukaryote-3b-base",
        help="Hugging Face ID of the GENERator model to use.",
    )
    parser.add_argument(
        "--row_id",
        type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed.",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to the reference sheet CSV containing DMS_ID column.",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing the DMS assay CSV files.",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory where the output .npy files will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sequences to process in each batch.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """Load the reference sheet and retrieve the DMS_ID for a specific row."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if not (0 <= row_id < len(ref_df)):
            raise ValueError(
                f"Row ID {row_id} is out of bounds for the reference sheet."
            )
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
    device: str = "cuda:0"
    batch_size: int = 8


def extract_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences: list[str],
    config: InferenceConfig,
) -> np.ndarray:
    """
    Extracts embeddings from the last token's hidden state of a Causal LM.

    Args:
        model: The pre-trained Causal LM (e.g., GENERator).
        tokenizer: The corresponding tokenizer.
        sequences: A list of preprocessed sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    model.to(config.device)
    model.eval()
    all_embeddings = []
    max_length = getattr(model.config, "max_position_embeddings", 512)

    with torch.inference_mode():
        for i in tqdm(
            range(0, len(sequences), config.batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            batch_sequences = sequences[i : i + config.batch_size]
            if not batch_sequences:
                continue

            while True:
                try:
                    inputs = tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(config.device)

                    outputs = model(**inputs, output_hidden_states=True)

                    # For Causal LMs, the embedding of the last token is a good representation of the sequence
                    last_hidden_state = outputs.hidden_states[-1]
                    attention_mask = inputs["attention_mask"]
                    last_token_indices = attention_mask.sum(dim=1) - 1

                    sequence_embeddings = last_hidden_state[
                        torch.arange(last_hidden_state.size(0)), last_token_indices, :
                    ]

                    all_embeddings.append(sequence_embeddings.cpu().numpy())
                    break  # Exit while loop on success

                except torch.cuda.OutOfMemoryError:
                    print(
                        f"CUDA Out of Memory on batch starting at index {i}. Skipping batch."
                    )
                    torch.cuda.empty_cache()
                    # Break here to avoid infinite loops on persistent OOM errors
                    break

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: InferenceConfig,
    args: argparse.Namespace,
    row_id: int,
):
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

    # 1. Load data
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    sequence_col = next((col for col in dms_df.columns if "sequence" in col), None)
    if not sequence_col:
        print(f"Error: No sequence column found in {dms_id}.csv. Skipping.")
        return
    sequences = [preprocess_sequence(seq) for seq in dms_df[sequence_col]]

    # 2. Extract features
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) doesn't match sequences ({len(sequences)})."
        )
        return

    # 3. Prepare labels and clean data
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

    # 4. Combine labels and embeddings, then save
    result_array = np.concatenate(
        [filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1
    )

    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    config = InferenceConfig(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size
    )

    # Load the GENERator model and tokenizer once
    print(f"Loading model and tokenizer from: {config.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Important for Causal LMs

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, trust_remote_code=True
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

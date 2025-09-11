import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# Evo-specific imports
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess an RNA sequence for a DNA-based model.
    - Converts RNA 'U' to DNA 'T'.
    - Converts the sequence to uppercase.
    - Removes leading/trailing whitespace.

    Args:
        sequence (str): The input RNA or DNA sequence.

    Returns:
        str: The preprocessed DNA sequence.
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo model inference to extract features from DMS assay sequences."
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
        help="Path to the reference sheet containing the 'DMS_ID' column.",
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
        "--model_name",
        type=str,
        default="evo-1-8k-base",
        help="Name of the Evo model to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DMS file {dms_file} is missing required columns: {missing_cols}"
        )
    return df


class SequenceDataset(Dataset):
    """Simple dataset to wrap a list of sequences for DataLoader."""

    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


@dataclass
class InferenceConfig:
    """Configuration for the feature extraction process."""

    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    batch_size: int = 32


def extract_features(
    model: torch.nn.Module, tokenizer, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the last hidden state of the Evo model.

    Args:
        model: The pre-trained Evo model.
        tokenizer: The corresponding tokenizer.
        sequences: A list of preprocessed sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    while True:
        try:
            model.to(config.device)
            model.eval()
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory while loading model. Retrying...")
            torch.cuda.empty_cache()
            continue
        break
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    all_embeddings = []

    with torch.inference_mode():
        for batch_sequences in tqdm(
            dataloader, desc="Extracting Features", unit="batch"
        ):
            input_ids, _ = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device=config.device,
            )

            try:
                # The Evo model forward pass returns (logits, hidden_states)
                while True:
                    try:
                        logits, _ = model(input_ids)
                    except torch.cuda.OutOfMemoryError:
                        print("CUDA Out of Memory during model inference. Retrying...")
                        torch.cuda.empty_cache()
                        continue
                    break

                sequence_embeddings = logits_to_logprobs(
                    logits, input_ids, trim_bos=True
                )

                all_embeddings.append(
                    sequence_embeddings.to(dtype=torch.float32).cpu().numpy()
                )

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA Out of Memory on a batch. Skipping batch.")
                torch.cuda.empty_cache()
                continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(evo_model, config: InferenceConfig, args: argparse.Namespace, row_id: int):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    dms_id = load_reference_data(args.ref_sheet, row_id)
    print(f"Processing DMS ID: {dms_id}")
    dms_df = load_dms_data(args.dms_dir_path, dms_id)
    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return
    # 2. Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"].tolist()]

    # 3. Extract features
    model, tokenizer = evo_model.model, evo_model.tokenizer
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)})."
        )
        return

    # 4. Prepare labels and clean data
    true_labels = dms_df["DMS_score"].values
    print(f"Embeddings shape: {embeddings.shape}, Labels shape: {true_labels.shape}")

    valid_mask = np.isfinite(embeddings).all(axis=1) & ~np.isnan(true_labels)
    filtered_embeddings = embeddings[valid_mask]
    filtered_labels = true_labels[valid_mask]

    if filtered_embeddings.shape[0] == 0:
        print("No valid data remaining after filtering. Skipping save.")
        return

    # 5. Combine labels and embeddings, then save
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

    # Load the Evo model once
    print(f"Loading Evo model: {config.model_name}...")
    try:
        evo_model = Evo(config.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
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
        main(evo_model, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

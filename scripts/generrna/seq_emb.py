import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

# Add the GenerRNA directory to the Python path
sys.path.append(str(Path(__file__).parent.resolve()))
from GenerRNA.model import GPT, GPTConfig
from transformers import AutoTokenizer


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses an RNA sequence: converts to uppercase and strips whitespace.
    """
    return str(sequence).strip().upper()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GenerRNA model inference to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to the GenerRNA model checkpoint file (e.g., 'ckpt.pt').",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="GenerRNA/tokenizer",
        help="Path to the directory containing the GenerRNA tokenizer.",
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
        help="Path to the reference sheet CSV listing all assays.",
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
        "--batch_size",
        type=int,
        default=32,
        help="Number of sequences to process in each batch.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> pd.Series:
    """Load the reference sheet and retrieve the full row for a specific ID."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        ref_df.rename(columns={ref_df.columns[0]: "DMS_ID"}, inplace=True)
        if not (0 <= row_id < len(ref_df)):
            raise ValueError(
                f"Row ID {row_id} is out of bounds for the reference sheet."
            )
        return ref_df.iloc[row_id]
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found at: {ref_sheet_path}")


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """Load DMS mutation data for a specific DMS_ID."""
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")
    df = pd.read_csv(dms_file)
    df.columns = [col.lower() for col in df.columns]
    return df


@dataclass
class InferenceConfig:
    """Configuration for the feature extraction process."""

    model_ckpt: str
    tokenizer_dir: str
    device: str = "cuda:0"
    batch_size: int = 32


def extract_features(
    model: torch.nn.Module, tokenizer, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the logits of the GenerRNA model.

    Args:
        model: The pre-trained GenerRNA GPT model.
        tokenizer: The corresponding tokenizer.
        sequences: A list of preprocessed sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    model.to(config.device)
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(sequences), config.batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            batch_sequences = sequences[i : i + config.batch_size]
            if not batch_sequences:
                continue

            try:
                # Prepare batches for autoregressive model (inputs are seq[:-1])
                inputs = [
                    torch.tensor(tokenizer.encode(seq)[:-1]) for seq in batch_sequences
                ]

                # Pad sequences to the max length in the batch
                padded_inputs = pad_sequence(
                    inputs, batch_first=True, padding_value=tokenizer.pad_token_id or 0
                )
                padded_inputs = padded_inputs.to(config.device)

                # The model's forward pass returns (logits, loss). We only need logits.
                logits, _ = model(padded_inputs, targets=None)  

                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
                    pad_id = tokenizer.pad_token_id

                # [B, S] -> [B, S, 1]
                attention_mask = (padded_inputs != pad_id).unsqueeze(-1)
                attention_mask = attention_mask.to(
                    dtype=logits.dtype, device=logits.device
                )

                sum_logits = (logits * attention_mask).sum(dim=1)  # [B, H]
                sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                pooled_embeddings = sum_logits / sum_mask  # [B, H]

                all_embeddings.append(pooled_embeddings.detach().cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                print(
                    f"CUDA Out of Memory on batch starting at index {i}. Skipping batch."
                )
                torch.cuda.empty_cache()
                continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(
    model, tokenizer, config: InferenceConfig, args: argparse.Namespace, row_id: int
):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
    print(f"Processing DMS ID: {dms_id}")
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    # Check for sequence column; some files might have it pre-generated
    sequence_col = next((col for col in dms_df.columns if "sequence" in col), None)
    if not sequence_col:
        print(f"Error: No sequence column found in {dms_id}.csv. Skipping.")
        return

    sequences = [preprocess_sequence(seq) for seq in dms_df[sequence_col]]

    # 2. Extract features
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)})."
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

    output_file = output_dir / f"{dms_id}.npy"
    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = InferenceConfig(
        model_ckpt=args.model_ckpt,
        tokenizer_dir=args.tokenizer_dir,
        device=device,
        batch_size=args.batch_size,
    )

    # Load the GenerRNA model and tokenizer once
    print(f"Loading GenerRNA model from '{config.model_ckpt}'...")
    checkpoint = torch.load(config.model_ckpt, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Loading tokenizer from '{config.tokenizer_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_dir)

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

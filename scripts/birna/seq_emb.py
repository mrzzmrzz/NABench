import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertConfig,
)
import transformers


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
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run BiRNA-BERT model inference to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed.",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        default="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv",
        help="Path to the reference sheet containing the 'DMS_ID' column.",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        default="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays",
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
        default=32,
        help="Number of sequences to process in each batch.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load the reference sheet and retrieve the DMS_ID for a specific row.

    Args:
        ref_sheet_path (str): Path to the reference sheet CSV file.
        row_id (int): The row index to process.

    Returns:
        str: The DMS_ID for the specified row.
    """
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
    """
    Load DMS data for a specific DMS_ID.

    Args:
        dms_dir_path (str): Directory containing DMS assay files.
        dms_id (str): The DMS ID to load.

    Returns:
        pd.DataFrame: A DataFrame with the DMS assay data.
    """
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


@dataclass
class InferenceConfig:
    """Configuration for the feature extraction process."""

    model_name: str = "buetnlpbio/birna-bert"
    tokenizer_name: str = "buetnlpbio/birna-tokenizer"
    device: str = "cuda:0"
    ref_sheet: str = "/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
    dms_dir_path: str = "/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"
    output_dir_path: str = "./results"
    batch_size: int = 32


def extract_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences,
    config: InferenceConfig,
) -> np.ndarray:
    """
    Extracts embeddings for a list of sequences using the provided model.

    Args:
        model: The pre-trained transformers model.
        tokenizer: The corresponding tokenizer.
        sequences: A list of preprocessed sequences.
        config: The configuration object.

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
                tokens = tokenizer(
                    batch_sequences, return_tensors="pt", padding=True, truncation=True
                ).to(config.device)

                # Get model outputs (logits are the pre-softmax scores from the final layer)
                outputs = model(**tokens).logits
                pool_embedding = outputs.mean(dim=1)
                cls_embedding = outputs[:, 0, :]
                sequence_embeddings = torch.cat((pool_embedding, cls_embedding), dim=1)

                all_embeddings.append(sequence_embeddings.cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                print(
                    f"CUDA Out of Memory on batch starting at index {i}. Reducing batch size for this attempt is not yet implemented. Skipping batch."
                )
                torch.cuda.empty_cache()
                continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: InferenceConfig,
    row_id: int,
):
    """
    Main processing pipeline for a single DMS dataset.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        config: The configuration object.
        row_id: The row ID from the reference sheet to process.
    """
    output_dir = Path(config.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data identifiers and sequences
    dms_id = load_reference_data(config.ref_sheet, row_id)
    print(f"Processing DMS ID: {dms_id}")
    dms_df = load_dms_data(config.dms_dir_path, dms_id)

    # 2. Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"].tolist()]

    # 3. Run inference to get embeddings
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)})."
        )
        return

    # 4. Prepare labels and clean data
    true_labels = dms_df["DMS_score"].values
    print(
        f"Initial embeddings shape: {embeddings.shape}, Labels shape: {true_labels.shape}"
    )

    # Create a mask for valid data points (finite embeddings and non-NaN labels)
    valid_mask = np.isfinite(embeddings).all(axis=1) & ~np.isnan(true_labels)

    filtered_embeddings = embeddings[valid_mask]
    filtered_labels = true_labels[valid_mask]

    if filtered_embeddings.shape[0] == 0:
        print("No valid data remaining after filtering NaNs and Infs. Skipping save.")
        return

    # 5. Combine labels and embeddings and save to .npy file
    result_array = np.concatenate(
        [filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1
    )

    output_file = output_dir / f"{dms_id}.npy"
    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    # Set up configuration from arguments
    config = InferenceConfig(
        device=args.device,
        ref_sheet=args.ref_sheet,
        dms_dir_path=args.dms_dir_path,
        output_dir_path=args.output_dir_path,
        batch_size=args.batch_size,
    )

    # Load the model and tokenizer once
    print(f"Loading tokenizer: {config.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    print(f"Loading model: {config.model_name}...")
    model_config = transformers.BertConfig.from_pretrained(config.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_name, config=model_config, trust_remote_code=True
    )
    # Replace the classification head with an identity layer to get embeddings directly
    model.cls = torch.nn.Identity()

    # Determine which rows to process
    if args.row_id is not None:
        rows_to_process = [args.row_id]
        total_rows = 1
    else:
        ref_df = pd.read_csv(config.ref_sheet)
        rows_to_process = range(len(ref_df))
        total_rows = len(ref_df)

    # Loop through and process each specified dataset
    for i, row_id in enumerate(rows_to_process):
        print("-" * 50)
        print(f"Processing row {row_id} ({i + 1}/{total_rows})...")
        main(model, tokenizer, config, row_id)

    print("-" * 50)
    print("All tasks completed.")

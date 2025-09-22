import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# Assuming 'modelgenerator' is a locally available library
# You might need to adjust the path if it's not in the standard python path
try:
    from modelgenerator.tasks import Embed
except ImportError:
    print(
        "Warning: 'modelgenerator' library not found. Please ensure it is installed and accessible."
    )
    sys.exit(1)


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess RNA sequence for a DNA-based model.
    - Converts RNA 'U' to DNA 'T'.
    - Converts sequence to uppercase.
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
        description="Run Aido model inference to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed.",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        default="",
        help="Path to the reference sheet containing the 'DMS_ID' column.",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        default="",
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
        default="aido_rna_1b600m",
        help="Name of the Aido model to use.",
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

    Raises:
        FileNotFoundError: If the reference sheet is not found.
        KeyError: If the 'DMS_ID' column is missing.
        ValueError: If the row_id is out of bounds or DMS_ID is missing.
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

    Raises:
        FileNotFoundError: If the DMS file for the given ID does not exist.
        ValueError: If required columns are missing from the DMS file.
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

    model_name: str = "aido_rna_1b600m"
    device: str = "cuda:0"
    ref_sheet: str = ""
    dms_dir_path: str = ""
    output_dir_path: str = ""
    batch_size: int = 32


def extract_features(
    model: torch.nn.Module, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings for a list of sequences using the provided model.

    Args:
        model (torch.nn.Module): The pre-trained model for inference.
        sequences (list[str]): A list of preprocessed sequences.
        config (InferenceConfig): The configuration object.

    Returns:
        np.ndarray: A 2D numpy array of sequence embeddings.
    """
    while True:
        try:
            model.to(config.device)
            model.eval()
            break
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory error occurred.")

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(sequences), config.batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            batch_sequences = sequences[i : i + config.batch_size]

            try:
                # The modelgenerator API seems to expect a dictionary
                transformed_batch = model.transform({"sequences": batch_sequences})

                # The model outputs embeddings for each token, shape: (batch, seq_len, embed_dim)
                token_embeddings = model(transformed_batch)
                # CLS embedding is typically the first token
                cls_embeddings = token_embeddings[:, 0, :]
                # Mean pool over the sequence length dimension to get a single vector per sequence
                pool_embeddings = token_embeddings.mean(dim=1)

                sequence_embeddings = torch.cat(
                    [cls_embeddings, pool_embeddings], dim=1
                )
                all_embeddings.append(sequence_embeddings.cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                print(
                    f"CUDA Out of Memory on batch starting at index {i}. Skipping batch."
                )
                torch.cuda.empty_cache()
                continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(model: torch.nn.Module, config: InferenceConfig, row_id: int):
    """
    Main processing pipeline for a single DMS dataset.

    Args:
        model (torch.nn.Module): The loaded model.
        config (InferenceConfig): The configuration object.
        row_id (int): The row ID from the reference sheet to process.
    """
    output_dir = Path(config.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data identifiers and sequences
    dms_id = load_reference_data(config.ref_sheet, row_id)
    print(f"Processing DMS ID: {dms_id}")
    dms_df = load_dms_data(config.dms_dir_path, dms_id)
    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return
    # 2. Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"].tolist()]

    # 3. Run inference to get embeddings
    embeddings = extract_features(model, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)}). This might be due to OOM errors."
        )
        # Re-align based on successful inference if necessary (not implemented here for simplicity)
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
    # Reshape labels to (n, 1) to concatenate with embeddings (n, embed_dim)
    result_array = np.concatenate(
        [filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1
    )

    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    # Load the model once
    print(f"Loading model: {args.model_name}...")
    try:
        model = Embed.from_config({"model.backbone": args.model_name})
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        sys.exit(1)

    # Set up configuration from arguments
    config = InferenceConfig(
        model_name=args.model_name,
        device=args.device,
        ref_sheet=args.ref_sheet,
        dms_dir_path=args.dms_dir_path,
        output_dir_path=args.output_dir_path,
    )

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
        main(model, config, row_id)

    print("-" * 50)
    print("All tasks completed.")

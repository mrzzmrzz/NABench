import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# Add the LucaOne source directory to the Python path
# You may need to adjust this path depending on your project structure
sys.path.append(
    ""
)
from get_embedding import predict_embedding


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence by converting it to uppercase and stripping whitespace.
    """
    return str(sequence).strip().upper()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LucaOne model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--llm_dir",
        type=str,
        required=True,
        help="Base directory of the LucaOne model.",
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
        "--seq_type",
        type=str,
        choices=["gene", "prot"],
        default="gene",
        help="Sequence type for the model.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["matrix", "vector"],
        default="vector",
        help="Type of embedding to extract from the model.",
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
        df = pd.read_csv(ref_sheet_path)
        # Standardize the first column to 'DMS_ID' for consistency
        df.rename(columns={df.columns[0]: "DMS_ID"}, inplace=True)
        if not (0 <= row_id < len(df)):
            raise ValueError(
                f"Row ID {row_id} is out of bounds for the reference sheet."
            )
        return df.iloc[row_id]
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found at: {ref_sheet_path}")


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

    llm_dir: str
    seq_type: str = "gene"
    embedding_type: str = "vector"
    device: str = "cuda:0"
    batch_size: int = 32
    truncation_seq_length: int = 4094


def extract_features(sequences: list[str], config: InferenceConfig) -> np.ndarray:
    """
    Extracts embeddings for a list of sequences using the LucaOne model in batches.

    Args:
        sequences: A list of preprocessed sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    all_embeddings = []

    for i in tqdm(
        range(0, len(sequences)),
        desc="Extracting Features",
        unit="batch",
    ):
        batch_sequences = sequences[i]
        if not batch_sequences:
            continue

        # Prepare batch input for the predict_embedding function
        batch_input = [str(i), config.seq_type, batch_sequences] 
        try:
            # Call the embedding function once per batch
            emb, _ = predict_embedding(
                config.llm_dir,
                batch_input,
                config.truncation_seq_length,
                config.embedding_type,
                truncation_seq_length=config.truncation_seq_length,
                repr_layers=[-1],
                device=config.device,
                matrix_add_special_token=False,
            )

            all_embeddings.append(emb)

        except Exception as e:
            print(
                f"An error occurred during embedding batch starting at index {i}: {e}. Skipping batch."
            )
            raise e
            # Add placeholder NaNs for failed batch to maintain array shape alignment
            num_failed = len(batch_sequences)
            # Assuming we know the embedding dimension, e.g., from a successful run or config.
            # If not, this part might need adjustment. Let's assume a placeholder size.
            placeholder_dim = 1280  # Example dimension
            all_embeddings.extend([np.full(placeholder_dim, np.nan)] * num_failed)
            continue

    if not all_embeddings:
        return np.array([])

    return np.vstack(all_embeddings)


def main(config: InferenceConfig, args: argparse.Namespace, row_id: int):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
    print(f"Processing DMS ID: {dms_id}")
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    sequence_col = next((col for col in dms_df.columns if "sequence" in col), None)
    if not sequence_col:
        print(f"Error: No sequence column found in {dms_id}.csv. Skipping.")
        return

    sequences = [preprocess_sequence(seq) for seq in dms_df[sequence_col]]

    # 2. Extract features
    embeddings = extract_features(sequences, config)
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
        llm_dir=args.llm_dir,
        seq_type=args.seq_type,
        embedding_type=args.embedding_type,
        device=device,
        batch_size=args.batch_size,
    )

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
        main(config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

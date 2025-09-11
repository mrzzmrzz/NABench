import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
import paddle
from tqdm.auto import tqdm

# PaddleNLP imports
from paddlenlp.transformers import ErnieForMaskedLM

# Local module import for RNA-ERNEI's data converter
from src.rna_ernie import BatchConverter


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence by converting it to uppercase and U->T.
    """
    return str(sequence).strip().upper().replace("U", "T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RNA-ERNEI model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the RNA-ERNEI model checkpoint directory.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary file for the tokenizer.",
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

    model_checkpoint: str
    vocab_path: str
    batch_size: int = 256
    max_seq_len: int = 512


def extract_features(
    model: ErnieForMaskedLM,
    batch_converter: BatchConverter,
    sequences: list[str],
    config: InferenceConfig,
) -> np.ndarray:
    """
    Extracts embeddings from the logits of the RNA-ERNEI model.

    Args:
        model: The pre-trained RNA-ERNEI model (PaddlePaddle).
        batch_converter: The data converter to tokenize and batch sequences.
        sequences: A list of preprocessed DNA sequences.
        config: The configuration object.

    Returns:
        A 2D numpy array of mean-pooled sequence embeddings.
    """
    model.eval()
    all_embeddings = []

    # The batch converter expects data in a specific tuple format
    data_to_process = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

    pbar = tqdm(total=len(data_to_process), desc="Extracting Features")
    for _, _, input_ids in batch_converter(data_to_process):
        try:
            with paddle.no_grad():
                logits_paddle = model(input_ids).detach()

            # Convert PaddlePaddle tensor to PyTorch tensor for consistent processing
            logits_torch = torch.from_numpy(logits_paddle.numpy())

            # Create an attention mask to ignore padding (assuming pad token ID is 0)
            # The batch_converter pads with 0
            cls_embedding = logits_torch[:, 0, :]
            pool_embedding = logits_torch[:, 1:, :].mean(dim=1)
            all_embedding = torch.cat([cls_embedding, pool_embedding], dim=1)
            all_embeddings.append(all_embedding.cpu().numpy())
            pbar.update(input_ids.shape[0])

        except Exception as e:
            print(f"An error occurred during a batch: {e}. Skipping batch.")
            pbar.update(input_ids.shape[0])
            continue
    pbar.close()

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(
    model: ErnieForMaskedLM,
    batch_converter: BatchConverter,
    config: InferenceConfig,
    args: argparse.Namespace,
    row_id: int,
):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
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

    # 2. Extract features
    embeddings = extract_features(model, batch_converter, sequences, config)
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

    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    config = InferenceConfig(
        model_checkpoint=args.model_checkpoint,
        vocab_path=args.vocab_path,
    )

    # Load the RNA-ERNEI model and batch converter once
    print(f"Loading RNA-ERNEI model from: {config.model_checkpoint}...")
    try:
        language_model = ErnieForMaskedLM.from_pretrained(config.model_checkpoint)
        batch_converter = BatchConverter(
            k_mer=1,
            vocab_path=config.vocab_path,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
        )
    except Exception as e:
        print(f"Error loading model or converter: {e}")
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
        main(language_model, batch_converter, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

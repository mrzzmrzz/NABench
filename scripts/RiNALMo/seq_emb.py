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

# Assuming rinalmo is installed and accessible
from rinalmo.pretrained import get_pretrained_model


def get_sequences(wt_sequence: str, df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    Generates mutated sequences based on a wild-type sequence and a DataFrame of mutations.
    Also returns the filtered DataFrame corresponding to the generated sequences.
    """
    # RNALMo works with 'U', so we convert T to U
    wt_sequence = wt_sequence.strip().upper().replace("T", "U")

    def apply_mutation(sequence, mutation_str):
        base_offset = 1
        pos = int(mutation_str[1:-1]) - base_offset
        original_base = mutation_str[0].replace("T", "U")
        new_base = mutation_str[-1].replace("T", "U")

        if original_base == "N":
            return sequence[: pos + 1] + new_base + sequence[pos + 1 :]
        elif new_base == "":
            return sequence[:pos] + sequence[pos + 1 :]
        else:
            if not (0 <= pos < len(sequence) and sequence[pos] == original_base):
                raise AssertionError(
                    f"Mutation '{mutation_str}' is inconsistent with sequence at position {pos + 1}."
                )
            return sequence[:pos] + new_base + sequence[pos + 1 :]

    def apply_mutations(sequence, mutations_cell):
        if pd.isna(mutations_cell):
            return sequence
        for mutation_str in str(mutations_cell).split(","):
            sequence = apply_mutation(sequence, mutation_str.strip())
        return sequence

    mutation_column = next(
        (
            col
            for col in df.columns
            if col.lower() in ["mutant", "mutation", "mutations"]
        ),
        None,
    )
    if not mutation_column:
        raise ValueError(
            "No 'mutant', 'mutation', or 'mutations' column found in the DataFrame."
        )

    df_filtered = df.dropna(subset=[mutation_column]).copy()
    df_filtered["mutated_sequence"] = df_filtered[mutation_column].apply(
        lambda x: apply_mutations(wt_sequence, x)
    )
    return df_filtered["mutated_sequence"].tolist(), df_filtered


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RNALMo model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="giga-v1",
        help="Name of the RNALMo pretrained model to use.",
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
        help="Path to the reference sheet CSV containing DMS_ID and wild-type sequence columns.",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing the DMS assay CSV files with mutation data.",
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


def load_reference_data(ref_sheet_path: str, row_id: int) -> pd.Series:
    """Load the reference sheet and retrieve the full row for a specific ID."""
    try:
        ref_df = pd.read_csv(ref_sheet_path, encoding="latin-1")
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

    model_name: str = "giga-v1"
    device: str = "cuda:0"
    batch_size: int = 32


def extract_features(
    model, alphabet, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts high-dimensional embeddings from the RNALMo model's 'representation' output.

    Args:
        model: The pre-trained RNALMo model.
        alphabet: The model's alphabet/tokenizer.
        sequences: A list of RNA sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of mean-pooled sequence embeddings.
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
                # Tokenize and pad the batch
                tokens = torch.tensor(alphabet.batch_tokenize(batch_sequences)).to(
                    config.device
                )

                with torch.cuda.amp.autocast():
                    outputs = model(tokens)

                # Extract the representation tensor
                representations = outputs["representation"]

                cls_embedding = representations[:, 0, :]
                pool_embedding = representations[:, 1 : len(sequences[0]), :].mean(
                    dim=1
                )
                all_embedding = torch.cat([cls_embedding, pool_embedding], dim=1)
                all_embeddings.append(all_embedding.cpu().numpy())

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
    model, alphabet, config: InferenceConfig, args: argparse.Namespace, row_id: int
):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata and mutation data
    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
    print(f"Processing DMS ID: {dms_id}")

    # Your workflow improvement: skip if the output file already exists
    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return

    wt_seq_col = next(
        (
            col
            for col in assay_data.index
            if "raw" in col.lower() and "seq" in col.lower()
        ),
        None,
    )
    if not wt_seq_col:
        print(f"Could not find wild-type sequence column for {dms_id}. Skipping.")
        return
    wt_seq = assay_data[wt_seq_col]

    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    # 2. Generate all sequences from mutations
    print("Generating sequences from mutations...")
    try:
        sequences, dms_df_filtered = get_sequences(wt_seq, dms_df)
    except Exception as e:
        print(f"Could not generate sequences for {dms_id}. Error: {e}")
        return

    # 3. Extract features
    embeddings = extract_features(model, alphabet, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)})."
        )
        return

    # 4. Prepare labels and clean data
    score_col = next(
        (col for col in dms_df_filtered.columns if "dms_score" in col), "dms_score"
    )
    true_labels = dms_df_filtered[score_col].values

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

    # Load the RNALMo model and alphabet once
    print(f"Loading RNALMo model: {config.model_name}...")
    try:
        model, alphabet = get_pretrained_model(model_name=config.model_name)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Determine which rows to process
    if args.row_id is not None:
        rows_to_process = [args.row_id]
    else:
        ref_df = pd.read_csv(args.ref_sheet, encoding="latin-1")
        rows_to_process = range(len(ref_df))

    # Loop through and process each dataset
    for i, row_id in enumerate(rows_to_process):
        print("-" * 50)
        print(f"Processing row {row_id} ({i + 1}/{len(rows_to_process)})...")
        main(model, alphabet, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

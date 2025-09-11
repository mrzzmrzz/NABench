import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def get_sequences(wt_sequence: str, df: pd.DataFrame) -> list[str]:
    """
    Generates mutated sequences based on a wild-type sequence and a DataFrame of mutations.

    Args:
        wt_sequence (str): The wild-type DNA/RNA sequence.
        df (pd.DataFrame): DataFrame containing a column with mutation information.

    Returns:
        list[str]: A list of all generated sequences.
    """
    wt_sequence = wt_sequence.strip().upper().replace("U", "T")

    def apply_mutation(sequence, mutation_str):
        base_offset = 1  # Mutation strings are typically 1-based
        pos = int(mutation_str[1:-1]) - base_offset
        original_base = mutation_str[0]
        new_base = mutation_str[-1]

        if original_base == "N":  # Insertion
            return sequence[: pos + 1] + new_base + sequence[pos + 1 :]
        elif new_base == "":  # Deletion
            return sequence[:pos] + sequence[pos + 1 :]
        else:  # Substitution
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

    df_filtered = df.dropna(subset=[mutation_column])
    mutated_sequences = df_filtered[mutation_column].apply(
        lambda x: apply_mutations(wt_sequence, x)
    )
    return mutated_sequences.tolist(), df_filtered.index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a generic Hugging Face model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_location",
        type=str,
        required=True,
        help="Path or Hugging Face ID of the model to use (e.g., 'InstaDeepAI/nucleotide-transformer-500m-human-ref').",
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
        help="Path to the reference sheet containing DMS_ID and wild-type sequence columns.",
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
        default=16,
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
    # Standardize column names for consistency
    df.columns = [col.lower() for col in df.columns]
    return df


@dataclass
class InferenceConfig:
    """Configuration for the feature extraction process."""

    model_location: str
    device: str = "cuda:0"
    batch_size: int = 16


def extract_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences: list[str],
    config: InferenceConfig,
) -> np.ndarray:
    """
    Extracts embeddings from the last hidden state of a transformers model.

    Args:
        model: The pre-trained transformers model.
        tokenizer: The corresponding tokenizer.
        sequences: A list of DNA sequences.
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
            while True:
                try:
                    tokens = tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    ).to(config.device)

                    outputs = model(**tokens, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]

                    # Mean-pool over the sequence length to get a fixed-size vector
                    cls_embeddings = last_hidden_state[:, 0, :]
                    pool_embeddings = last_hidden_state[
                        :, 1 : len(tokens["input_ids"][0]) - 1, :
                    ].mean(dim=1)
                    sequence_embeddings = torch.cat(
                        (cls_embeddings, pool_embeddings), dim=1
                    )

                    all_embeddings.append(sequence_embeddings.cpu().numpy())

                except torch.cuda.OutOfMemoryError:
                    print(
                        f"CUDA Out of Memory on batch starting at index {i}. Skipping batch."
                    )
                    torch.cuda.empty_cache()
                    continue
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

    # 1. Load metadata and mutation data
    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
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
    print(f"Processing DMS ID: {dms_id}")
    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    # 2. Generate all sequences from mutations
    print("Generating sequences from mutations...")
    try:
        sequences, valid_indices = get_sequences(wt_seq, dms_df)
        dms_df_filtered = dms_df.loc[valid_indices]
    except Exception as e:
        print(f"Could not generate sequences for {dms_id}. Error: {e}")
        return

    # 3. Extract features
    embeddings = extract_features(model, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)})."
        )
        return

    # 4. Prepare labels and clean data
    true_labels = dms_df_filtered["dms_score"].values
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
        model_location=args.model_location,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Load the Hugging Face model and tokenizer once
    print(f"Loading model and tokenizer from: {config.model_location}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_location, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(config.model_location, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
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
        main(model, tokenizer, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

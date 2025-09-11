import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Assuming GenSLM is installed and accessible
from genslm import GenSLM, SequenceDataset


def get_sequences(wt_sequence: str, df: pd.DataFrame) -> tuple[list[str], pd.Index]:
    """
    Generates mutated sequences based on a wild-type sequence and a DataFrame of mutations.

    Args:
        wt_sequence (str): The wild-type DNA/RNA sequence.
        df (pd.DataFrame): DataFrame containing a column with mutation information.

    Returns:
        A tuple containing:
        - list[str]: A list of all generated sequences.
        - pd.Index: The original indices of the rows from which sequences were generated.
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
        description="Run GenSLM model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        required=True,
        help="Directory to cache or load the GenSLM model checkpoints.",
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
        default=4,
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

    model_name: str = "genslm_2.5B_patric"
    model_cache_dir: str = ""
    device: str = "cuda:0"
    batch_size: int = 4


def extract_features(
    model: GenSLM, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the last hidden state of the GenSLM model.

    Args:
        model: The pre-trained GenSLM model.
        sequences: A list of DNA sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    model.to(config.device)
    model.eval()

    seq_length = model.seq_length  # Use model's configured max sequence length
    dataset = SequenceDataset(sequences, seq_length, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features", unit="batch"):
            try:
                outputs = model(
                    batch["input_ids"].to(config.device),
                    attention_mask=batch["attention_mask"].to(config.device),
                    output_hidden_states=True,
                )

                # Use the last hidden state for embeddings
                last_hidden_state = outputs.hidden_states[-1]

                # Mean-pool over the sequence length, using attention mask to ignore padding
                emb = last_hidden_state.detach().cpu().numpy()
                all_embeddings.append(np.mean(emb, axis=1))

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA Out of Memory on a batch. Skipping batch.")
                torch.cuda.empty_cache()
                continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(model: GenSLM, config: InferenceConfig, args: argparse.Namespace, row_id: int):
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
    embeddings = extract_features(model, sequences, config)
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

    output_file = output_dir / f"{dms_id}.npy"
    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    config = InferenceConfig(
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Load the GenSLM model once
    print(f"Loading GenSLM model: {config.model_name}...")
    try:
        model = GenSLM(config.model_name, model_cache_dir=config.model_cache_dir)
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
        main(model, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

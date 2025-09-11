import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# Add the SPACE source directory to the Python path
sys.path.append('./SPACE')
from SPACE.model.modeling_space import Space, SpaceConfig


def get_sequences(wt_sequence: str, df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    Generates mutated sequences based on a wild-type sequence and a DataFrame of mutations.
    Also returns the filtered DataFrame corresponding to the generated sequences.
    """
    wt_sequence = wt_sequence.strip().upper().replace("U", "T")

    def apply_mutation(sequence, mutation_str):
        base_offset = 1
        pos = int(mutation_str[1:-1]) - base_offset
        original_base = mutation_str[0]
        new_base = mutation_str[-1]

        if original_base == "N":
            return sequence[: pos + 1] + new_base + sequence[pos + 1 :]
        elif new_base == "":
            return sequence[:pos] + sequence[pos + 1 :]
        else:
            if not (0 <= pos < len(sequence) and sequence[pos] == original_base):
                 raise AssertionError(f"Mutation '{mutation_str}' is inconsistent with sequence at position {pos+1}.")
            return sequence[:pos] + new_base + sequence[pos + 1 :]

    def apply_mutations(sequence, mutations_cell):
        if pd.isna(mutations_cell):
            return sequence
        for mutation_str in str(mutations_cell).split(','):
            sequence = apply_mutation(sequence, mutation_str.strip())
        return sequence

    mutation_column = next((col for col in df.columns if col.lower() in ["mutant", "mutation", "mutations"]), None)
    if not mutation_column:
        raise ValueError("No 'mutant', 'mutation', or 'mutations' column found in the DataFrame.")

    df_filtered = df.dropna(subset=[mutation_column]).copy()
    df_filtered['mutated_sequence'] = df_filtered[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    return df_filtered['mutated_sequence'].tolist(), df_filtered


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SPACE model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_name", type=str, default="yangyz1230/space",
        help="Hugging Face ID of the SPACE model to use."
    )
    parser.add_argument(
        "--row_id", type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed."
    )
    parser.add_argument(
        "--ref_sheet", type=str, required=True,
        help="Path to the reference sheet CSV containing DMS_ID and wild-type sequence columns."
    )
    parser.add_argument(
        "--dms_dir_path", type=str, required=True,
        help="Directory containing the DMS assay CSV files with mutation data."
    )
    parser.add_argument(
        "--output_dir_path", type=str, required=True,
        help="Directory where the output .npy files will be saved."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda:0' or 'cpu')."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Number of sequences per batch. SPACE is memory-intensive; use a small batch size."
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> pd.Series:
    """Load the reference sheet and retrieve the full row for a specific ID."""
    try:
        ref_df = pd.read_csv(ref_sheet_path, encoding="latin-1")
        ref_df.rename(columns={ref_df.columns[0]: "DMS_ID"}, inplace=True)
        if not (0 <= row_id < len(ref_df)):
            raise ValueError(f"Row ID {row_id} is out of bounds for the reference sheet.")
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
    model_name: str = "yangyz1230/space"
    device: str = "cuda:0"
    batch_size: int = 1
    max_sequence_length: int = 131072
    token_mapping: dict = lambda: {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'U': 3, '-': -1}

def extract_features(
    model: Space,
    sequences,
    config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the logits of the SPACE model.

    Args:
        model: The pre-trained SPACE model.
        sequences: A list of DNA sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of sequence embeddings.
    """
    model.to(config.device)
    model.eval()
    
    mapping = config.token_mapping()
    max_len = config.max_sequence_length
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences)), desc="Extracting Features", unit="batch"):
            sequence = sequences[i]
            if not sequence:
                continue

            try:
                max_length = 131072
                # Manually tokenize and pad sequences for SPACE model
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                elif len(sequence) < max_length:
                    sequence = sequence.ljust(max_length, '-')
                # Map ACGTN to 01234, -1 for padding
                mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': -1 , 'X': 4, "U": 3}
                tokens = torch.tensor([mapping[base] for base in sequence], dtype=torch.long)
                # Move tensors to the appropriate device
                tokens = tokens.to(config.device)


                # Pad sequences in the batch to the same length

                logits = model(tokens)['out'].unsqueeze(0) # Shape: (batch, seq_len, vocab_size)
                # Create attention mask to ignore padding during pooling
                cls_embeddings = logits[:, 0, :]  # Take the CLS token embeddings
                pooled_embeddings = logits[:, 1:, :].mean(dim=1)
                all_embedding = torch.cat([cls_embeddings, pooled_embeddings], dim=1)
                all_embeddings.append(all_embedding.cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA Out of Memory on batch starting at index {i}. Skipping batch.")
                torch.cuda.empty_cache()
                continue
    
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)


def main(model: Space, config: InferenceConfig, args: argparse.Namespace, row_id: int):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    assay_data = load_reference_data(args.ref_sheet, row_id)
    dms_id = assay_data["DMS_ID"]
    print(f"Processing DMS ID: {dms_id}")

    output_file = output_dir / f"{dms_id}.npy"
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping...")
        return

    wt_seq_col = next((col for col in assay_data.index if "raw" in col.lower() and "seq" in col.lower()), None)
    if not wt_seq_col:
        print(f"Could not find wild-type sequence column for {dms_id}. Skipping.")
        return
    wt_seq = assay_data[wt_seq_col]
    
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    print("Generating sequences from mutations...")
    try:
        sequences, dms_df_filtered = get_sequences(wt_seq, dms_df)
    except Exception as e:
        print(f"Could not generate sequences for {dms_id}. Error: {e}")
        return
    
    embeddings = extract_features(model, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match sequences ({len(sequences)}).")
        return

    score_col = next((col for col in dms_df_filtered.columns if "dms_score" in col), "dms_score")
    true_labels = dms_df_filtered[score_col].values
    
    print(f"Embeddings shape: {embeddings.shape}, Labels shape: {true_labels.shape}")

    valid_mask = np.isfinite(embeddings).all(axis=1) & ~np.isnan(true_labels)
    filtered_embeddings = embeddings[valid_mask]
    filtered_labels = true_labels[valid_mask]

    if filtered_embeddings.shape[0] == 0:
        print("No valid data remaining after filtering. Skipping save.")
        return

    result_array = np.concatenate([filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1)
    
    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    config = InferenceConfig(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )


    if args.row_id is not None:
        rows_to_process = [args.row_id]
    else:
        ref_df = pd.read_csv(args.ref_sheet, encoding="latin-1")
        rows_to_process = range(len(ref_df))

    for i, row_id in enumerate(rows_to_process):
        print("-" * 50)
            # Load the SPACE model once
        print(f"Loading SPACE model: {config.model_name}...")
        try:
            assay_data = load_reference_data(args.ref_sheet, row_id)
            dms_id = assay_data["DMS_ID"]
            input_path = Path(args.dms_dir_path) / f"{dms_id}.csv"
            model_config = SpaceConfig.from_pretrained('yangyz1230/space')
            model_config.input_file = str(input_path)
            model = Space.from_pretrained('yangyz1230/space', config=model_config)
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        print(f"Processing row {row_id} ({i + 1}/{len(rows_to_process)})...")
        main(model, config, args, row_id)
    
    print("-" * 50)
    print("All tasks completed.")
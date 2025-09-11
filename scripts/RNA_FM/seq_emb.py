import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/scripts/RNA_FM/RNA-FM")
import fm
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm


def setup_rna_fm_path(model_location):
    """Adds the RNA-FM model directory to the Python path."""
    try:
        # Assumes model_location is the parent directory of the 'fm' module
        sys.path.insert(0, str(Path(model_location).resolve()))
        import fm

        return fm
    except ImportError:
        print(
            f"Error: Could not import 'fm' from the provided model location: {model_location}"
        )
        print(
            "Please ensure the path points to the parent directory of the 'fm' package."
        )
        sys.exit(1)


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence for RNA-FM: converts to uppercase and T->U.
    """
    return str(sequence).strip().upper().replace("T", "U")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RNA-FM model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_location",
        type=str,
        required=True,
        help="Path to the parent directory of the RNA-FM 'fm' module.",
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
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Index of the transformer layer to extract embeddings from.",
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

    model_location: str
    device: str = "cuda:0"
    batch_size: int = 32
    layer: int = 12


def extract_features(
    model, alphabet, sequences: list[str], config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from a specified layer of the RNA-FM model.

    Args:
        model: The pre-trained RNA-FM model.
        alphabet: The model's alphabet object.
        sequences: A list of preprocessed RNA sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of mean-pooled sequence embeddings.
    """
    model.to(config.device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
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

            # RNA-FM's batch converter expects a list of tuples: (name, sequence)
            batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_sequences)]

            while True:
                try:
                    _, _, batch_tokens = batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(config.device)

                    results = model(batch_tokens, repr_layers=[config.layer])

                    # Extract representations from the specified layer
                    representations = results["representations"][config.layer]

                    # Mean-pool over sequence length, excluding BOS and EOS tokens
                    pooled_embeddings = representations[:, 1:-1, :].mean(dim=1)
                    cls_embeddings = representations[:, 0, :]
                    all_embedding = torch.cat(
                        [cls_embeddings, pooled_embeddings],
                        dim=1,
                    )
                    all_embeddings.append(all_embedding.cpu().numpy())
                    break  # Success, exit while loop

                except torch.cuda.OutOfMemoryError:
                    print(
                        f"CUDA Out of Memory on batch starting at index {i}. Retrying..."
                    )
                    torch.cuda.empty_cache()
                    continue  # Retry while loop

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def main(
    model, alphabet, config: InferenceConfig, args: argparse.Namespace, row_id: int
):
    """Main processing pipeline for a single DMS dataset."""
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    embeddings = extract_features(model, alphabet, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(
            f"Warning: Number of embeddings ({embeddings.shape[0]}) doesn't match sequences ({len(sequences)})."
        )
        return

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

    result_array = np.concatenate(
        [filtered_labels.reshape(-1, 1), filtered_embeddings], axis=1
    )

    np.save(output_file, result_array)
    print(f"Saved {result_array.shape[0]} results to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    # Setup path to import the local 'fm' module
    fm = setup_rna_fm_path(args.model_location)

    config = InferenceConfig(
        model_location=args.model_location,
        device=args.device,
        batch_size=args.batch_size,
        layer=args.layer,
    )

    # Load the RNA-FM model and alphabet once
    print("Loading RNA-FM model...")
    try:
        model, alphabet = fm.pretrained.rna_fm_t12()
        model.eval()
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
        main(model, alphabet, config, args, row_id)

    print("-" * 50)
    print("All tasks completed.")

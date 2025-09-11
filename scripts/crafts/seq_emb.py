import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# Add the CRAFTS_LM source directory to the Python path
# You may need to adjust this path depending on your project structure
base_dir = Path(__file__).parent.resolve()
sys.path.extend([str(base_dir), "/home/ma_run_ze/lzm/rnagym/fitness/baselines/crafts/crafts_lm"])
sys.path.append("/data_share/marunze/lzm/rnagym/fitness/scripts/crafts/crafts_lm")
from crafts_lm.utils.lm import get_extractor


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocesses a sequence by converting it to uppercase and stripping whitespace.
    """
    return str(sequence).strip().upper()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CRAFTS_LM model to extract features from DMS assay sequences."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Model name or path to the checkpoint."
    )
    parser.add_argument(
        "--row_id", type=int,
        help="Optional: Specific row ID in the reference sheet to process. If not set, all rows are processed."
    )
    parser.add_argument(
        "--ref_sheet", type=str, required=True,
        help="Path to the reference sheet CSV containing DMS_ID column."
    )
    parser.add_argument(
        "--dms_dir_path", type=str, required=True,
        help="Directory containing the DMS assay CSV files."
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
        "--batch_size", type=int, default=64,
        help="Number of sequences to process in each batch."
    )
    parser.add_argument(
        "--tok_mode", type=str, choices=["char", "word", "phone"], default="char",
        help="Tokenizer mode for the model."
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> pd.Series:
    """Load the reference sheet and retrieve the full row for a specific ID."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        ref_df.rename(columns={ref_df.columns[0]: "DMS_ID"}, inplace=True)
        if not (0 <= row_id < len(ref_df)):
            raise ValueError(f"Row ID {row_id} is out of bounds for the reference sheet.")
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
    model_name_or_path: str
    tok_mode: str = "char"
    device: str = "cuda:0"
    batch_size: int = 64


def extract_features(
    extractor,
    tokenizer,
    sequences: list[str],
    config: InferenceConfig
) -> np.ndarray:
    """
    Extracts embeddings from the last hidden state of the CRAFTS_LM model.

    Args:
        extractor: The pre-trained CRAFTS_LM feature extractor.
        tokenizer: The corresponding tokenizer.
        sequences: A list of preprocessed sequences.
        config: The configuration object for inference settings.

    Returns:
        A 2D numpy array of mean-pooled sequence embeddings.
    """
    extractor.to(config.device)
    extractor.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), config.batch_size), desc="Extracting Features", unit="batch"):
            batch_sequences = sequences[i : i + config.batch_size]
            if not batch_sequences:
                continue
            
            try:
                inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: torch.tensor(v).to(config.device) for k, v in inputs.items()}

                outputs = extractor(**inputs)
                
                # Robustly get the last hidden state and mean-pool over sequence length
                last_hidden_state = getattr(outputs, 'last_hidden_state', outputs)
                last_hidden_state = last_hidden_state if isinstance(last_hidden_state, torch.Tensor) else last_hidden_state[1]
                # Create attention mask to ignore padding during pooling
                attention_mask = torch.tensor(inputs['attention_mask'])
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_embeddings = sum_embeddings / sum_mask

                all_embeddings.append(pooled_embeddings.cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA Out of Memory on batch starting at index {i}. Skipping batch.")
                torch.cuda.empty_cache()
                continue
    
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)


def main(extractor, tokenizer, config: InferenceConfig, args: argparse.Namespace, row_id: int):
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
    
    embeddings = extract_features(extractor, tokenizer, sequences, config)
    if embeddings.shape[0] != len(sequences):
        print(f"Warning: Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({len(sequences)}).")
        return

    score_col = next((col for col in dms_df.columns if "dms_score" in col or "selex_score" in col), None)
    if not score_col:
        print(f"Error: No DMS/SELEX score column found in {dms_id}.csv. Skipping.")
        return
    true_labels = dms_df[score_col].values
    
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
        model_name_or_path=args.model_name_or_path,
        tok_mode=args.tok_mode,
        device=args.device,
        batch_size=args.batch_size
    )

    # Load the CRAFTS_LM model components once
    print(f"Loading CRAFTS_LM model from: {config.model_name_or_path}...")
    try:
        # The get_extractor function handles model and tokenizer loading
        extractor, tokenizer = get_extractor(args)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
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
        main(extractor, tokenizer, config, args, row_id)
    
    print("-" * 50)
    print("All tasks completed.")
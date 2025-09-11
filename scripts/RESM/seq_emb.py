import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import sys

sys.path.append("./RESM")
import rna_esm
from evo.tokenization import Vocab, mapdict
from model import ESM2
from dataset import RNADataset
from dataclasses import dataclass
import argparse
import pandas as pd


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess RNA sequence for DNA model:
    - Convert RNA (U) to DNA (T)
    - Convert to uppercase
    - Remove any whitespace

    Args:
        sequence: Input RNA or DNA sequence

    Returns:
        str: Preprocessed DNA sequence
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RESM model inference on DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        required=False,
        help="Row ID in the reference sheet to process",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=False,
        default="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv",
        help="Path to reference sheet containing DMS_ID column",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=False,
        default="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays",
        help="Directory containing DMS CSV files",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="RESM_650M",
        help="Base model to use: RESM_150M or RESM_650M (default: RESM_650M)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint file. If not provided, defaults will be used based on the base model.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load reference sheet and get DMS_ID for specified row.

    Args:
        ref_sheet_path: Path to reference sheet
        row_id: Row ID to process

    Returns:
        str: DMS_ID for specified row

    Raises:
        ValueError: If row_id is not found or DMS_ID is missing
    """
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if row_id >= len(ref_df):
            raise ValueError(
                f"Row ID {row_id} exceeds number of rows in reference sheet"
            )

        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}")

        return str(dms_id)

    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    except KeyError:
        raise KeyError("Reference sheet must contain 'DMS_ID' column")


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """
    Load DMS data for specified DMS_ID.

    Args:
        dms_dir_path: Directory containing DMS files
        dms_id: DMS ID to process

    Returns:
        pd.DataFrame: DataFrame containing sequences to process

    Raises:
        FileNotFoundError: If DMS file is not found
    """
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")

    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")
    return df


current_directory = Path(__file__).parent.absolute()


@dataclass
class DataConfig:
    pass


@dataclass
class OptimizerConfig:
    pass


@dataclass
class TrainConfig:
    pass


@dataclass
class TransformerConfig:
    pass


@dataclass
class LoggingConfig:
    pass


@dataclass
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: TransformerConfig = TransformerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    logging: LoggingConfig = LoggingConfig()
    fast_dev_run: bool = False
    resume_from_checkpoint: str = None
    val_check_interval: int = 1000


@dataclass
class InferenceConfig:
    architecture: str = "rna-esm"
    base_model: str = "RESM_650M"  # or "RESM_650M"
    data_path: str = str("./data")
    msa_path: str = str("")
    data_split: str = "extract_ss_data_alphaid.txt"
    model_path: str = None  # Will be set in __post_init__
    device: str = "cuda"
    output_dir: str = str("./data/features")
    max_seqlen: int = 1024

    def __post_init__(self):
        # Set model parameters and path based on base model
        if self.base_model == "RESM_150M":
            self.embed_dim = 640
            self.num_attention_heads = 20
            self.num_layers = 30
            if self.model_path is None:
                self.model_path = "./ckpt/RESM-150M-KDNY.ckpt"
        elif self.base_model == "RESM_650M":
            self.embed_dim = 1280
            self.num_attention_heads = 20
            self.num_layers = 33
            if self.model_path is None:
                self.model_path = "./ckpt/RESM-650M-KDNY.pt"
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")


def extract_features(config: InferenceConfig, seq) -> None:
    """Extract RNA embeddings and attention maps from RESM pre-trained model."""

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Initialize vocabularies and mapping
    if config.base_model == "RESM_150M":
        _, protein_alphabet = rna_esm.pretrained.esm2_t30_150M_UR50D()
    elif config.base_model == "RESM_650M":
        _, protein_alphabet = rna_esm.pretrained.esm2_t33_650M_UR50D()

    rnas = seq

    # Initialize dataset
    dataset = RNADataset(
        config.data_path,
        config.msa_path,
        rna_map_vocab,
        split_files=rnas,
        max_seqs_per_msa=1,
    )

    # Create output directories

    # Extract features
    with torch.no_grad():
        emb_list = []
        for rna_id, tokens in tqdm(dataset, desc="Extracting features"):
            tokens = tokens.unsqueeze(0).to(device)

            # Forward pass
            results = model(
                tokens, repr_layers=[config.num_layers], need_head_weights=True
            )

            # Extract embeddings
            embedding = results["representations"][config.num_layers]
            start_idx = int(rna_map_vocab.prepend_bos)
            end_idx = embedding.size(-2) - int(rna_map_vocab.append_eos)
            cls_embed = embedding[:, 0, :].squeeze(0).cpu().numpy()
            pool_embed = (
                embedding[:, start_idx:end_idx, :].mean(1).squeeze(0).cpu().numpy()
            )
            all_embed = np.concatenate([cls_embed, pool_embed])
            # Save embeddings
            emb_list.append(all_embed)

    print(
        "Feature extraction completed! Shape of embeddings:", np.array(emb_list).shape
    )

    return np.array(emb_list)


def main(args):
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DMS ID from reference sheet
    dms_id = load_reference_data(args.ref_sheet, args.row_id)
    print(f"Processing DMS ID: {dms_id}")

    # Load DMS data
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    # Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [
        preprocess_sequence(seq)
        for seq in tqdm(dms_df["sequence"].tolist(), desc="Preprocessing", unit="seq")
    ]

    # Run inference in batches
    print(f"Running inference")
    embed = extract_features(config, sequences)
    # 1. 准备数据
    true_labels = dms_df["DMS_score"].values
    print(f"Embed shape: {embed.shape}, True labels shape: {true_labels.shape}")
    # 2. 把 inf 替换为 NaN
    embed[~np.isfinite(embed)] = np.nan
    mask = (
        ~np.isnan(true_labels) & np.isfinite(embed).all(axis=1)
        if embed.ndim > 1
        else ~np.isnan(true_labels) & np.isfinite(embed)
    )
    embed = embed[mask]
    true_labels = true_labels[mask]

    # Saving results
    result_array = np.concatenate(
        [true_labels.reshape(-1, 1), embed], axis=1
    )  # Shape: (num_samples, 1 + embed_dim)
    output_file = output_dir / f"{dms_id}.npy"
    np.save(output_file, result_array)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    reference_sheet_path = Path(args.ref_sheet)
    df = pd.read_csv(reference_sheet_path)
    total_rows = len(df)
    while True:
        try:
            config = InferenceConfig()
            config.base_model = args.checkpoint

            # Initialize model
            @dataclass
            class TransformerConfig:
                embed_dim: int = config.embed_dim
                num_attention_heads: int = config.num_attention_heads
                num_layers: int = config.num_layers
                max_seqlen: int = config.max_seqlen
                dropout: float = 0.1
                attention_dropout: float = 0.1
                activation_dropout: float = 0.1
                attention_type: str = "standard"
                performer_attention_features: int = 256

            if config.base_model == "RESM_150M":
                _, protein_alphabet = rna_esm.pretrained.esm2_t30_150M_UR50D()
            elif config.base_model == "RESM_650M":
                _, protein_alphabet = rna_esm.pretrained.esm2_t33_650M_UR50D()

            rna_alphabet = rna_esm.data.Alphabet.from_architecture(config.architecture)

            protein_vocab = Vocab.from_esm_alphabet(protein_alphabet)
            rna_vocab = Vocab.from_esm_alphabet(rna_alphabet)
            rna_to_protein = {"A": "K", "U": "D", "C": "N", "G": "Y"}
            rna_map_dict = mapdict(protein_vocab, rna_vocab, rna_to_protein)
            rna_map_vocab = Vocab.from_esm_alphabet(rna_alphabet, rna_map_dict)

            model = ESM2(
                vocab=protein_vocab,
                model_config=TransformerConfig(),
                optimizer_config=OptimizerConfig(),  # Use the defined class
                contact_train_data=None,
                token_dropout=False,  # Disable dropout for inference
            )

            ckpt = torch.load(config.model_path, map_location="cpu")

            new_ckpt = {}

            for k, v in ckpt.items():
                if k.startswith("module."):
                    new_ckpt[k[len("module.") :]] = v
                else:
                    new_ckpt[k] = v
            # Load model weights
            model.load_state_dict(
                new_ckpt,
                strict=True,
            )
            model = model.eval()
            model = model.to(args.device)
            print("Model loaded successfully.")
            break
        except torch.OutOfMemoryError:
            print("CUDA out of memory error")
        model.eval()
    for row_id in range(total_rows):
        config = InferenceConfig()
        if args.checkpoint:
            config.base_model = args.checkpoint
        if args.device:
            config.device = args.device
        if args.model_path:
            config.model_path = args.model_path
        print("MSA file path:", config.msa_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        args.row_id = row_id
        print(f"Processing row {row_id + 1}/{total_rows}...")
        args.ref_sheet = reference_sheet_path
        args.dms_dir_path = Path(args.dms_dir_path)
        args.output_dir_path = Path(args.output_dir_path)
        main(args)

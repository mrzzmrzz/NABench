import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

#!/usr/bin/env python3
"""
Script to run Evo model inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import PreTrainedModel
import re
import torch.nn.functional as F

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
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
    return sequence.strip().upper().replace('U', 'T')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo model inference on DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        required=False,
        help="Row ID in the reference sheet to process"
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=False,
        default="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv",
        help="Path to reference sheet containing DMS_ID column"
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=False,
        default="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays",
        help="Directory containing DMS CSV files"
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available, else cpu)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing sequences (default: 16)"
    )

    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=None,
        help="Number of samples for few-shot evaluation (default: None, use full data)"
    )
    parser.add_argument(
        "--few_shot_repeat",
        type=int,
        default=5,
        help="Number of repeats for few-shot evaluation (default: 5)"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable cross-validation"
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
            raise ValueError(f"Row ID {row_id} exceeds number of rows in reference sheet")
        
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

class SequenceDataset(Dataset):
    """Dataset for processing sequences in batches."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def evaluate(embeddings: np.ndarray, scores: np.ndarray, cv=False, few_shot_k=None, few_shot_repeat=5, seed=42):
    np.random.seed(seed)
    
    mask = ~np.isnan(scores)
    if not mask.all():
        num_nan = len(scores) - mask.sum()
        print(f"Warning: {num_nan} samples have NaN scores and will be excluded from evaluation")

    emb = embeddings[mask]
    sc = scores[mask]
    
    # Few-shot 模式
    if few_shot_k is not None:
        print(f"Running few-shot evaluation with k={few_shot_k}, repeated {few_shot_repeat} times")
        corrs = []
        best_model = None
        for r in range(few_shot_repeat):
            indices = np.random.choice(len(sc), size=few_shot_k, replace=False)
            emb_train = emb[indices]
            sc_train = sc[indices]

            emb_test = np.delete(emb, indices, axis=0)
            sc_test = np.delete(sc, indices)

            model = Ridge(alpha=1.0)
            model.fit(emb_train, sc_train)
            preds = model.predict(emb_test)

            corr, pval = spearmanr(preds, sc_test)
            corrs.append(corr)
            if best_model is None or corr > np.mean(corrs):
                best_model = model
        avg_emb = best_model.predict(emb)
        print(f"Average correlation over {few_shot_repeat} repeats: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
        print("Shape of average embedding:", avg_emb.shape)
        return np.mean(corrs), np.std(corrs), avg_emb

    # 原始 CV 模式
    if cv:
        model = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_values=True)
        model.fit(emb, sc)
        preds = model.predict(emb)
        corr, pval = spearmanr(preds, sc)
        avg_emb = preds
    else:
        avg_emb = embeddings[:,0]
        corr, pval = spearmanr(avg_emb, scores)
    
    return corr, pval, avg_emb



def run_inference(model, tokenizer, sequences: list, device: str, batch_size: int) -> np.ndarray:
    """
    Run Evo model inference on sequences in batches.
    
    Args:
        model: Loaded Evo model
        tokenizer: Evo tokenizer
        sequences: List of sequences to process
        device: Device to run inference on
        batch_size: Number of sequences to process in each batch
        
    Returns:
        np.ndarray: Log probabilities array
    """
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    all_embed = []
    
    # Create progress bar
    total_batches = len(dataloader)
    pbar = tqdm(
        total=len(sequences),
        desc="Scoring sequences",
        unit="seq"
    )
    for i in range(0,len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]

        tok_seq = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True)
        tok_seq = {k: v.cuda() for k, v in tok_seq.items() if k != "token_type_ids"}
        # place on device, convert to tensor

        model.to(device)
        with torch.no_grad():
            embed = model(**tok_seq)[0]  # get logits from model
            embed = embed.mean(dim=1)

        all_embed.extend(embed.cpu().numpy())
    # tunc dim 1 to be the same
    
    return all_embed
    

def main(args):

    

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_file = output_dir / f"correlation_summary.txt"


    # Load DMS ID from reference sheet
    dms_id = load_reference_data(args.ref_sheet, args.row_id)
    print(f"Processing DMS ID: {dms_id}")
    with open(correlation_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:

            line = line.strip()
            last_dms_id, last_corr, last_pval = line.split(',')
            if last_dms_id == dms_id:
                print(f"Skipping {dms_id} as it has already been processed with correlation {last_corr} and p-value {last_pval}")
                return
    # Load DMS data
    dms_df = load_dms_data(args.dms_dir_path, dms_id)
    
    # Preprocess sequences
    print("Preprocessing sequences...")
    sequences = [preprocess_sequence(seq) for seq in tqdm(
        dms_df["sequence"].tolist(),
        desc="Preprocessing",
        unit="seq"
    )]
    print(f"Preprocessed {len(sequences)} sequences")

    max_length = 1000000  # auto selects
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)


        
    # Run inference in batches
    print(f"Running inference with batch size {args.batch_size}...")
    embed = run_inference(
        model, 
        tokenizer, 
        sequences, 
        args.device, 
        args.batch_size
    )
    # 1. 准备数据
    embed = np.array(embed, dtype=np.float64)       # 用 float64 更稳健
    true_labels = dms_df["DMS_score"].values

    # 2. 把 inf 替换为 NaN
    embed[~np.isfinite(embed)] = np.nan
    mask = np.isfinite(embed).all(axis=1) & ~np.isnan(true_labels)
    embed = embed[mask] 
    true_labels = true_labels[mask]
    print("Masked rows:", len(true_labels), "out of", len(dms_df))
    # 3. 构造 pipeline：先填补，再缩放，再回归
    # mdl = make_pipeline(
    #     SimpleImputer(strategy='mean'),    # 或者 'median'
    #     StandardScaler(),                  # 均值 0 方差 1
    #     Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    # )

    # # 4. 交叉验证计算 MSE
    # scores = cross_val_score(
    #     mdl, embed, true_labels,
    #     cv=5,
    #     scoring='neg_mean_squared_error'
    # )
    # print("5-fold CV MSE:", -scores)

    # 5. 交叉验证预测
    # y_pred = cross_val_predict(mdl, embed, true_labels, cv=5)
    # sequence_scores = y_pred       # Add scores to DataFrame
    correlation, pvalue, sequence_scores = evaluate(embed, true_labels, cv=args.cv, few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)

    score_column = f"{args.model_name.replace('-', '_')}_score"
    dms_df[score_column] = np.nan  # Initialize with NaN
    dms_df.loc[mask, score_column] = sequence_scores
    
    # Calculate Spearman correlation
    # correlation, pvalue = spearmanr(dms_df['DMS_score'], dms_df[score_column])
    print(f"Spearman correlation: {correlation:.3f}, p-value: {pvalue:.2e}")
    # Save results
    output_file = output_dir / f"{dms_id}.csv"
    dms_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Spearman correlation with DMS scores: {correlation:.3f} (p-value: {pvalue:.2e})")

    print(f"Output saved to: {output_file}")

    #Save the Spearman correlation
    correlation_file = output_dir / f"correlation_summary.txt"
    if not correlation_file.exists():
        with open(correlation_file, 'w') as f:
            f.write("DMS_ID,Spearman_Correlation\n")
    # Append the correlation result
    with open(correlation_file, 'a') as f:
        f.write(f"{dms_id},{correlation:.3f},{pvalue:.3e}\n")



if __name__ == "__main__":
    args = parse_args()
    reference_sheet_path = Path(args.ref_sheet)
    df = pd.read_csv(reference_sheet_path)
    total_rows = len(df)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    for row_id in range(total_rows):
        args.row_id = row_id
        print(f"Processing row {row_id + 1}/{total_rows}...")
        args.ref_sheet = reference_sheet_path
        args.dms_dir_path = Path(args.dms_dir_path)
        args.output_dir_path = Path(args.output_dir_path)
        args.model_name = "DNABERT-2-117M"
        main(args)
# python score_evo_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path evo_output --model_name evo-1-8k-base
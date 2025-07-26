#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for LucaOne on DMS assays
"""
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge

import torch
import sys
sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/LucaOneTasks/src/llm/lucagplm/")
# 引入LucaOne推理函数
from get_embedding import predict_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LucaOne on DMS assay data")
    parser.add_argument("--ref_sheet", type=str, required=True,
                        help="Path to reference CSV listing DMS assays and metadata")
    parser.add_argument("--dms_dir", type=str, required=True,
                        help="Directory containing per-assay DMS CSV files (with columns 'sequence' and 'score')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--llm_dir", type=str, required=True,
                        help="Base directory of LucaOne model (parent of llm/models/...)")
    parser.add_argument("--seq_type", type=str, choices=["gene", "prot"], default="gene",
                        help="Sequence type: gene or prot")
    parser.add_argument("--embedding_type", type=str, choices=["matrix", "vector"], default="vector",
                        help="Type of embedding to extract")
    parser.add_argument("--truncation_seq_length", type=int, default=4094,
                        help="Max sequence length for LucaOne inference (not include special tokens)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding inference")
    parser.add_argument("--cv", action='store_true',
                        help="Use cross-validation for evaluation")
    parser.add_argument("--few_shot_k", type=int, default=None,
                        help="Number of samples for few-shot evaluation (default: None, use full data)")
    parser.add_argument("--few_shot_repeat", type=int, default=5,
                        help="Number of repeats for few-shot evaluation (default: 5)")
    parser.add_argument("--cache_dir", type=str, default="/data4/marunze/lucaone/cache/",
                        help="Directory to cache embeddings (optional, for large datasets)")
    return parser.parse_args()


def load_reference(ref_sheet: Path, row_id: int):
    df = pd.read_csv(ref_sheet)
    if row_id < 0 or row_id >= len(df):
        raise IndexError(f"row_id {row_id} out of range (0..{len(df)-1})")
    return df.iloc[row_id]


def load_dms_data(dms_dir: Path, assay_name: str):
    # 假设文件名为 {assay_name}.csv
    csv_path = dms_dir / f"{assay_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DMS file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # 期待列 'sequence' 和 'DMS_score'
    if 'sequence' not in df or 'DMS_score' not in df:
        raise ValueError("DMS CSV must contain 'sequence' and 'DMS_score' columns")
    return df['sequence'].tolist(), df['DMS_score'].values , df


def batch_embed(sequences, args):
    embeddings = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seq_id, seq in tqdm(enumerate(sequences), total=len(sequences), desc="Embedding seqs"):
        emb, _ = predict_embedding(
            args.llm_dir,
            [str(seq_id), args.seq_type, seq],
            args.truncation_seq_length,
            args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=args.truncation_seq_length,
            device=device,
            matrix_add_special_token=False
        )
        if emb is None:
            raise RuntimeError(f"Embedding failed for sequence {seq_id}")
        # 如果是矩阵类型，可处理为CLS向量或平均
        if args.embedding_type == 'matrix':
            # 默认取第一个位置CLS
            vec = emb[0]
        else:
            vec = emb
        embeddings.append(vec)
    return np.vstack(embeddings)


import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
import random

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
        if few_shot_k > len(sc):
            print(f"Warning: few_shot_k ({few_shot_k}) is larger than available samples ({len(sc)}), skipping few-shot evaluation")
            return None, None, None
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




def main():
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "correlation_summary.txt"

    with open(out_path, 'w') as f:
        f.write(f"DMS_ID,Spearman_Correlation\n")
    for row_id in range(0, 45): 
        # 加载参考sheet行
        ref = load_reference(Path(args.ref_sheet), row_id)
        assay_name = ref['assay_name'] if 'assay_name' in ref else ref[0]
        print(f"Evaluating assay: {assay_name}")

        # 加载DMS数据
        sequences, scores, df = load_dms_data(Path(args.dms_dir), assay_name)
        print(f"Loaded {len(sequences)} sequences")


        # 批量嵌入
        if args.cache_dir:
            cache_file = Path(args.cache_dir) / f"{assay_name}_embeddings.npy"
            if cache_file.exists():
                print(f"Loading cached embeddings from {cache_file}")
                embed = np.load(cache_file)
                true_labels = scores
            else:
                embed = batch_embed(sequences, args)
                embed = np.array(embed)       # 用 float64 更稳健
                true_labels = scores
                np.save(cache_file, embed)
                print(f"Saved embeddings to {cache_file}")
        else:
            embed = batch_embed(sequences, args)
            embed = np.array(embed)
            true_labels = scores
        # 2. 把 inf 替换为 NaN
        embed[~np.isfinite(embed)] = np.nan
        mask = np.isfinite(embed).all(axis=1) & ~np.isnan(true_labels)
        embed = embed[mask] 
        true_labels = true_labels[mask]
        csv_file = args.output_dir / f"{assay_name}.csv"
        # 评估
        corr, pval, pred_scores = evaluate(embed, true_labels, cv=args.cv,
                                few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)
        if corr is None:
            print(f"Skipping {assay_name} due to few-shot evaluation failure")
            continue
        print(f"Spearman correlation: {corr:.3f}, p-value: {pval:.2e}")
        # 将预测分数添加到 DataFrame'
        df["LucaOne_scores"] = np.nan
        df.loc[mask, "LucaOne_scores"] = pred_scores    
        df.to_csv(csv_file, index=False)
        # 保存结果
        with open(out_path, 'a') as f:
            f.write(f"{assay_name},{corr},{pval}\n")
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

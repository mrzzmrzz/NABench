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
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
from enformer_pytorch import from_pretrained

#Config
learning_rate = 1e-4
num_epochs = 50
batch_size = 64
train_batch_size = 32
class MLPHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim//8),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(input_dim//8, input_dim//64),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(input_dim//64, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LucaOne on DMS assay data")
    parser.add_argument("--ref_sheet", type=str, required=True,
                        help="Path to reference CSV listing DMS assays and metadata")
    parser.add_argument("--dms_dir", type=str, required=True,
                        help="Directory containing per-assay DMS CSV files (with columns 'sequence' and 'score')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
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
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache model files (optional, defaults to model_dir/cache)")


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
    # 归一化 DMS_score
    if df['DMS_score'].dtype == 'object':
        # 如果是字符串类型，尝试转换为数值
        df['DMS_score'] = pd.to_numeric(df['DMS_score'], errors='coerce')
    if df['DMS_score'].isnull().any():
        raise ValueError("DMS_score column contains NaN values after conversion")
    # 归一化到0-1范围
    df['DMS_score'] = (df['DMS_score'] - df['DMS_score'].min()) / (df['DMS_score'].max() - df['DMS_score'].min())

    return df['sequence'].tolist(), df['DMS_score'].values , df


def batch_embed(cat_name,sequences,model,args, batch_size=64):
    # Look for the cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(args.llm_dir) / "cache"    
    # Check if embeddings are already cached
    cache_file = cache_dir / f"{cat_name}_embeddings.npy"
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(sequences)
    print("Processing total sequence of:", N)

    # 预先拿到 mapping，避免循环里重复创建
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4,'-':-1, 'X':4,"U":3}
    max_length = 196608
    scores = []

    for start in tqdm(range(0, N, batch_size), desc="Batched processing"):
        batch_seqs = sequences[start : start + batch_size]
        B = len(batch_seqs)

        # 准备一个 (B, max_length) 的 numpy.array，先填 -1
        arr = np.full((B, max_length), fill_value=-1, dtype=np.int64)
        for i, seq in enumerate(batch_seqs):
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq.ljust(max_length, '-')
            # 转成数字
            arr[i] = [mapping[b] for b in seq]

        # 转成 Tensor 并移动到 GPU
        tokens = torch.from_numpy(arr).to(device)

        with torch.no_grad():
            # logits: (B, L, C)
            logits = model(tokens)["human"]
            # 如果想用 log-probs： 
            # logits = F.log_softmax(logits, dim=-1)

            # (B, C) = mean over length dim
            avg_logits = logits.mean(dim=1)  

        # 移回 CPU、转 numpy
        scores.append(avg_logits.cpu().numpy())

    # 把所有 batch 的结果合并成 (N, C)
    scores = np.vstack(scores)
    print("Scores shape:", scores.shape)
        

    # Cache the embeddings
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.llm_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cat_name}_embeddings.npy"
    np.save(cache_file, scores)

    return scores


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
    category = ["ribozyme", "promoter", "aptamer", "tRNA","mRNA","enhancer"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print('Loading model...')
    model = from_pretrained('EleutherAI/enformer-official-rough').to(device)
    model.eval()
    with open(out_path, 'w') as f:
        f.write(f"Training_Set,Test_Set,Spearman_Correlation,P_value\n")
    for cat in category:
        training_seq=[]
        training_scores=[]
        test_seq={k:[] for k in category if k != cat}
        test_scores={k:[] for k in category if k != cat}
        for row_id in range(0, 45): 
            # 加载参考sheet行
            ref = load_reference(Path(args.ref_sheet), row_id)
            assay_name = ref['assay_name'] if 'assay_name' in ref else ref[0]
            print(f"Processing: {assay_name}")

            # 加载DMS数据
            sequences, scores, df = load_dms_data(Path(args.dms_dir), assay_name)
            print(f"Loaded {len(sequences)} sequences")
            if cat in assay_name:
                # 训练集
                training_seq.extend(sequences)
                training_scores.extend(scores)
            else:
                # 测试集
                for i in category:
                    if i in assay_name:
                        test_seq[i].extend(sequences)
                        test_scores[i].extend(scores)

        # Training on the training set
        print(f"Training on {len(training_seq)} sequences")
        traning_emb = batch_embed(cat,training_seq,model, args,batch_size=args.batch_size)
        traning_emb = np.array(traning_emb)  # 用 float64 更稳健
        training_scores = np.array(training_scores)
        # delete invalid data
        traning_emb[~np.isfinite(traning_emb)] = np.nan
        mask = np.isfinite(traning_emb).all(axis=1) & ~np.isnan(training_scores)
        traning_emb = traning_emb[mask]
        training_scores = training_scores[mask]

        # Training loop
        head = MLPHead(input_dim=traning_emb.shape[1], output_dim=1)
        head.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        head.train()
        sequences = torch.tensor(traning_emb, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        scores = torch.tensor(training_scores, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            head.train()
            for i in tqdm(range(0, len(sequences), train_batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_sequences = sequences[i:i + train_batch_size]
                batch_scores = scores[i:i + train_batch_size].unsqueeze(1)
                optimizer.zero_grad()
                outputs = head(batch_sequences)
                loss = criterion(outputs, batch_scores)
                loss.backward()
                optimizer.step()
            # Print loss and correlation every epoch
            head.eval()
            with torch.no_grad():
                total_loss = 0.0
                for i in range(0, len(sequences), train_batch_size):
                    batch_sequences = sequences[i:i + train_batch_size]
                    batch_scores = scores[i:i + train_batch_size].unsqueeze(1)
                    outputs = head(batch_sequences)
                    loss = criterion(outputs, batch_scores)
                    total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(sequences):.4f}")
        # Calculate the correlation on the training set
        with torch.no_grad():
            head.eval()
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                if i == 0:
                    # 如果是第一个batch，直接计算
                    train_outputs = head(batch_sequences).cpu().numpy().squeeze()
                else:
                    # 否则累加结果
                    batch_outputs = head(batch_sequences).cpu().numpy().squeeze()
                    train_outputs = np.concatenate((train_outputs, batch_outputs), axis=0)
        mask = np.isfinite(train_outputs) & ~np.isnan(training_scores)
        train_outputs = train_outputs[mask]
        training_scores = training_scores[mask]
        train_corr, train_pval = spearmanr(train_outputs, training_scores)
        # head = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_values=True)
        # head.fit(traning_emb, training_scores)
        # train_preds = head.predict(traning_emb)
        # train_corr, train_pval = spearmanr(train_preds, training_scores)
        print(f"Training set correlation: {train_corr:.3f}, p-value: {train_pval:.2e}")
        with open(out_path, 'a') as f:
            f.write(f"{cat},{cat},{train_corr},{train_pval}\n")


        # Evaluate on the test set
        for test_cat in test_seq.keys():
            if not test_seq[test_cat]:
                print(f"No sequences for category {test_cat}, skipping...")
                continue
            print(f"Evaluating on {len(test_seq[test_cat])} sequences for category {test_cat}")
            # 批量嵌入
            test_emb = batch_embed(test_cat,test_seq[test_cat],model, args,batch_size=args.batch_size)
            test_emb = np.array(test_emb)
            test_scores_cat = np.array(test_scores[test_cat])
            # delete invalid data
            test_emb[~np.isfinite(test_emb)] = np.nan
            mask = np.isfinite(test_emb).all(axis=1) & ~np.isnan(test_scores_cat)
            test_emb = test_emb[mask]
            test_scores_cat = test_scores_cat[mask]

            if len(test_emb) == 0 or len(test_scores_cat) == 0:
                print(f"No valid data for category {test_cat}, skipping...")
                continue

            # Eval
            head.eval()
            test_sequences = torch.tensor(test_emb, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            with torch.no_grad():
                for i in range(0, len(test_sequences), batch_size):
                    batch_sequences = test_sequences[i:i + batch_size]
                    if i == 0:
                        # 如果是第一个batch，直接计算
                        test_outputs = head(batch_sequences).cpu().numpy().squeeze()
                    else:
                        # 否则累加结果
                        batch_outputs = head(batch_sequences).cpu().numpy().squeeze()


                        if batch_outputs.ndim == 0:
                            continue
                        test_outputs = np.concatenate((test_outputs, batch_outputs), axis=0)
            # test_sequences = torch.tensor(test_emb, dtype=torch.float32)
                
            # test_outputs = head.predict(test_sequences)
            test_corr, test_pval = spearmanr(test_outputs, test_scores_cat)
            print(f"Test set correlation for {test_cat}: {test_corr:.3f}, p-value: {test_pval:.2e}")
            # 保存结果
            with open(out_path, 'a') as f:
                f.write(f"{cat},{test_cat},{test_corr},{test_pval}\n")
        


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_contiguous.py  ——  LucaOne + Contiguous-CV（重叠突变版）
(2025-07-13  update: fold 测试集含“至少一个位点落入区间”的样本；若>20%则随机截断)
"""

import argparse, re, sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import torch

# LucaOne 依赖
sys.path.append(
    "/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/"
    "LucaOneTasks/src/llm/lucagplm/"
)
from get_embedding import predict_embedding


# ────────────── 工具函数 ────────────── #
def parse_mutation(mut_str: str):
    """'G1A,U2C' → [1,2]；空字符串 / NaN → []"""
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]


def contiguous_intervals(all_pos, k=5):
    """基于出现的突变位点均分 k 段"""
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]


def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    model = RidgeCV(alphas=alphas, store_cv_values=False)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


# ────────────── LucaOne 嵌入 ────────────── #
def batch_embed(dms_id,seqs, args):
    # Look up cache directory
    cache_file = Path(args.cache_dir) / f"{dms_id}_embeddings.npy"
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vecs = []
    for sid, seq in tqdm(enumerate(seqs), total=len(seqs), desc="Embedding"):
        emb, _ = predict_embedding(
            args.llm_dir,
            [str(sid), args.seq_type, seq],
            args.truncation_seq_length,
            args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=args.truncation_seq_length,
            device=device,
            matrix_add_special_token=False,
        )
        vecs.append(emb[0] if args.embedding_type == "matrix" else emb)
    vecs = np.array(vecs)
    # Save to cache
    print(f"Saving embeddings to {cache_file}")
    np.save(cache_file, vecs)
    return np.vstack(vecs)


# ────────────── Contiguous-CV (重叠版) ────────────── #
def evaluate_contiguous_overlap(
    emb, scores, pos_lists, n_folds=5, max_frac=0.2, seed=0
):
    N = len(scores)
    rng = np.random.RandomState(seed)
    preds_all = np.full(N, np.nan)

    # 1. 生成区间
    all_pos = {p for pl in pos_lists for p in pl}
    intervals = contiguous_intervals(all_pos, n_folds)
    K = len(intervals)

    fold_corrs, fold_mses = [], []
    for k, (lo, hi) in enumerate(intervals):
        # 2. 选出“至少一个位点落在区间”的样本
        test_idx = [
            i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)
        ]
        # 若过大则随机下采样至 20 %
        max_size = int(max_frac * N)
 

        train_idx = [i for i in range(N) if i not in test_idx]
        if len(test_idx) > max_size:
            test_idx = rng.choice(test_idx, size=max_size, replace=False).tolist()
        if not test_idx:
            continue
        print("Using fold", k + 1, "with",len(train_idx),"train samples and" ,len(test_idx), "test samples")
        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Warning: Fold {k + 1} has too few training samples ({len(train_idx)}). Skipping.")
            continue
        Xtr, ytr = emb[train_idx], scores[train_idx]
        Xte, yte = emb[test_idx], scores[test_idx]

        y_pred = ridge_pred(Xtr, ytr, Xte)
        preds_all[test_idx] = y_pred

        fold_corrs.append(spearmanr(y_pred, yte).correlation)
        fold_mses.append(mean_squared_error(yte, y_pred))

    return {
        "corr_mean": np.nanmean(fold_corrs),
        "corr_std":  np.nanstd(fold_corrs),
        "mse_mean":  np.nanmean(fold_mses),
        "mse_std":   np.nanstd(fold_mses),
        "fold_corr": fold_corrs,
        "fold_mse":  fold_mses,
        "preds":     preds_all,
    }


# ────────────── CLI & 主流程 ────────────── #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--llm_dir", required=True)
    p.add_argument("--seq_type", choices=["gene", "prot"], default="gene")
    p.add_argument("--embedding_type", choices=["matrix", "vector"], default="vector")
    p.add_argument("--truncation_seq_length", type=int, default=4094)
    p.add_argument("--mutation_col", default="mutant")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--cache_dir", default="/data4/marunze/lucaone/cache/")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "correlation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("DMS_ID,Spearman_mean,Spearman_std\n")

    ref_df = pd.read_csv(args.ref_sheet)

    for _, row in ref_df.iterrows():
        assay_id = row.get("DMS_ID", row.iloc[0])
        csv_path = Path(args.dms_dir) / f"{assay_id}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} 不存在")
            continue

        df = pd.read_csv(csv_path)
        if not {"sequence", "DMS_score", args.mutation_col}.issubset(df.columns):
            print(f"[skip] {assay_id}: 缺少必须列")
            continue

        sequences = df["sequence"].tolist()
        scores = df["DMS_score"].values.astype(float)
        pos_lists = df[args.mutation_col].apply(parse_mutation).tolist()

        emb = batch_embed(assay_id,sequences, args)
        mask = np.isfinite(emb).all(axis=1) & np.isfinite(scores)
        emb, scores = emb[mask], scores[mask]
        pos_lists = [pl for (pl, m) in zip(pos_lists, mask) if m]

        res = evaluate_contiguous_overlap(
            emb,
            scores,
            pos_lists,
            n_folds=args.n_folds,
            max_frac=args.max_test_frac,
        )

        print(
            f"[{assay_id}] Spearman: {res['corr_mean']:.3f} ± {res['corr_std']:.3f}"
        )

        # 保存 CSV（带预测）
        df["LucaOne_scores"] = np.nan
        df.loc[mask, "LucaOne_scores"] = res["preds"]
        df.to_csv(out_dir / f"{assay_id}.csv", index=False)

        with open(summary_path, "a") as f:
            f.write(f"{assay_id},{res['corr_mean']},{res['corr_std']}\n")

    print(f"\n全部评测完毕，汇总写入 {summary_path}")


if __name__ == "__main__":
    main()

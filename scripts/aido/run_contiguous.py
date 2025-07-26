#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_aido_contiguous.py —— Aido RNA-1B600M + Contiguous-CV 评测
2025-07-13
"""

import argparse, json, re, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

import torch
from modelgenerator.tasks import Embed

# ─────────────────── 序列预处理 ──────────────────── #
def preprocess_sequence(seq: str) -> str:
    return seq.strip().upper().replace("U", "T")


# ─────────────────── 突变解析 & 区间工具 ───────────── #
def parse_mutation(mut_str: str):
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]


def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]


# ─────────────────── 评估：Contiguous-CV (重叠+截断) ─────────────────── #
def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    mdl = RidgeCV(alphas=alphas, store_cv_results=False)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)


def evaluate_contiguous(
    emb, scores, pos_lists, n_folds=5, max_frac=0.20, seed=0
):
    N = len(scores)
    rng = np.random.RandomState(seed)
    preds_all = np.full(N, np.nan)

    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    fold_corrs, fold_mses = [], []

    for k, (lo, hi) in enumerate(intervals):
        candidate = [i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)]
        if not candidate:
            continue

        max_size = int(max_frac * N)
        if len(candidate) > max_size:
            test_idx = rng.choice(candidate, size=max_size, replace=False).tolist()
        else:
            test_idx = candidate

        # 训练集排除所有 candidate（多余样本被弃置）
        train_idx = [i for i in range(N) if i not in candidate]
        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Warning: Fold {k + 1} has too few training samples ({len(train_idx)}). Skipping.")
            continue
        Xtr, ytr = emb[train_idx], scores[train_idx]
        Xte, yte = emb[test_idx], scores[test_idx]

        y_pred = ridge_pred(Xtr, ytr, Xte)
        preds_all[test_idx] = y_pred

        fold_corrs.append(spearmanr(y_pred, yte).correlation)
        fold_mses.append(mean_squared_error(yte, y_pred))

        print(
            f"Fold {k+1}: train={len(train_idx)}, test={len(test_idx)}, "
            f"ρ={fold_corrs[-1]:.3f}"
        )

    return {
        "corr_mean": np.nanmean(fold_corrs),
        "corr_std":  np.nanstd(fold_corrs),
        "mse_mean":  np.nanmean(fold_mses),
        "mse_std":   np.nanstd(fold_mses),
        "preds":     preds_all,
    }


# ─────────────────── Aido 嵌入 (含缓存) ─────────────────── #
def embed_sequences(dms_id, sequences, model, device, cache_dir, batch=64):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dms_id}_emb.npy"
    if cache_file.exists():
        print(f"Loading cached embeddings: {cache_file}")
        return np.load(cache_file)

    model.to(device).eval()
    all_vec = []
    for i in tqdm(range(0, len(sequences), batch), desc="Embedding", unit="batch"):
        batch_seqs = sequences[i : i + batch]
        with torch.no_grad():
            t = model.transform({"sequences": batch_seqs})
            emb = model(t)  # (B, L, D)
        emb = emb.mean(dim=1).cpu().numpy()  # token-avg → (B, D)
        all_vec.append(emb)
    all_vec = np.vstack(all_vec).astype("float32")
    np.save(cache_file, all_vec)
    print(f"Saved embeddings to {cache_file}")
    return all_vec


# ─────────────────── CLI ─────────────────── #
def parse_args():
    p = argparse.ArgumentParser("Aido Contiguous-CV evaluator")
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="/data4/marunze/aido/cache/")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────── 主流程 ─────────────────── #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "correlation_summary.txt"
    if not summary_path.exists():
        summary_path.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    ref_df = pd.read_csv(args.ref_sheet)
    model = Embed.from_config({"model.backbone": "aido_rna_1b600m"})
    print("Model loaded.")

    for _, row in ref_df.iterrows():
        dms_id = row.get("DMS_ID", row.iloc[0])
        dms_path = Path(args.dms_dir) / f"{dms_id}.csv"
        if not dms_path.exists():
            print(f"[skip] {dms_path} absent")
            continue

        df = pd.read_csv(dms_path)
        if not {"mutant", "DMS_score", "sequence"}.issubset(df.columns):
            print(f"[skip] {dms_id}: missing columns")
            continue

        seqs = [preprocess_sequence(s) for s in df["sequence"]]
        scores = df["DMS_score"].astype(float).values
        pos_lists = df["mutant"].apply(parse_mutation).tolist()

        emb = embed_sequences(dms_id, seqs, model, args.device, args.cache_dir)
        # 清洗非法值
        mask = np.isfinite(emb).all(axis=1) & np.isfinite(scores)
        emb, scores = emb[mask], scores[mask]
        pos_lists = [pl for pl, m in zip(pos_lists, mask) if m]

        res = evaluate_contiguous(
            emb,
            scores,
            pos_lists,
            n_folds=args.n_folds,
            max_frac=args.max_test_frac,
        )
        print(
            f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}"
        )

        # 写带分数的 CSV
        df[f"{dms_id}_score"] = np.nan
        df.loc[mask, f"{dms_id}_score"] = res["preds"]
        df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary_path, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Done. Summary → {summary_path}")


if __name__ == "__main__":
    main()

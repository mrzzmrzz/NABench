#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_enformer_contiguous.py —— Enformer + Contiguous-CV 评测
2025-07-13
"""

import argparse, re, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
from enformer_pytorch import from_pretrained

# ─────────────────── 序列 & 突变工具 ─────────────────── #
def preprocess_seq(seq: str, max_len=196_608):
    seq = seq.upper().replace("U", "T")
    return (seq[:max_len]).ljust(max_len, "-")

def parse_mutation(mut_str: str):
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]

def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]

# ─────────────────── 评价：Contiguous-CV ─────────────────── #
def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    mdl = RidgeCV(alphas=alphas, store_cv_results=False)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)

def evaluate_contiguous(emb, scores, pos_lists, n_folds=5, max_frac=0.20, seed=0):
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
        train_idx = [i for i in range(N) if i not in candidate]   # 超量样本弃置
        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Warning: Fold {k + 1} has too few training samples ({len(train_idx)}). Skipping.")
            continue
        Xtr, ytr = emb[train_idx], scores[train_idx]
        Xte, yte = emb[test_idx], scores[test_idx]

        y_pred = ridge_pred(Xtr, ytr, Xte)
        preds_all[test_idx] = y_pred

        fold_corrs.append(spearmanr(y_pred, yte).correlation)
        fold_mses.append(mean_squared_error(yte, y_pred))
        print(f"Fold {k+1}: train={len(train_idx)}, test={len(test_idx)}, ρ={fold_corrs[-1]:.3f}")

    return {
        "corr_mean": np.nanmean(fold_corrs),
        "corr_std":  np.nanstd(fold_corrs),
        "mse_mean":  np.nanmean(fold_mses),
        "mse_std":   np.nanstd(fold_mses),
        "preds":     preds_all,
    }

# ─────────────────── 嵌入（logits）缓存 ─────────────────── #
def enformer_logits(dms_id, sequences, model, device, cache_dir, batch=4, max_len=196_608):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dms_id}_embeddings.npy"
    if cache_file.exists():
        print(f"Loading cached logits: {cache_file}")
        return np.load(cache_file)

    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4,'-':-1,'X':4}
    all_logits = []
    model.eval().to(device)

    for i in tqdm(range(0, len(sequences), batch), desc="Embedding", unit="batch"):
        batch_seqs = [preprocess_seq(s, max_len) for s in sequences[i:i+batch]]
        arr = np.array([[mapping[b] for b in seq] for seq in batch_seqs], dtype=np.int64)
        tokens = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            logits = model(tokens)["human"]          # (B, L, C)
            avg = logits.mean(dim=1).cpu().numpy()   # (B, C)
        all_logits.append(avg)

    all_logits = np.vstack(all_logits).astype("float32")
    np.save(cache_file, all_logits)
    print(f"Saved logits to {cache_file}")
    return all_logits

# ─────────────────── CLI ─────────────────── #
def parse_args():
    p = argparse.ArgumentParser("Enformer Contiguous-CV evaluator")
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="/data4/marunze/enformer/cache/")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ─────────────────── Main ─────────────────── #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "correlation_summary.txt"
    if not summary_path.exists():
        summary_path.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    print("Loading Enformer model...")
    model = from_pretrained("EleutherAI/enformer-official-rough")

    ref_df = pd.read_csv(args.ref_sheet)
    for _, row in ref_df.iterrows():
        dms_id = row.get("DMS_ID", row.iloc[0])
        dms_file = Path(args.dms_dir) / f"{dms_id}.csv"
        if not dms_file.exists():
            print(f"[skip] {dms_id}: file absent")
            continue

        df = pd.read_csv(dms_file)
        mut_col = next((c for c in ["mutant","mutation","mutations"] if c in df.columns), None)
        if mut_col is None or "DMS_score" not in df.columns or "sequence" not in df.columns:
            print(f"[skip] {dms_id}: missing columns")
            continue

        seqs   = df["sequence"].astype(str).tolist()
        scores = df["DMS_score"].astype(float).values
        pos_lists = df[mut_col].apply(parse_mutation).tolist()

        logits = enformer_logits(dms_id, seqs, model, args.device, args.cache_dir)
        mask = np.isfinite(logits).all(axis=1) & np.isfinite(scores)
        logits, scores = logits[mask], scores[mask]
        pos_lists = [pl for pl, m in zip(pos_lists, mask) if m]

        res = evaluate_contiguous(
            logits, scores, pos_lists,
            n_folds=args.n_folds, max_frac=args.max_test_frac
        )
        print(f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}")

        # 保存带分数的 CSV
        df[f"{dms_id}_score"] = np.nan
        df.loc[mask, f"{dms_id}_score"] = res["preds"]
        df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary_path, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Done. Summary → {summary_path}")

if __name__ == "__main__":
    main()

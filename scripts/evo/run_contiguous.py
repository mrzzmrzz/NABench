#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evo_contiguous.py —— Evo + Contiguous-CV 评测
2025-07-13
"""

import argparse, re, sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import torch
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs


# ──────────────── 工具函数 ──────────────── #
def preprocess(seq: str) -> str:
    return seq.upper().replace("U", "T")


def parse_mutation(mut_str: str):
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]


def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]


# ──────────────── 评估 ──────────────── #
def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    mdl = RidgeCV(alphas=alphas, store_cv_values=False)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)


def evaluate_contiguous(feat, y, pos_lists, n_folds=5, max_frac=0.20, seed=0):
    N = len(y)
    rng = np.random.RandomState(seed)
    preds_all = np.full(N, np.nan)

    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    fold_corrs, fold_mses = [], []
    is_vector = feat.ndim == 1

    for k, (lo, hi) in enumerate(intervals):
        cand = [i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)]
        if not cand:
            continue
        max_sz = int(max_frac * N)
        test_idx = rng.choice(cand, size=min(len(cand), max_sz), replace=False).tolist()
        train_idx = [i for i in range(N) if i not in cand]  # 超量样本弃置
        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Fold {k+1} skipped: train={len(train_idx)}, test={len(test_idx)}")
            continue
        if is_vector:
            y_pred = feat[test_idx]
        else:
            y_pred = ridge_pred(feat[train_idx,:], y[train_idx], feat[test_idx,:])
        preds_all[test_idx] = y_pred
        fold_corrs.append(spearmanr(y_pred, y[test_idx]).correlation)
        fold_mses.append(mean_squared_error(y[test_idx], y_pred))
        print(f"Fold {k+1}: train={len(train_idx)}, test={len(test_idx)}, ρ={fold_corrs[-1]:.3f}")

    return {
        "corr_mean": np.nanmean(fold_corrs),
        "corr_std":  np.nanstd(fold_corrs),
        "mse_mean":  np.nanmean(fold_mses),
        "mse_std":   np.nanstd(fold_mses),
        "preds":     preds_all,
    }


# ──────────────── Evo 推断（含缓存） ──────────────── #
def evo_logprobs(dms_id, seqs, model, tokenizer, device, cache_dir, batch=128):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    file_cache = cache_dir / f"{dms_id}_logprobs.npy"
    if file_cache.exists():
        print(f"Loading cached log-probs: {file_cache}")
        return np.load(file_cache)

    all_lp = []
    for i in tqdm(range(0, len(seqs), batch), desc=f"Scoring {dms_id}", unit="seq"):
        batch_seqs = seqs[i:i+batch]
        ids, _ = prepare_batch(batch_seqs, tokenizer, prepend_bos=True, device=device)
        with torch.no_grad():
            logits, *_ = model(ids)
        lp = logits_to_logprobs(logits, ids, trim_bos=True).float().cpu().numpy()  # (B,L)
        all_lp.append(lp)
    all_lp = np.concatenate(all_lp)
    np.save(file_cache, all_lp)
    return all_lp


# ──────────────── CLI ──────────────── #
def parse_args():
    p = argparse.ArgumentParser("Evo Contiguous-CV evaluator")
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="evo-1-8k-base")
    p.add_argument("--cache_dir", default="/data4/marunze/evo/cache/")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


# ──────────────── Main ──────────────── #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "correlation_summary.txt"
    if not summary.exists():
        summary.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    print("Loading Evo model...")
    evo = Evo(args.model_name)
    model, tokenizer = evo.model.to(args.device), evo.tokenizer
    model.eval()

    ref = pd.read_csv(args.ref_sheet)
    for _, row in ref.iterrows():
        dms_id = row["DMS_ID"]
        csv_file = Path(args.dms_dir) / f"{dms_id}.csv"
        if not csv_file.exists():
            print(f"[skip] {dms_id}: file absent")
            continue

        df = pd.read_csv(csv_file)
        if not {"sequence","DMS_score","mutant"}.issubset(df.columns):
            print(f"[skip] {dms_id}: missing columns")
            continue

        seqs   = df["sequence"].astype(str).map(preprocess).tolist()
        scores = df["DMS_score"].astype(float).values
        pos_ls = df["mutant"].apply(parse_mutation).tolist()

        logp = evo_logprobs(dms_id, seqs, model, tokenizer, args.device, args.cache_dir, args.batch_size)
        seq_scores = logp.astype("float32")  # (N,L) scalar feature

        mask = np.isfinite(seq_scores).all(axis=1) & np.isfinite(scores)
        seq_scores, scores = seq_scores[mask], scores[mask]
        pos_ls = [pl for pl,m in zip(pos_ls, mask) if m]

        res = evaluate_contiguous(seq_scores, scores, pos_ls,
                                  n_folds=args.n_folds, max_frac=args.max_test_frac)
        print(f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}")

        df[f"{args.model_name}_score"] = np.nan
        df.loc[mask, f"{args.model_name}_score"] = res["preds"]
        df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Finished. Summary → {summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rnafm_contiguous.py —— RNA-FM + Contiguous-CV 评测
2025-07-13
"""

import argparse, os, re, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# RNA-FM 依赖
def load_rnafm(model_dir):
    sys.path.insert(1, model_dir)
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    bc = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, bc


# ────────── 通用工具 ────────── #
def preprocess_seq(seq: str):
    return seq.upper().replace("T", "U")


def parse_mutation(mut_str: str):
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]


def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    splits = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in splits if len(b)]


# ────────── 评估：Contiguous-CV ────────── #
def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    md = RidgeCV(alphas=alphas, store_cv_values=False)
    md.fit(Xtr, ytr)
    return md.predict(Xte)


def evaluate_contiguous(features, y, pos_lists, n_folds=5, max_frac=0.20, seed=0):
    """
    features : (N,D) or (N,)  ——  2D = embeddings → RidgeCV; 1D = 已是打分
    """
    N = len(y)
    rng = np.random.RandomState(seed)
    preds_all = np.full(N, np.nan)

    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    fold_corrs, fold_mses = [], []
    is_vector = features.ndim == 1

    for k, (lo, hi) in enumerate(intervals):
        candidate = [i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)]
        if not candidate:
            continue
        max_sz = int(max_frac * N)
        test_idx = rng.choice(candidate, size=min(len(candidate), max_sz), replace=False).tolist()
        train_idx = [i for i in range(N) if i not in candidate]  # 超量样本弃置

        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Fold {k + 1} skipped (too few train samples)")
            continue

        if is_vector:
            y_pred = np.repeat(np.nan, len(test_idx))
            y_pred = features[test_idx]          # scalar predictions
        else:
            y_pred = ridge_pred(features[train_idx], y[train_idx], features[test_idx])

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


# ────────── 嵌入 & 缓存 ────────── #
def embed_rnafm(dms_id, sequences, model, batch_converter, device, cache_dir, batch_size=32):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    f_cache = cache_dir / f"{dms_id}_emb.npy"
    if f_cache.exists():
        print(f"Loading cached embeddings for {dms_id}")
        return np.load(f_cache)

    model = model.to(device).eval()
    all_vec = []
    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Embedding {dms_id}", unit="batch"):
        batch = [("seq", s) for s in sequences[i:i+batch_size]]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            rep = model(toks, repr_layers=[12])["representations"][12]  # (B,L,D)
            vec = rep.mean(dim=1).cpu().numpy()                        # token-avg
        all_vec.append(vec)
    all_vec = np.vstack(all_vec).astype("float32")
    np.save(f_cache, all_vec)
    return all_vec


# ────────── CLI ────────── #
def parse_args():
    p = argparse.ArgumentParser("RNA-FM Contiguous-CV evaluator")
    p.add_argument("--model_location", required=True)
    p.add_argument("--reference_sequences", required=True)
    p.add_argument("--dms_directory", required=True)
    p.add_argument("--output_directory", required=True)
    p.add_argument("--cache_dir", default="/data4/marunze/rnafm/cache/")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ────────── Main ────────── #
def main():
    args = parse_args()
    out_dir = Path(args.output_directory); out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "correlation_summary.txt"
    if not summary.exists():
        summary.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    print("Loading RNA-FM ...")
    model, alphabet, batch_converter = load_rnafm(args.model_location)

    ref = pd.read_csv(args.reference_sequences)
    ref["PATH"] = ref["DMS_ID"].apply(lambda x: f"{args.dms_directory}/{x}.csv")

    for _, row in ref.iterrows():
        dms_id   = row["DMS_ID"]
        csv_file = Path(row["PATH"])
        if not csv_file.exists():
            print(f"[skip] {dms_id}: file absent")
            continue

        # 读取并生成突变序列
        mut_df = pd.read_csv(csv_file)
        mcol = next((c for c in ["mutant","mutation","mutations"] if c in mut_df.columns), None)
        if mcol is None or "DMS_score" not in mut_df.columns:
            print(f"[skip] {dms_id}: missing columns")
            continue

        # 若 data 中已有 'sequence' 列请直接用；否则按 WT + mutations 生成
        if "sequence" not in mut_df.columns:
            wt_seq = preprocess_seq(row["RAW_CONSTRUCT_SEQ"])
            mut_df["sequence"] = mut_df[mcol].apply(lambda mut: apply_mutations(wt_seq, mut))  # 需要自行实现 apply_mutations

        seqs   = mut_df["sequence"].astype(str).map(preprocess_seq).tolist()
        scores = mut_df["DMS_score"].astype(float).values
        pos_ls = mut_df[mcol].apply(parse_mutation).tolist()

        emb = embed_rnafm(dms_id, seqs, model, batch_converter, args.device, args.cache_dir)
        mask = np.isfinite(emb).all(axis=1) & np.isfinite(scores)
        emb, scores = emb[mask], scores[mask]
        pos_ls = [pl for pl,m in zip(pos_ls, mask) if m]

        res = evaluate_contiguous(
            emb, scores, pos_ls,
            n_folds=args.n_folds, max_frac=args.max_test_frac
        )
        print(f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}")

        mut_df[f"{dms_id}_score"] = np.nan
        mut_df.loc[mask, f"{dms_id}_score"] = res["preds"]
        mut_df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Finished. Summary → {summary}")


if __name__ == "__main__":
    main()

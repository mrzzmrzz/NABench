#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_space_contiguous.py —— SPACE + Contiguous-CV 评测
2025-07-13
"""

import argparse, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F

# 加载 SPACE
sys.path.append("./SPACE")
from SPACE.model.modeling_space import Space, SpaceConfig


# ──────────────────── 工具 ──────────────────── #
def preprocess(seq: str, L=131_072):
    seq = seq.upper().replace("U", "T")
    return seq[:L].ljust(L, "-")


def parse_mutation(mut_str: str):
    if pd.isna(mut_str) or not mut_str.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut_str)]


def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]


# ──────────────────── 评估 ──────────────────── #
def ridge_pred(Xtr, ytr, Xte, alphas=np.logspace(-3, 3, 7)):
    mdl = RidgeCV(alphas=alphas, store_cv_results=False)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)


def evaluate_contiguous(feat, y, pos_lists, n_folds=5, max_frac=0.20, seed=0):
    N = len(y)
    rng = np.random.RandomState(seed)
    preds_all = np.full(N, np.nan)

    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    fold_corrs, fold_mses = [], []
    for k, (lo, hi) in enumerate(intervals):
        cand = [i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)]
        if not cand:
            continue
        max_sz = int(max_frac * N)
        test_idx = rng.choice(cand, size=min(len(cand), max_sz), replace=False).tolist()
        train_idx = [i for i in range(N) if i not in cand]

        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Fold {k+1} skipped: train={len(train_idx)}, test={len(test_idx)}")
            continue
        y_pred = ridge_pred(feat[train_idx], y[train_idx], feat[test_idx])
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


# ──────────────────── SPACE 嵌入 & 缓存 ──────────────────── #
_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': -1, 'X': 4}

def space_embed(dms_id, seqs, model, device, cache_dir, batch=1, L=131_072):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    f_cache = cache_dir / f"{dms_id}_space_emb.npy"
    if f_cache.exists():
        print(f"Loading cached embeddings: {f_cache}")
        return np.load(f_cache)

    model.eval()
    all_vec = []
    for s in tqdm(seqs, desc=f"SPACE {dms_id}", unit="seq"):
        seq = preprocess(s, L=L)
        tok = torch.tensor([_mapping[b] for b in seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(tok)["out"]            # (L,C)
            logp   = F.log_softmax(logits, dim=-1)
            vec    = logp.mean(dim=0).cpu().numpy()   # (C,)
        all_vec.append(vec)
    all_vec = np.vstack(all_vec).astype("float32")
    np.save(f_cache, all_vec)
    return all_vec


# ──────────────────── CLI ──────────────────── #
def parse_args():
    p = argparse.ArgumentParser("SPACE Contiguous-CV evaluator")
    p.add_argument("--reference_sheet", required=True)
    p.add_argument("--dms_directory", required=True)
    p.add_argument("--output_directory", required=True)
    p.add_argument("--cache_dir", default="/data4/marunze/space/cache/")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ──────────────────── Main ──────────────────── #
def main():
    args = parse_args()
    out_dir = Path(args.output_directory); out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "correlation_summary.txt"
    if not summary.exists():
        summary.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    print("Loading SPACE backbone ...")
    cfg = SpaceConfig.from_pretrained("yangyz1230/space")
    cfg.input_file = ""
    model = Space.from_pretrained("yangyz1230/space", config=cfg).to(args.device)

    ref = pd.read_csv(args.reference_sheet)
    for _, row in ref.iterrows():
        dms_id = row["DMS_ID"]
        csv_file = Path(args.dms_directory) / f"{dms_id}.csv"
        if not csv_file.exists():
            print(f"[skip] {dms_id}: file absent")
            continue

        df = pd.read_csv(csv_file)
        mcol = next((c for c in ["mutant","mutation","mutations"] if c in df.columns), None)
        if mcol is None or "DMS_score" not in df.columns or "sequence" not in df.columns:
            print(f"[skip] {dms_id}: missing columns")
            continue

        seqs   = df["sequence"].astype(str).tolist()
        scores = df["DMS_score"].astype(float).values
        pos_ls = df[mcol].apply(parse_mutation).tolist()

        emb = space_embed(dms_id, seqs, model, args.device, args.cache_dir)
        mask = np.isfinite(emb).all(axis=1) & np.isfinite(scores)
        emb, scores = emb[mask], scores[mask]
        pos_ls = [pl for pl,m in zip(pos_ls, mask) if m]

        res = evaluate_contiguous(emb, scores, pos_ls,
                                  n_folds=args.n_folds, max_frac=args.max_test_frac)
        print(f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}")

        df[f"SPACE_score"] = np.nan
        df.loc[mask, "SPACE_score"] = res["preds"]
        df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Done. Summary → {summary}")


if __name__ == "__main__":
    main()

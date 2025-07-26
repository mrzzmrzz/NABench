#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generrna_contiguous.py — GenerRNA + Contiguous-CV 评测
2025-07-13
"""
import argparse, re, sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

import torch
import sys
sys.path.append("/data4/marunze/generrna/")
from GenerRNA.model import GPT, GPTConfig
from transformers import AutoTokenizer


# ────────────────── 工具 ────────────────── #
def preprocess(seq: str) -> str:
    return seq.strip().upper()


def parse_mut(mut: str):
    if pd.isna(mut) or not mut.strip():
        return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut)]


def contiguous_intervals(all_pos, k=5):
    arr = np.array(sorted(all_pos))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(b[0], b[-1]) for b in blocks if len(b)]


# ────────────────── 评估 ────────────────── #
def ridge_pred(Xtr, ytr, Xte):
    model = RidgeCV(alphas=np.logspace(-3, 3, 7))
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def evaluate_contiguous(feat, y, pos_lists, n_folds=5, max_frac=0.20, seed=0):
    N = len(y)
    rng = np.random.RandomState(seed)
    preds = np.full(N, np.nan)

    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    corrs, mses = [], []

    for k, (lo, hi) in enumerate(intervals):
        cand = [i for i, pl in enumerate(pos_lists) if any(lo <= p <= hi for p in pl)]
        if not cand:
            continue
        M = int(max_frac * N)
        test_idx = rng.choice(cand, size=min(len(cand), M), replace=False).tolist()
        train_idx = [i for i in range(N) if i not in cand]         # 全排除
        if len(train_idx) < 0.4 * N and len(train_idx) < 500:
            print(f"Fold {k+1} skipped: train={len(train_idx)}, test={len(test_idx)}")
            continue
        y_pred = ridge_pred(feat[train_idx, :], y[train_idx], feat[test_idx, :])
        preds[test_idx] = y_pred
        corrs.append(spearmanr(y_pred, y[test_idx]).correlation)
        mses.append(mean_squared_error(y[test_idx], y_pred))
        print(f"Fold {k+1}: train={len(train_idx)}  test={len(test_idx)}  ρ={corrs[-1]:.3f}")

    return {
        "corr_mean": np.nanmean(corrs),
        "corr_std":  np.nanstd(corrs),
        "preds":     preds,
    }


# ────────────────── GenerRNA 推断（缓存） ────────────────── #
from pathlib import Path
import numpy as np, torch
from tqdm.auto import tqdm

def _cache_path(cache_dir, did, kind):
    return Path(cache_dir)/f"{did}_{kind}.npy"


@torch.no_grad()
def _seq_hidden(model, toks, device, pool="mean"):
    """
    返回一条序列的池化隐向量
    pool: 'mean' | 'cls' | 'sum'
    """
    x = torch.tensor(toks, device=device)[None]          # (1,L)
    # GenerRNA GPT.forward(*) 默认返回 logits，如需 hidden_state
    # 假设 forward(..., return_h=True) -> (logits, h)  或者 model.get_hidden(x)
    # ↓ 根据你的模型接口调整 ↓
    logits, _, h = model(x, return_hidden=True)                 # h:(1,L,D)

    h = h.squeeze(0)                                    # (L,D)
    vec =h                                 # GPT 形式 BOS token
    return vec.cpu().numpy()                            # (D,)


def gener_hidden_features(
        did, seqs, model, tokenizer, device,
        cache_dir="/data4/marunze/generrna/cache",
        pool="mean") -> np.ndarray:
    """
    对整个数据集抽取隐藏层特征并缓存
    """
    cache_file = _cache_path(cache_dir, did, f"h_{pool}")
    if cache_file.exists():
        print(f"[cache] load {cache_file}")
        return np.load(cache_file)

    model.eval()
    feats = []
    for s in tqdm(seqs, desc=f"GenerRNA-hid {did}", unit="seq"):
        toks = tokenizer.encode(s)
        feats.append(_seq_hidden(model, toks, device, pool))
    feats = np.vstack(feats).astype("float32")
    np.save(cache_file, feats)
    return feats

# ────────────────── CLI ────────────────── #
def cli():
    p = argparse.ArgumentParser("GenerRNA Contiguous-CV evaluator")
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--tokenizer_dir", default="GenerRNA/tokenizer")
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="/data4/marunze/generrna/cache/")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_test_frac", type=float, default=0.20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ────────────────── Main ────────────────── #
def main():
    args = cli()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "correlation_summary.txt"
    if not summary.exists():
        summary.write_text("DMS_ID,Spearman_mean,Spearman_std\n")

    print("Loading GenerRNA ...")
    ckpt = torch.load(args.model_ckpt, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.replace('_orig_mod.', ''): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state); model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    ref = pd.read_csv(args.ref_sheet)
    for _, row in ref.iterrows():
        dms_id = row["DMS_ID"]
        wt_seq = preprocess(row["RAW_CONSTRUCT_SEQ"])
        dms_file = Path(args.dms_dir) / f"{dms_id}.csv"
        if not dms_file.exists():
            print(f"[skip] {dms_id}: file absent"); continue

        df = pd.read_csv(dms_file)
        mcol = next((c for c in ["mutant","mutation","mutations"] if c in df.columns), None)
        if mcol is None or "DMS_score" not in df.columns:
            print(f"[skip] {dms_id}: missing columns"); continue

        if "mutated_sequence" in df.columns:
            seqs = df["mutated_sequence"].astype(str).map(preprocess).tolist()
        else:
            seqs = df["sequence"].astype(str).map(preprocess).tolist()

        scores_exp = df["DMS_score"].astype(float).values
        pos_lists  = df[mcol].apply(parse_mut).tolist()

        feat = gener_hidden_features(
            dms_id, seqs,
            model, tokenizer,
            "cuda",
            cache_dir=args.cache_dir,
            pool="mean"          # 或 "cls"、"sum"
)

        mask = np.isfinite(feat).all(axis=1) & np.isfinite(scores_exp)
        feat, scores_exp = feat[mask], scores_exp[mask]
        pos_lists = [pl for pl, m in zip(pos_lists, mask) if m]

        res = evaluate_contiguous(feat, scores_exp, pos_lists,
                                  n_folds=args.n_folds, max_frac=args.max_test_frac)
        print(f"[{dms_id}] ρ={res['corr_mean']:.3f} ± {res['corr_std']:.3f}")

        df[f"GenerRNA_score"] = np.nan
        df.loc[mask, "GenerRNA_score"] = res["preds"]
        df.to_csv(out_dir / f"{dms_id}.csv", index=False)

        with open(summary, "a") as f:
            f.write(f"{dms_id},{res['corr_mean']:.3f},{res['corr_std']:.3f}\n")

    print(f"Finished. Summary → {summary}")


if __name__ == "__main__":
    main()

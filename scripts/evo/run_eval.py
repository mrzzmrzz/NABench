#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evo_eval.py  ——  Evo 模型  Random-CV / Contiguous-CV / Few-shot  统一评测脚本
2025-07-13
"""

import argparse, re, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import torch
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs

# ───────────────────────── 通用工具 ───────────────────────── #
def preprocess(seq: str) -> str:
    return seq.upper().replace("U", "T")

def parse_mut(mut: str):
    if pd.isna(mut) or not mut: return []
    return [int(m.group()) for m in re.finditer(r"\d+", mut)]

def contiguous_intervals(pos_set, k=5):
    arr = np.array(sorted(pos_set))
    blocks = np.array_split(arr, min(k, len(arr)))
    return [(blk[0], blk[-1]) for blk in blocks if len(blk)]

def ridge_pred(Xtr, ytr, Xte):
    mdl = RidgeCV(alphas=np.logspace(-3,3,7), store_cv_values=False)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)

# ───────────────────────── 三种评测 ───────────────────────── #
def eval_contiguous(X, y, pos_lists, n_folds=5, max_frac=0.2, seed=0):
    N = len(y); rng = np.random.RandomState(seed)
    intervals = contiguous_intervals({p for pl in pos_lists for p in pl}, n_folds)
    preds = np.full(N, np.nan); corrs=[]; mses=[]
    for (lo,hi) in intervals:
        cand=[i for i,pl in enumerate(pos_lists) if any(lo<=p<=hi for p in pl)]
        if not cand: continue
        test_idx=rng.choice(cand,size=min(len(cand),int(max_frac*N)),replace=False)
        train_idx=[i for i in range(N) if i not in cand]
        if len(train_idx) < 0.4 * N and len(train_idx) < 400:
            print(f"Skipping fold with too few training samples: {len(train_idx)}")
            continue
        y_pred=ridge_pred(X[train_idx],y[train_idx],X[test_idx])
        preds[test_idx]=y_pred
        corrs.append(spearmanr(y_pred,y[test_idx]).correlation)
        mses.append(mean_squared_error(y[test_idx],y_pred))
    return np.nanmean(corrs),np.nanstd(corrs),preds

def eval_random(X, y, n_folds=5, seed=0):
    N=len(y); rng=np.random.RandomState(seed)
    idx=np.arange(N); rng.shuffle(idx)
    fold_ids=np.array_split(idx,n_folds)
    preds=np.full(N,np.nan); corrs=[]
    for test_idx in fold_ids:
        train_idx=np.setdiff1d(idx,test_idx)
        y_pred=ridge_pred(X[train_idx,:],y[train_idx],X[test_idx,:])
        preds[test_idx]=y_pred
        corrs.append(spearmanr(y_pred,y[test_idx]).correlation)
    return np.nanmean(corrs),np.nanstd(corrs),preds

def eval_fewshot(X,y,k,repeat=5,seed=0):
    if k > len(y):
        print(f"Warning: few_shot_k ({k}) is larger than available samples ({len(y)}), skipping few-shot evaluation")
        return None, None, None
    rng=np.random.RandomState(seed)
    corrs=[]; best_model=None
    for _ in range(repeat):
        idx=rng.choice(len(y),size=k,replace=False)
        model=Ridge(alpha=1.0).fit(X[idx],y[idx])
        y_pred=model.predict(np.delete(X,idx,axis=0))
        corr=spearmanr(y_pred,np.delete(y,idx)).correlation
        corrs.append(corr)
        if best_model is None or corr>np.max(corrs): best_model=model
    full_pred=best_model.predict(X)
    return np.mean(corrs),np.std(corrs),full_pred

# ───────────────────────── Evo 推断&缓存 ───────────────────────── #
def evo_logp(did, seqs, model, tok, device, cache_dir, batch=128):
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    f=cache_dir/f"{did}_logprobs.npy"
    if f.exists(): return np.load(f)
    out=[]
    for i in tqdm(range(0,len(seqs),batch),desc=f"Evo {did}",unit="seq"):
        ids,_=prepare_batch(seqs[i:i+batch],tok,prepend_bos=True,device=device)
        with torch.no_grad(): logits,_=model(ids)
        lp=logits_to_logprobs(logits,ids,trim_bos=True).cpu().numpy()
        out.append(lp)
    arr=np.concatenate(out); np.save(f,arr); return arr

# ───────────────────────── CLI ───────────────────────── #
parser=argparse.ArgumentParser()
parser.add_argument("--ref_sheet",required=True)
parser.add_argument("--dms_dir",required=True)
parser.add_argument("--output_dir",required=True)
parser.add_argument("--model_name",default="evo-1-8k-base")
parser.add_argument("--scheme",choices=["contiguous","random","few-shot"],default="contiguous")
parser.add_argument("--n_folds",type=int,default=5)
parser.add_argument("--max_test_frac",type=float,default=0.2)
parser.add_argument("--few_shot_k",type=int,default=100)
parser.add_argument("--few_shot_repeat",type=int,default=5)
parser.add_argument("--cache_dir",default="/data3/marunze/evo/cache/")
parser.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size",type=int,default=128)
args=parser.parse_args()

# ───────────────────────── 主流程 ───────────────────────── #
out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
summary=out_dir/"correlation_summary.txt"
if not summary.exists(): summary.write_text("DMS_ID,scheme,Spearman_mean,Spearman_std\n")

print("Loading Evo …")
evo=Evo(args.model_name)
model,tokenizer=evo.model.to(args.device),evo.tokenizer
model.eval()

ref=pd.read_csv(args.ref_sheet)
for _,row in ref.iterrows():
    did=row["DMS_ID"]; csv=Path(args.dms_dir)/f"{did}.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv)
    if not {"sequence","DMS_score","mutant"}.issubset(df.columns): continue
    seqs=df["sequence"].astype(str).map(preprocess).tolist()
    scores=df["DMS_score"].astype(float).values
    pos_lists=df["mutant"].apply(parse_mut).tolist()

    logp=evo_logp(did,seqs,model,tokenizer,args.device,args.cache_dir,args.batch_size)
    feat=logp  # (N,L)  -> Ridge 接口仍接受

    if args.scheme=="contiguous":
        mu,std,pred=eval_contiguous(feat,scores,pos_lists,
                                    n_folds=args.n_folds,max_frac=args.max_test_frac)
    elif args.scheme=="random":
        mu,std,pred=eval_random(feat,scores,n_folds=args.n_folds)
    else:  # few-shot
        k=min(args.few_shot_k,len(scores)-1)
        mu,std,pred=eval_fewshot(feat,scores,k,args.few_shot_repeat)

    print(f"[{did}] {args.scheme} ρ={mu:.3f}±{std:.3f}")
    df[f"{args.model_name}_{args.scheme}_score"]=pred
    df.to_csv(out_dir/f"{did}.csv",index=False)

    with open(summary,"a") as f:
        f.write(f"{did},{args.scheme},{mu:.3f},{std:.3f}\n")

print("Done →",summary)

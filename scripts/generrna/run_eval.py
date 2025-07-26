#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generrna_eval.py — GenerRNA  Random-CV / Contiguous-CV / Few-shot 评测
"""

import argparse, re, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import torch, torch.nn.functional as F
sys.path.append("/data4/marunze/generrna/")
from GenerRNA.model import GPT, GPTConfig
from transformers import AutoTokenizer

# ───────────────────── 通用工具 ────────────────────── #
def preprocess(s:str)->str: return s.strip().upper()
def parse_mut(m:str): return [] if pd.isna(m) or not m.strip() else [int(x.group()) for x in re.finditer(r"\d+",m)]
def contiguous_intervals(pos,k=5):
    arr=np.array(sorted(pos)); blocks=np.array_split(arr,min(k,len(arr)))
    return [(b[0],b[-1]) for b in blocks if len(b)]

def ridge_pred(Xtr,ytr,Xte):
    mdl=RidgeCV(alphas=np.logspace(-3,3,7)); mdl.fit(Xtr,ytr); return mdl.predict(Xte)

# ─────────── 三种评测策略 ─────────── #
def eval_contiguous(feat,y,pos,k=5,max_frac=.2,seed=0):
    N=len(y); rng=np.random.RandomState(seed); pr=np.full(N,np.nan)
    ints=contiguous_intervals({p for pl in pos for p in pl},k); corrs=[]
    for lo,hi in ints:
        cand=[i for i,pl in enumerate(pos) if any(lo<=p<=hi for p in pl)]
        if not cand: continue
        test=rng.choice(cand,size=min(len(cand),int(max_frac*N)),replace=False)
        train=[i for i in range(N) if i not in cand]
        if len(train)<max(0.4*N,500): continue
        pr[test]=ridge_pred(feat[train,None],y[train],feat[test,None])
        corrs.append(spearmanr(pr[test],y[test]).correlation)
    return np.nanmean(corrs),np.nanstd(corrs),pr

def eval_random(feat,y,k=5,seed=0):
    N=len(y); rng=np.random.RandomState(seed); idx=rng.permutation(N); pr=np.full(N,np.nan); corrs=[]
    folds=np.array_split(idx,k)
    for test in folds:
        train=np.setdiff1d(idx,test)
        pr[test]=ridge_pred(feat[train,:],y[train],feat[test,:])
        corrs.append(spearmanr(pr[test],y[test]).correlation)
    return np.mean(corrs),np.std(corrs),pr

def eval_fewshot(feat,y,k,repeat=5,seed=0):
    rng=np.random.RandomState(seed); corrs=[]; best=None
    for _ in range(repeat):
        idx=rng.choice(len(y),size=k,replace=False)
        mdl=Ridge(alpha=1.).fit(feat[idx,:],y[idx])
        pr=mdl.predict(feat[np.setdiff1d(np.arange(len(y)),idx),:])
        corrs.append(spearmanr(pr,y[np.setdiff1d(np.arange(len(y)),idx)]).correlation)
        if best is None or corrs[-1]>np.mean(corrs): best=mdl
    full=best.predict(feat[:,:])
    return np.mean(corrs),np.std(corrs),full

# ─────────── GenerRNA 打分缓存 ─────────── #
# ─── utils_hid.py ──────────────────────────────────────────
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

# ─────────── CLI ─────────── #
def get_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_ckpt",required=True)
    ap.add_argument("--tokenizer_dir",required=True)
    ap.add_argument("--ref_sheet",required=True)
    ap.add_argument("--dms_dir",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--cache_dir",default="/data4/marunze/generrna/cache")
    ap.add_argument("--scheme",choices=["contiguous","random","few-shot"],default="contiguous")
    ap.add_argument("--n_folds",type=int,default=5)
    ap.add_argument("--max_test_frac",type=float,default=.2)
    ap.add_argument("--few_shot_k",type=int,default=100)
    ap.add_argument("--few_shot_repeat",type=int,default=5)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

# ─────────── Main ─────────── #
def main():
    A=get_args(); out=Path(A.output_dir); out.mkdir(parents=True,exist_ok=True)
    summ=out/"correlation_summary.txt"
    if not summ.exists(): summ.write_text("DMS_ID,scheme,Spearman_mean,Spearman_std\n")

    print("Loading GenerRNA…")
    ck=torch.load(A.model_ckpt,map_location=A.device)
    model=GPT(GPTConfig(**ck["model_args"])); model.load_state_dict({k.replace('_orig_mod.',''):v for k,v in ck["model"].items()})
    model.eval().to(A.device)
    tokenizer=AutoTokenizer.from_pretrained(A.tokenizer_dir)

    ref=pd.read_csv(A.ref_sheet)
    for _,row in ref.iterrows():
        did=row["DMS_ID"]; wt=preprocess(row["RAW_CONSTRUCT_SEQ"])
        csv=Path(A.dms_dir)/f"{did}.csv"
        if not csv.exists(): continue
        df=pd.read_csv(csv)
        mcol=next((c for c in ["mutant","mutation","mutations"] if c in df.columns),None)
        if mcol is None or "DMS_score" not in df.columns: continue
        seqs=(df["mutated_sequence"] if "mutated_sequence" in df.columns else df["sequence"]).astype(str).map(preprocess).tolist()
        y=df["DMS_score"].astype(float).values
        pos=df[mcol].apply(parse_mut).tolist()

        feat = gener_hidden_features(
            did, seqs,
            model, tokenizer,
            "cuda",
            cache_dir=A.cache_dir,
            pool="mean"          # 或 "cls"、"sum"
)
        mask = np.isfinite(feat).all(axis=1) & np.isfinite(y)
        feat,y,pos=feat[mask],y[mask],[pl for pl,m in zip(pos,mask) if m]
        print("shape of feat:", feat.shape, "y:", y.shape, "pos:", len(pos))
        if A.scheme=="contiguous":
            mu,std,pred=eval_contiguous(feat,y,pos,A.n_folds,A.max_test_frac)
        elif A.scheme=="random":
            mu,std,pred=eval_random(feat,y,A.n_folds)
        else:
            k=min(A.few_shot_k,len(y)-1)
            mu,std,pred=eval_fewshot(feat,y,k,A.few_shot_repeat)

        print(f"[{did}] {A.scheme} ρ={mu:.3f}±{std:.3f}")
        df[f"GenerRNA_{A.scheme}_score"]=np.nan
        df.loc[mask,f"GenerRNA_{A.scheme}_score"]=pred
        df.to_csv(out/f"{did}.csv",index=False)
        with open(summ,"a") as f: f.write(f"{did},{A.scheme},{mu:.3f},{std:.3f}\n")

    print("All done →",summ)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ernie_eval.py —— Paddle-Ernie  Random / Contiguous / Few-shot 评测
2025-07-13
"""
import argparse, re, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import torch, torch.nn.functional as F
from paddlenlp.transformers import ErnieForMaskedLM
import paddle
from src.rna_ernie import BatchConverter   # 你自己的 BatchConverter

# ──────────────────── 工具 ──────────────────── #
def preprocess(s:str)->str: return s.upper().replace("U","T")
def parse_mut(m:str): return [] if pd.isna(m) or not m.strip() else [int(x.group()) for x in re.finditer(r"\d+",m)]
def contiguous_intervals(pos,k=5):
    arr=np.array(sorted(pos)); blocks=np.array_split(arr,min(k,len(arr)))
    return [(b[0],b[-1]) for b in blocks if len(b)]
def ridge_pred(Xtr,ytr,Xte): 
    mdl=RidgeCV(alphas=np.logspace(-3,3,7)); mdl.fit(Xtr,ytr); return mdl.predict(Xte)

# ──────────── 三种评测 ──────────── #
def eval_contig(feat,y,pos,k=5,max_frac=.2,seed=0):
    N=len(y); rng=np.random.RandomState(seed); pr=np.full(N,np.nan); corrs=[]
    for lo,hi in contiguous_intervals({p for pl in pos for p in pl},k):
        cand=[i for i,pl in enumerate(pos) if any(lo<=p<=hi for p in pl)]
        if not cand: continue
        test=rng.choice(cand,size=min(len(cand),int(max_frac*N)),replace=False)
        train=[i for i in range(N) if i not in cand]
        if len(train)<max(0.4*N,500): continue
        pr[test]=ridge_pred(feat[train],y[train],feat[test])
        corrs.append(spearmanr(pr[test],y[test]).correlation)
    return np.nanmean(corrs),np.nanstd(corrs),pr

def eval_random(feat,y,k=5,seed=0):
    N=len(y); rng=np.random.RandomState(seed); idx=rng.permutation(N)
    pr=np.full(N,np.nan); corrs=[]
    for fold in np.array_split(idx,k):
        train=np.setdiff1d(idx,fold)
        pr[fold]=ridge_pred(feat[train],y[train],feat[fold])
        corrs.append(spearmanr(pr[fold],y[fold]).correlation)
    return np.mean(corrs),np.std(corrs),pr

def eval_fewshot(feat,y,k,repeat=5,seed=0):
    rng=np.random.RandomState(seed); corrs=[]; best=None
    for _ in range(repeat):
        idx=rng.choice(len(y),size=k,replace=False)
        mdl=Ridge(alpha=1.).fit(feat[idx],y[idx])
        pr=mdl.predict(np.delete(feat,idx,axis=0))
        corrs.append(spearmanr(pr,np.delete(y,idx)).correlation)
        if best is None or corrs[-1]>np.mean(corrs): best=mdl
    full=best.predict(feat)
    return np.mean(corrs),np.std(corrs),full

# ──────────── 抽取嵌入 & 缓存──────────── #
def ernie_embed(did,seqs,model,batch_converter,cache_dir,batch=256):
    cache=Path(cache_dir); cache.mkdir(parents=True,exist_ok=True)
    f=cache/f"{did}_ernie_emb.npy"
    if f.exists(): return np.load(f)
    vec=[]
    for i in tqdm(range(0,len(seqs),batch),desc=f"Ernie {did}",unit="batch"):
        batch=[("seq",s) for s in seqs[i:i+batch]]
        for _,_,ids in batch_converter(batch):
            ids=ids.to("gpu")
            with paddle.no_grad(): logits=model(ids).detach()
            logits=torch.tensor(logits.numpy()).mean(dim=1).cpu().numpy()
            vec.append(logits)
    arr=np.vstack(vec).astype("float32"); np.save(f,arr); return arr

# ──────────── CLI ──────────── #
def get_args():
    ap=argparse.ArgumentParser("Ernie evaluator")
    ap.add_argument("--model_checkpoint",required=True)
    ap.add_argument("--vocab_path",required=True)
    ap.add_argument("--reference_sequences",required=True)
    ap.add_argument("--dms_directory",required=True)
    ap.add_argument("--output_directory",required=True)
    ap.add_argument("--cache_dir",default="/data4/marunze/ernie/cache/")
    ap.add_argument("--scheme",choices=["contiguous","random","few-shot"],default="contiguous")
    ap.add_argument("--n_folds",type=int,default=5)
    ap.add_argument("--max_test_frac",type=float,default=.2)
    ap.add_argument("--few_shot_k",type=int,default=100)
    ap.add_argument("--few_shot_repeat",type=int,default=5)
    ap.add_argument("--device",default="gpu" if paddle.is_compiled_with_cuda() else "cpu")
    return ap.parse_args()

# ──────────── Main ──────────── #
def main():
    A=get_args(); out=Path(A.output_directory); out.mkdir(parents=True,exist_ok=True)
    summ=out/"correlation_summary.txt"
    if not summ.exists(): summ.write_text("DMS_ID,scheme,Spear_mean,Spear_std\n")

    paddle.set_device(A.device); print("Loading Ernie ...")
    ernie=ErnieForMaskedLM.from_pretrained(A.model_checkpoint)
    bc=BatchConverter(k_mer=1,vocab_path=A.vocab_path,batch_size=256,max_seq_len=512)

    ref=pd.read_csv(A.reference_sequences)
    ref["PATH"]=ref["DMS_ID"].apply(lambda x:f"{A.dms_directory}/{x}.csv")

    for _,row in ref.iterrows():
        did=row["DMS_ID"]; wt=preprocess(row["RAW_CONSTRUCT_SEQ"])
        if len(wt)>512: continue
        csv=Path(row["PATH"]); 
        if not csv.exists(): continue
        df=pd.read_csv(csv)
        mcol=next((c for c in ["mutant","mutation","mutations"] if c in df.columns),None)
        if mcol is None or "DMS_score" not in df.columns or "sequence" not in df.columns: continue

        seqs=df["sequence"].astype(str).map(preprocess).tolist()
        y=df["DMS_score"].astype(float).values
        pos=df[mcol].apply(parse_mut).tolist()
        feat=ernie_embed(did,seqs,ernie,bc,A.cache_dir)

        mask=np.isfinite(feat).all(axis=1)&np.isfinite(y)
        feat,y,pos=feat[mask],y[mask],[pl for pl,m in zip(pos,mask) if m]

        if A.scheme=="contiguous":
            mu,std,pred=eval_contig(feat,y,pos,A.n_folds,A.max_test_frac)
        elif A.scheme=="random":
            mu,std,pred=eval_random(feat,y,A.n_folds)
        else:
            k=min(A.few_shot_k,len(y)-1)
            mu,std,pred=eval_fewshot(feat,y,k,A.few_shot_repeat)

        print(f"[{did}] {A.scheme} ρ={mu:.3f}±{std:.3f}")
        df[f"Ernie_{A.scheme}_score"]=np.nan
        df.loc[mask,f"Ernie_{A.scheme}_score"]=pred
        df.to_csv(out/f"{did}.csv",index=False)
        with open(summ,"a") as f: f.write(f"{did},{A.scheme},{mu:.3f},{std:.3f}\n")

    print("Done →",summ)

if __name__=="__main__":
    main()

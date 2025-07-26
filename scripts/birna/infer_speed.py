#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for BiRNA-BERT on 100-length sequences.
Metrics:
  • total wall‐time
  • throughput (seq/s)
  • avg latency (ms/seq)
"""
import argparse, random, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig

DNA = "ACGT"
AA  = "ACDEFGHIKLMNPQRSTVWY"

def rand_seq(n: int, alphabet: str) -> str:
    return "".join(random.choices(alphabet, k=n))

def main() -> None:
    p = argparse.ArgumentParser("BiRNA-BERT speed benchmark (100 nt/aa)")
    p.add_argument("--checkpoint", default="buetnlpbio/birna-bert",
                   help="HF model checkpoint (default: buetnlpbio/birna-bert)")
    p.add_argument("--seq_type", choices=["gene", "prot"], default="gene")
    p.add_argument("--n_seq", type=int, default=1024, help="# sequences (default 1k)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = p.parse_args()

    # ── model & tokenizer ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")
    config    = BertConfig.from_pretrained(args.checkpoint)
    model     = AutoModelForMaskedLM.from_pretrained(args.checkpoint,
                                                     config=config,
                                                     trust_remote_code=True)
    # 去掉 MLM 头，仅取最后隐藏层
    model.cls = torch.nn.Identity()
    model.to(device).eval()

    # ── synthetic data ─────────────────────────────────────────────────
    alphabet = DNA if args.seq_type == "gene" else AA
    seqs = [rand_seq(args.seq_len, alphabet) for _ in range(args.n_seq)]

    # ── timed inference ────────────────────────────────────────────────
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i : i + args.batch_size]
            tok   = tokenizer(batch,
                              padding=True,
                              truncation=True,
                              max_length=args.seq_len,
                              return_tensors="pt").to(device)
            # logits shape: [B, L, hidden]; 取 token dim 均值即 [B, hidden]
            _ = model(**tok).logits.mean(dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - start

    # ── report ─────────────────────────────────────────────────────────
    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {t:.2f} s")
    print(f"Throughput      : {args.n_seq / t:.1f} seq/s")
    print(f"Avg. latency    : {t / args.n_seq * 1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"BiRNA-BERT {args.seq_type} {args.n_seq} {t:.2f} {args.batch_size} {t / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

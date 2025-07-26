#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for RinAlMo (e.g., giga-v1) on 100-nt RNA sequences.

Reports:
  • total wall-time
  • throughput (seq/s)
  • avg latency (ms/seq)

Optional:
  • --mm_scoring  true   # benchmark masked-marginal scoring (slower)
"""
import argparse, random, time, torch
from rinalmo.pretrained import get_pretrained_model
import numpy as np

RNA = "ACGU"

def rand_seq(n: int, alphabet: str = RNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def masked_marginal(model, alphabet, seqs, device):
    """Slow: one forward per masked position, like主脚本."""
    total_calls = 0
    for s in seqs:
        toks = torch.tensor(alphabet.batch_tokenize([s]), device=device)
        L = toks.size(1) - 2                            # exclude BOS/EOS
        for p in range(L):
            mtoks = toks.clone()
            mtoks[0, p+1] = alphabet.mask_idx
            _ = model(mtoks)["logits"]
            total_calls += 1
    return total_calls

def plain_forward(model, alphabet, seqs, device):
    toks = torch.tensor(alphabet.batch_tokenize(seqs), device=device)
    _ = model(toks)["logits"]

def main():
    p = argparse.ArgumentParser("RinAlMo speed benchmark")
    p.add_argument("--model", default="giga-v1", help="checkpoint name (default giga-v1)")
    p.add_argument("--n_seq", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mm_scoring", action="store_true",
                   help="benchmark masked-marginal scoring (slow)")
    p.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = get_pretrained_model(model_name=args.model)
    model.to(device).eval()

    # 生成随机 RNA
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast():
        if args.mm_scoring:
            # 每批次拆开，逐序列测 M-M-S
            for i in range(0, args.n_seq, args.batch_size):
                batch = seqs[i : i + args.batch_size]
                _ = masked_marginal(model, alphabet, batch, device)
        else:
            # 普通一次性前向
            for i in range(0, args.n_seq, args.batch_size):
                batch = seqs[i : i + args.batch_size]
                plain_forward(model, alphabet, batch, device)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    mode = "MM-Scoring" if args.mm_scoring else "Plain forward"
    print(f"Mode            : {mode}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"RinAlMo {args.model} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

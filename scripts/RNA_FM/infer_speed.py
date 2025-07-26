#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for RNA-FM T12 on 100-nt RNA sequences.

Reports:
  • total wall-time
  • throughput  (seq/s)
  • avg latency (ms/seq)
"""
import argparse, random, time, torch
import numpy as np
import sys, pathlib
sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/baselines/RNA_FM/RNA-FM")
def rand_seq(n=100, alphabet="ACGU"):
    return "".join(random.choices(alphabet, k=n))

def load_model():
    # ensure fm package in path
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    return model, alphabet

def main():
    p = argparse.ArgumentParser("RNA-FM speed benchmark (100 nt)")
    p.add_argument("--n_seq", type=int, default=1024,
                   help="# sequences to benchmark (default 1024)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=100,
                   help="Length of each sequence (default: 100)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = load_model()
    model.to(device)

    # --- synthetic data ----------------------------------------------------
    seqs = [rand_seq() for _ in range(args.n_seq)]
    batch_converter = alphabet.get_batch_converter()

    # --- timed inference ---------------------------------------------------
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i : i + args.batch_size]
            _, _, tokens = batch_converter([(f"id{j}", s) for j, s in enumerate(batch)])
            logits = model(tokens.to(device))["logits"]      # (B, L, V)
            _ = torch.log_softmax(logits, dim=-1)            # mimic real use
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # --- report ------------------------------------------------------------
    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"RNA-FM T12 {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

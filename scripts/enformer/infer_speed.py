#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for Enformer (EleutherAI/enformer-official-rough).
Measures wall-time, throughput, and avg latency.

NOTE: Enformer expects fixed 196 608-token windows; shorter
      sequences will be right-padded with '-' as in your pipeline.
"""
import argparse, random, time, numpy as np, torch
from enformer_pytorch import from_pretrained

DNA = "ACGT"

# -------- helper -------------------------------------------------------------
def rand_seq(n: int, alphabet: str = DNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def encode_batch(seqs, max_len=196_608):
    """Map A,C,G,T,N,- to ints 0-4 / -1, pad / truncate to max_len."""
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4,'-':-1}
    B = len(seqs)
    arr = np.full((B, max_len), -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s.upper().replace("U", "T")
        s = s[:max_len].ljust(max_len, '-')    # pad / truncate
        arr[i] = [mapping[b] for b in s]
    return torch.from_numpy(arr)

# -------- main ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("Enformer speed benchmark (196k window)")
    p.add_argument("--checkpoint", default="EleutherAI/enformer-official-rough")
    p.add_argument("--n_seq", type=int, default=1024, help="#windows to benchmark")
    p.add_argument("--batch_size", type=int, default=16, help="windows per forward pass")
    p.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = from_pretrained(args.checkpoint).to(device).eval()

    # synthetic data: 100-nt random seed, rest padded to window
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch_seqs = seqs[i : i + args.batch_size]
            tokens = encode_batch(batch_seqs).to(device)     # (B, 196 608)
            logits = model(tokens)["human"]                  # (B, L, C)
            _ = logits.mean(dim=1)                           # (B, C) – discard
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Device          : {device}")
    print(f"Total windows   : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.2f} windows/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/window")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"Enformer gene {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed/args.n_seq*1000:.1f}\n")
if __name__ == "__main__":
    main()

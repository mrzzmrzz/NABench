#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for Evo models on 100-nt sequences.

Outputs:
  • total wall-time
  • throughput (seq/s)
  • avg latency (ms/seq)
"""
import argparse, random, time, numpy as np, torch
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs

DNA = "ACGT"

def rand_seq(n: int, alphabet: str = DNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def main():
    p = argparse.ArgumentParser("Evo speed benchmark (100 nt)")
    p.add_argument("--model_name", default="evo-1.5-8k-base")
    p.add_argument("--n_seq", type=int, default=1024, help="# sequences (default 1k)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evo = Evo(args.model_name)
    model, tok = evo.model.to(device).eval(), evo.tokenizer

    # ── synthetic data ──────────────────────────────────────────
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    # ── timed inference ────────────────────────────────────────
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i : i + args.batch_size]
            inp, _ = prepare_batch(batch, tok, prepend_bos=True, device=device)
            logits, *_ = model(inp)
            _ = logits_to_logprobs(logits, inp, trim_bos=True)  # 仅耗时，不存结果
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # ── report ─────────────────────────────────────────────────
    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq / elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed / args.n_seq * 1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"Evo {args.model_name} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")

if __name__ == "__main__":
    main()

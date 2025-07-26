#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for Ernie-RNA MLM on 100-nt RNA sequences.

Reports:
  • total wall-time
  • throughput   (seq/s)
  • avg latency  (ms/seq)
"""
import argparse, random, time, numpy as np, paddle
from paddlenlp.transformers import ErnieForMaskedLM
from src.rna_ernie import BatchConverter      # 确保 PYTHONPATH 含 src 目录

RNA = "ACGU"

def rand_seq(n=100, alphabet=RNA):
    return "".join(random.choices(alphabet, k=n))

def main():
    ap = argparse.ArgumentParser("Ernie-RNA speed benchmark (100 nt)")
    ap.add_argument("--model_checkpoint", required=True,
                    help="Dir containing Ernie params & model_state.pdparams")
    ap.add_argument("--vocab_path", required=True,
                    help="Path to vocab file used by BatchConverter")
    ap.add_argument("--n_seq", type=int, default=1024,
                    help="# sequences to benchmark (default 1024)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=100,
                    help="Length of each sequence (default: 100)")
    args = ap.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    device   = "gpu" 
    paddle.set_device(device)

    # ── model & converter ───────────────────────────────────────────────
    model = ErnieForMaskedLM.from_pretrained(args.model_checkpoint)
    model.eval()
    converter = BatchConverter(k_mer=1,
                               vocab_path=args.vocab_path,
                               batch_size=args.batch_size,
                               max_seq_len=512)

    # ── synthetic data ─────────────────────────────────────────────────
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    # ── timed inference ────────────────────────────────────────────────
    start = time.perf_counter()
    with paddle.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i:i+args.batch_size]
            for _, _, token_ids in converter([(f"{j}", s) for j, s in enumerate(batch)]):
                logits = model(token_ids)    # (B, L, V)
                _ = paddle.nn.functional.softmax(logits, axis=-1)  # mimic实际路径
    paddle.device.cuda.synchronize() 
    elapsed = time.perf_counter() - start

    # ── report ─────────────────────────────────────────────────────────
    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"Ernie-RNA {args.model_checkpoint} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

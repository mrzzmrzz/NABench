#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for LucaOne inference speed on 100-length sequences.
Measures:
  • total wall-time for all sequences (throughput)
  • average latency per sequence
"""
import argparse, time, random, string, torch, numpy as np
from pathlib import Path
import sys
sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/baselines/LucaOne/LucaOneTasks/src/llm/lucagplm/")

from get_embedding import predict_embedding   # LucaOne helper
DNA_ALPHABET  = "ACGT"
AA_ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"

def gen_sequence(length: int, alphabet: str) -> str:
    return "".join(random.choices(alphabet, k=length))

def main():
    parser = argparse.ArgumentParser("LucaOne speed benchmark (100-length)")
    parser.add_argument("--llm_dir", required=True,
                        help="Parent directory containing LucaOne models")
    parser.add_argument("--seq_type", choices=["gene", "prot"], default="gene")
    parser.add_argument("--n_seq", type=int, default=1024,  # enough for stable avg
                        help="Total sequences to benchmark (default 1 k)")
    parser.add_argument("--truncation_seq_length", type=int, default=100,
                        help="Max sequence length for LucaOne inference (not include special tokens)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_type", choices=["vector", "matrix"], default="vector")
    args = parser.parse_args()
    file_path = f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.truncation_seq_length}_logs.txt"

    alphabet = DNA_ALPHABET if args.seq_type == "gene" else AA_ALPHABET
    sequences = [gen_sequence(args.truncation_seq_length, alphabet) for _ in range(args.n_seq)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()

    # === batched inference ===
    for i in range(0, args.n_seq, args.batch_size):
        batch = sequences[i:i + args.batch_size]
        # predict_embedding 接收一条序列时的参数是 list[str] ⇒ 这里逐条跑
        # 若想进一步加速，可修改 predict_embedding 支持批量输入
        for idx, seq in enumerate(batch):
            predict_embedding(
                args.llm_dir,
                [str(idx), args.seq_type, seq],
                args.truncation_seq_length,
                truncation_seq_length=args.truncation_seq_length,
                embedding_type=args.embedding_type,
                repr_layers=[-1],
                device=device,
                matrix_add_special_token=False
            )
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/seq")
    with open(file_path, "a") as f:
        f.write(f"LucaVirus {args.seq_type} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed/args.n_seq*1000:.1f}\n")
if __name__ == "__main__":
    main()

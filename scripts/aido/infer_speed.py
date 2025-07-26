#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, random, string, torch
import numpy as np
from modelgenerator.tasks import Embed

DNA = "ACGT"
AA  = "ACDEFGHIKLMNPQRSTVWY"

def rand_seq(n: int, alphabet: str) -> str:
    return "".join(random.choices(alphabet, k=n))

def main() -> None:
    parser = argparse.ArgumentParser("Infer Speed")
    parser.add_argument("--backbone", default="aido_rna_1b600m",
                        help="Embed backbone name (default: aido_rna_1b600m)")
    parser.add_argument("--seq_type", choices=["gene", "prot"], default="gene")
    parser.add_argument("--n_seq", type=int, default=1024, help="# sequences (default 1 k)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = parser.parse_args()
    # ── model ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = Embed.from_config({"model.backbone": args.backbone}).to(device).eval()

    # ── data ───────────────────────────────────────────────────────────────
    alphabet = DNA if args.seq_type == "gene" else AA
    seqs = [rand_seq(args.seq_len, alphabet) for _ in range(args.n_seq)]

    # ── timed inference ────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i : i + args.batch_size]
            # Evo 模型的典型两步调用
            transformed = model.transform({"sequences": batch})
            _ = model(transformed)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # ── report ─────────────────────────────────────────────────────────────
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq / elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed / args.n_seq * 1000:.1f} ms/seq")

    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"Aido.RNA {args.seq_type} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")

if __name__ == "__main__":
    main()

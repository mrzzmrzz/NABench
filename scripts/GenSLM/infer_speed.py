#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for GenSLM models on 100-nt sequences.

Metrics:
  • total wall-time
  • throughput  (seq/s)
  • avg latency (ms/seq)
"""
import argparse, random, time, torch
import numpy as np
from genslm import GenSLM, SequenceDataset
from torch.utils.data import DataLoader

DNA = "ACGT"

def rand_seq(n: int, alphabet: str = DNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def main():
    ap = argparse.ArgumentParser("GenSLM speed benchmark (100 nt)")
    ap.add_argument("--model_name", default="genslm_2.5B_patric")
    ap.add_argument("--checkpoint_dir", required=True,
                    help="Directory containing GenSLM checkpoints")
    ap.add_argument("--n_seq", type=int, default=1024,
                    help="# sequences to benchmark (default 1k)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=100,
                    help="Length of each sequence (default: 100)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenSLM(args.model_name, model_cache_dir=args.checkpoint_dir).to(device).eval()
    print("Parameters:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
    # ── synthetic 100-nt data ──────────────────────────────────────────────
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]
    ds   = SequenceDataset(seqs, seq_length=args.seq_len+2, tokenizer=model.tokenizer)  # +2 for BOS/EOS
    dl   = DataLoader(ds, batch_size=args.batch_size)

    # ── timed inference ───────────────────────────────────────────────────
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dl:
            out = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        use_cache=False,              # 与主评测脚本一致
                        output_hidden_states=True)
            _ = out.hidden_states[-1].detach().cpu().numpy()                        # 只触发前向，结果不存
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # ── report ────────────────────────────────────────────────────────────
    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq / elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed / args.n_seq * 1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"GenSLM {args.model_name} {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

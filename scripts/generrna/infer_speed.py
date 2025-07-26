#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for GenerRNA GPT model on 100-nt sequences.

Outputs:
  • total wall-time
  • throughput  (seq/s)
  • avg latency (ms/seq)
"""
import argparse, random, time, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import sys
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append("/data4/marunze/generrna/")
from GenerRNA.model import GPT, GPTConfig

DNA = "ACGT"

def rand_seq(n: int, alphabet: str = DNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    # 兼容 _orig_mod. 前缀
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval()

def nll_no_grad(model, input_ids: torch.Tensor):
    """返回 batch NLL；只为计算耗时，结果不存。"""
    with torch.no_grad():
        logits, loss = model(input_ids[:, :-1], input_ids[:, 1:])
    return loss

def main():
    p = argparse.ArgumentParser("GenerRNA speed benchmark (100 nt)")
    p.add_argument("--ckpt", required=True, help="GenerRNA ckpt.pt")
    p.add_argument("--tokenizer_dir", default="GenerRNA/tokenizer")
    p.add_argument("--n_seq", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=100, help="Length of each sequence (default: 100)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok    = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.add_special_tokens({'pad_token': '[PAD]'})
    model  = load_model(args.ckpt, device)

    # synthetic data
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    # timed loop
    start = time.perf_counter()
    for i in range(0, args.n_seq):
        batch = seqs[i]
        ids   = tok.encode(batch)
        ids =  torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        _ = nll_no_grad(model, ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Device          : {device}")
    print(f"Total sequences : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq / elapsed:.1f} seq/s")
    print(f"Avg. latency    : {elapsed / args.n_seq * 1000:.1f} ms/seq")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"GenerRNA gene {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed / args.n_seq * 1000:.1f}\n")
if __name__ == "__main__":
    main()

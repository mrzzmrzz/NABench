#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark for SPACE model (yangyz1230/space).

• 固定窗口 131 072 bp；短序列右填 '-'，长序列截断。  
• 输出总耗时、吞吐率 (seq/s) 与单窗口平均延迟 (ms/seq)。
"""
import argparse, random, time, numpy as np, torch, sys
sys.path.append("./space/SPACE")           # 确保能找到自定义包
from SPACE.model.modeling_space import Space  ,SpaceConfig         # pip install git+https://github.com/yangyz1230/SPACE 亦可

DNA = "ACGT"
MAX_LEN = 131_072                    # model upper-bound

def rand_seq(n: int, alphabet: str = DNA) -> str:
    return "".join(random.choices(alphabet, k=n))

def encode_batch(seqs):
    """A,C,G,T,N -> 0-4, pad '-' -> -1."""
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4,'-':-1,'U':3,'X':4}
    B = len(seqs)
    arr = np.full((B, MAX_LEN), -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s.upper().replace("U", "T")
        s = s[:MAX_LEN].ljust(MAX_LEN, '-')
        arr[i] = [mapping[b] for b in s]
    return torch.from_numpy(arr)

def main():
    ap = argparse.ArgumentParser("SPACE speed benchmark")
    ap.add_argument("--checkpoint", default="yangyz1230/space")
    ap.add_argument("--n_seq", type=int, default=1024,
                    help="#windows to benchmark (default 256)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=100,
                    help="Length of each sequence (default: 500)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SpaceConfig.from_pretrained('yangyz1230/space')
    config.input_file = ""
    model = Space.from_pretrained(args.checkpoint,config=config).to(device).eval()

    # synthetic data: 500-bp random seq padded到131 072
    seqs = [rand_seq(args.seq_len) for _ in range(args.n_seq)]

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, args.n_seq, args.batch_size):
            batch = seqs[i : i + args.batch_size]
            tokens = encode_batch(batch).to(device)        # (B, 131 072)
            logits = model(tokens)["out"]                  # (B, L, C)
            _ = logits.mean(dim=1)                         # (B, C) 仅为耗时
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Device          : {device}")
    print(f"Total windows   : {args.n_seq}")
    print(f"Total time      : {elapsed:.2f} s")
    print(f"Throughput      : {args.n_seq/elapsed:.2f} windows/s")
    print(f"Avg. latency    : {elapsed/args.n_seq*1000:.1f} ms/window")
    with open(f"/home/ma_run_ze/lzm/rnagym/fitness/speed_test/{args.seq_len}_logs.txt", "a") as f:
        f.write(f"SPACE gene {args.n_seq} {elapsed:.2f} {args.batch_size} {elapsed/args.n_seq*1000:.1f}\n")
if __name__ == "__main__":
    main()

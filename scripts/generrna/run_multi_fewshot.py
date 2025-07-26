#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_generrna_fewshot.py
一次性跑多个 few_shot_k，对比 GenerRNA Few-shot 性能
----------------------------------------------------------------
用法示例
python batch_generrna_fewshot.py \
  --k_list 5 20 50 100 \
  --model_ckpt   /path/generrna_ckpt.pt \
  --tokenizer_dir /path/tokenizer \
  --ref_sheet    /path/reference_sheet_final.csv \
  --dms_dir      /path/fitness_processed_assays \
  --out_root     /path/scores/GenerRNA_fewshot_runs \
  --few_shot_repeat 10
"""

import argparse, subprocess, shutil, csv, os
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k_list", type=int, nargs="+", required=True,
                   help="多个 few_shot_k (空格分隔)")
    p.add_argument("--few_shot_repeat", type=int, default=5)
    # 以下参数原样转发
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--ref_sheet", required=True)
    p.add_argument("--dms_dir", required=True)
    p.add_argument("--out_root", required=True,
                   help="各 k 结果会写入 <out_root>/k_<k>/ ...")
    p.add_argument("--cache_dir", default="/data4/marunze/generrna/cache")
    p.add_argument("--device",  default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    merged_csv = out_root / "fewshot_all_k.csv"
    rows = []

    for k in args.k_list:
        print(f"\n=== running few-shot k={k} ===")
        out_dir = out_root / f"k_{k}"
        cmd = [
            "python", "run_eval.py",
            "--scheme", "few-shot",
            "--few_shot_k", str(k),
            "--few_shot_repeat", str(args.few_shot_repeat),
            "--model_ckpt", args.model_ckpt,
            "--tokenizer_dir", args.tokenizer_dir,
            "--ref_sheet", args.ref_sheet,
            "--dms_dir", args.dms_dir,
            "--output_dir", str(out_dir),
            "--cache_dir", args.cache_dir,
            "--device", args.device,
        ]
        subprocess.run(cmd, check=True)

        # 每次跑完读取 summary
        summ_path = out_dir / "correlation_summary.txt"
        if not summ_path.exists():
            raise FileNotFoundError(f"{summ_path} missing!")
        tmp = pd.read_csv(summ_path)
        tmp["k"] = k
        rows.append(tmp[["k","DMS_ID","Spearman_mean","Spearman_std"]])

    # 合并并写总表
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(merged_csv, index=False)
    print(f"\n合并完成 → {merged_csv}")

if __name__ == "__main__":
    main()

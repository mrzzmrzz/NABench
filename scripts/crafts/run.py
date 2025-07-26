#!/usr/bin/env python3
"""
脚本说明: 使用 CRAFTS_LM 模型对 SELEX 或其他高通量筛选实验数据集进行推理。

该脚本会加载一个预训练的 CRAFTS_LM 模型，然后遍历一个清单文件（reference sheet）中列出的所有实验。
对于每个实验，它会处理相应的RNA序列，计算模型给出的对数似然分数，并评估模型分数与实验测量值的相关性。
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

# 将当前目录（crafts_lm 根目录）添加到 Python 路径，以便导入其模块
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/baselines/crafts/crafts_lm")
from crafts_lm.utils.lm import get_extractor, get_model_args
# 从 utils.predictor 导入 SS_predictor
from crafts_lm.utils.predictor import SS_predictor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用 CRAFTS_LM 模型在指定的RNA高通量筛选数据集上运行推理。"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="指定要加载的 CRAFTS_LM 模型名称或检查点路径 (例如 'esm2' 或 './checkpoints/my_model.pt')。"
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="实验清单文件的路径 (CSV格式)，其中列出了所有要处理的数据集。"
    )
    parser.add_argument(
        "--assay_dir_path",
        type=str,
        required=True,
        help="包含所有数据集CSV文件的目录路径。"
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="./outputs_crafts_lm",
        help="保存输出结果的目录路径。"
    )
    parser.add_argument(
        "--score_column",
        type=str,
        default="SELEX_score",  # SELEX 数据中常见的列名
        help="数据集中包含实验测量分数（如富集度）的列名。"
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="sequence",  # SELEX 数据中常见的列名
        help="数据集中包含RNA序列的列名。"
    )
    parser.add_argument(
        "--wt_sequence_column",
        type=str,
        default="RAW_CONSTRUCT_SEQ",
        help="清单文件中包含野生型或参考序列的列名。"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64, # CRAFTS_LM 可以处理更大的批次
        help="用于模型推理的批处理大小。"
    )
    parser.add_argument(
        "--tok_mode",
        type=str,
        choices=["char", "word", "phone"],
        default="char",)
    return parser.parse_args()

def preprocess_sequence(sequence: str) -> str:
    """
    预处理RNA序列：
    - 转换为大写
    - 去除首尾空白
    """
    # 确保输入是字符串
    if not isinstance(sequence, str):
        return ""
    return sequence.strip().upper()

def main():
    """主执行函数"""
    args = parse_args()
    
    # --- 1. 环境设置并加载模型 (仅一次) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"使用的设备: {device}")
    
    # output_dir = Path(args.output_dir_path)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # print(f"正在从 '{args.model_name_or_path}' 加载 CRAFTS_LM 模型...")
    # try:
    #     # 使用 Predictor 类加载模型，它会自动处理设备、词汇表等
    #     predictor = Predictor(args.model_name_or_path, device=device)
    #     print("模型加载成功。")
    # except Exception as e:
    #     print(f"错误: 加载模型失败: {e}")
    #     return
    # print("正在初始化 SS_predictor 模型…")
    try:
        # 1. 构建 extractor 和 tokenizer
        extractor, tokenizer = get_extractor(args)
        # 2. 根据 model_scale、model_type 和 vocab_size 生成模型配置
        model_config = get_model_args(
            "lx",
            False,
            tokenizer.vocab_size
        )
        # 3. 实例化 SS_predictor
        model = SS_predictor(extractor, model_config, is_freeze=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        # 4. 如指定了预训练权重，则加载
        if args.model_name_or_path:
            state_dict = torch.load(
                os.path.join(args.model_name_or_path),
                map_location='cpu'
            )
            model_dict = predictor.state_dict()
            # 只保留匹配的 key
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            predictor.load_state_dict(model_dict)
        # 5. 移动到设备并设为 eval
        predictor.to(device)
        predictor.eval()
        print("SS_predictor 模型加载成功。")
    except Exception as e:
        print(f"错误: 初始化 SS_predictor 失败: {e}")
        raise e
        return


    # --- 2. 加载清单文件并遍历所有实验 ---
    try:
        ref_sheet = pd.read_csv(args.ref_sheet)
        print(f"\n在 '{args.ref_sheet}' 中找到 {len(ref_sheet)} 个实验需要处理。")
    except FileNotFoundError:
        print(f"错误: 清单文件未找到: {args.ref_sheet}")
        return
    output_dir = Path(args.output_dir_path)
    summary_filepath = output_dir / "correlation_summary.csv"
    if summary_filepath.exists():
        print(f"发现已存在的汇总文件 {summary_filepath}，将追加结果。")

    for row_id, assay_info in ref_sheet.iterrows():
        print(f"\n{'='*20} 正在处理第 {row_id} 行: 目标 '{assay_info.get('Target', 'N/A')}' {'='*20}")
        
        try:
            # --- 加载每个实验的数据 ---
            assay_filename = assay_info['DMS_ID']  + '.csv' # 保持列名与原始脚本一致，增加兼容性
            assay_file = Path(args.assay_dir_path) / assay_filename
            if not assay_file.exists():
                print(f"警告: 第 {row_id} 行的数据集文件 '{assay_filename}' 未找到，已跳过。")
                continue

            print(f"正在从以下路径加载数据: {assay_file}")
            assay_df = pd.read_csv(assay_file)

            if args.wt_sequence_column not in assay_info or pd.isna(assay_info[args.wt_sequence_column]):
                print(f"警告: 第 {row_id} 行未提供参考序列 (wt_seq)，已跳过。")
                continue
            wt_sequence = preprocess_sequence(assay_info[args.wt_sequence_column])
            if not wt_sequence:
                print(f"警告: 第 {row_id} 行的参考序列为空，已跳过。")
                continue

            # --- 推理逻辑 (使用 CRAFTS_LM Predictor) ---
            # 1. 计算参考序列 (wt) 的分数
            # prepare input for the model
            data_dict = {
                'input_ids': tokenizer.encode(wt_sequence, return_tensors='pt').to(device),
                'attention_mask': None
            }
            with torch.no_grad():
                log_p_wt = torch.mean(torch.softmax(predictor(data_dict),dim=1))

            # 2. 批量计算所有突变/变体序列的分数
            sequences_to_process = [
                preprocess_sequence(seq) for seq in assay_df[args.sequence_column].tolist()
            ]
            
            # 过滤掉预处理后为空的序列
            valid_indices = [i for i, seq in enumerate(sequences_to_process) if seq]
            valid_sequences = [sequences_to_process[i] for i in valid_indices]

            print(f"开始对 {len(valid_sequences)} 个有效序列进行推理...")
            all_log_p_mut = []
            w = 0.6
            for i in tqdm(range(0, len(valid_sequences)), desc="推理进度"):
                seq = valid_sequences[i]
                # prepare input for the model
                data_dict = {
                    'input_ids': tokenizer.encode(seq, return_tensors='pt').to(device),
                    'attention_mask': None
                }
                # 计算对数似然分数
                with torch.no_grad():
                    log_p_mut = torch.softmax(predictor(data_dict), dim=1)
                all_log_p_mut.extend(log_p_mut.cpu().numpy())
            all_log_p_mut = np.array(all_log_p_mut)
            # 3. 计算相对分数 (log p(mut) - log p(wt))
            relative_scores = all_log_p_mut[:, 1] + w * all_log_p_mut[:, 2]  
            model_scores = np.full(len(assay_df), np.nan) # 初始化为NaN
            
            # 将分数放回原位
            for i, score in zip(valid_indices, relative_scores):
                model_scores[i] = score

            assay_df['model_score'] = model_scores
            
            # --- 计算相关性并保存结果 ---
            final_df = assay_df.dropna(subset=[args.score_column, 'model_score'])
            
            if len(final_df) > 1:
                correlation, p_value = spearmanr(final_df[args.score_column], final_df['model_score'])
                print(f"斯皮尔曼相关系数 (ρ): {correlation:.4f} (p-value: {p_value:.4g})")
            else:
                correlation = np.nan
                p_value = np.nan
                print("没有足够的数据来计算相关性。")

            output_filename = f"crafts_lm_results_row_{row_id}_{assay_filename.replace('.csv', '')}.csv"
            output_filepath = output_dir / output_filename
            assay_df.to_csv(output_filepath, index=False)
            print(f"单个实验的结果已保存至: {output_filepath}")

            summary_data = {
                'row_id': row_id,
                'target': assay_info.get('Target', 'N/A'),
                'spearman_correlation': correlation,
                'p_value': p_value,
                'num_sequences': len(final_df),
                'model_name': Path(args.model_name_or_path).name,
                'assay_file': assay_filename
            }
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(summary_filepath, mode='a', header=not summary_filepath.exists(), index=False)

        except Exception as e:
            print(f"处理第 {row_id} 行时发生意外错误: {e}。已跳过。")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*20} 所有实验处理完毕。 {'='*20}")
    print(f"所有结果的汇总已保存至: {summary_filepath}")

if __name__ == "__main__":
    main()
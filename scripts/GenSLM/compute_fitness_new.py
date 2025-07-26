import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

def get_sequences(wt_sequence, df, experiment_id=None):
    # Function to apply a single mutation
    def apply_mutation(sequence, mutation):
        sequence= sequence.replace('U', 'T')
        possible_bases = ['A', 'T', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        offset = 1  # Adjust for 1-based indexing in the mutation string

        pos = int(mutation[1:-1]) - offset  # Get the position, 1-based to 0-based index
        new_base = mutation[-1]  # Get the new base
        old_base = mutation[0]
        old_base = 'T' if old_base == 'U' else old_base
        new_base = 'T' if new_base == 'U' else new_base

        assert old_base in possible_bases, mutation
        assert new_base in possible_bases, mutation

        if old_base == 'N':
            # This is going to be an insertion
            mutated_sequence = sequence[:pos+1] + new_base + sequence[pos+1:]
        elif new_base == '':
            # This is going to be a deletion
            mutated_sequence = sequence[:pos] + sequence[pos+1:]
        else:
            # This is a substitution
            assert old_base == sequence[pos], mutation
            mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]

        return mutated_sequence

    # Function to apply multiple mutations
    def apply_mutations(sequence, mutations):
        # check if nan in mutations
        if pd.isna(mutations):
            return sequence
        for mutation in mutations.split(','):
            mutation = mutation.strip()
            sequence = apply_mutation(sequence, mutation)
        return sequence

    # Determine which column to use for mutations
    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    if mutation_column:
        # Apply the mutations to create a new column with mutated sequences
        df['mutated_sequence'] = df[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    else:
        raise ValueError("No 'mutant' or 'mutation' column found in the DataFrame")

    return df

def evaluate(embeddings: np.ndarray, scores: np.ndarray):
    mask = ~np.isnan(scores)
    if not mask.all():
        num_nan = len(scores) - mask.sum()
        print(f"Warning: {num_nan} samples have NaN scores and will be excluded from evaluation")
    emb = embeddings[mask]
    sc = scores[mask]
    # 使用 RidgeCV 做交叉验证
    alphas = np.logspace(-3, 3, 7)
    model = RidgeCV(alphas=alphas, store_cv_results=True)
    model.fit(emb, sc)
    preds = model.predict(emb)
    corr, pval = spearmanr(preds, sc)
    return corr, pval, preds


def process_single_row(row, model, device, base_dir, results_dir, score_column):
    dataset = row['DMS_ID']
    dataset_name = row["DMS_ID"]
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = row['RAW_CONSTRUCT_SEQ'].upper()
    try:
        sequences = get_sequences(wt_seq, df, dataset_name)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return

    output_file = os.path.join(results_dir, f"{dataset}.csv")

    seq_length = min(model.seq_length, sequences["mutated_sequence"].str.len().max() + 2)
    if model.seq_length < sequences["mutated_sequence"].str.len().max() + 2:
        print("warning: max str length exceeded")
    dataset = SequenceDataset(sequences["mutated_sequence"], seq_length, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                output_hidden_states=True,
                use_cache=False,             # 关键：关闭缓存
                return_dict=True,
            )
            logits = outputs.logits
            print("logits shape:", logits.shape)
            normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1)
            for j, seq in enumerate(batch["input_ids"]):
                input_ids = batch['input_ids'][j]
                # Ignore padding tokens
                valid_length = (input_ids != model.tokenizer.pad_token_id).sum().item()

                if valid_length == 0:
                    continue

                # Get log probabilities of the correct tokens
                log_probs = normalized_logits[j, torch.arange(valid_length), input_ids[:valid_length]]
                avg_log_prob = log_probs.mean().item()

                scores.append(avg_log_prob)

    sequences[score_column] = scores
    sequences.to_csv(output_file, index=False)

    corr = sequences[score_column].corr(sequences["dms_score"])
    print("Correlation between logit scores and DMS scores:", corr)
    summary_path = os.path.join(results_dir, f"summary.txt")

    with open(summary_path,"a") as f:
        f.write(f"{dataset_name},{corr}\n")


def main(args):

    model = GenSLM("genslm_2.5B_patric", model_cache_dir=args.checkpoint_dir)
    model.eval()
    # disable cache to avoid the mask-dimension mismatch


    # Select GPU device if it is available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    base_dir = args.dms_directory
    results_dir = args.output_directory
    summary_path = os.path.join(results_dir, f"summary.txt")

    # with open(summary_path, "w") as f:
    #     f.write("Experiment_ID\tCorrelation\n")
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs['YEAR'] = wt_seqs['YEAR'].astype(int)
    wt_seqs['AUTHOR'] = wt_seqs['AUTHOR'].astype(str)
    for curr_task_id in range(args.task_id):
        # Select the row corresponding to the task ID
        row = wt_seqs.iloc[curr_task_id]
        # Reset the first index to "DMS_ID" for consistency
        index_list = row.index.tolist()

        # Change the first element's label
        if index_list: # Ensure the index is not empty
            index_list[0] = "DMS_ID"

        # Assign the modified list back to the Series' index
        row.index = index_list

        print(row)
        process_single_row(row, model, device, base_dir, results_dir, score_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="GenSLM checkpoints directory")
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    args = parser.parse_args()
    main(args)
from transformers import AutoTokenizer, AutoModelForMaskedLM
from numpy import dot
from numpy.linalg import norm
from Bio import SeqIO
import torch
import argparse
import pandas as pd
import os
import copy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
import sys
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge

def get_sequences(wt_sequence, df, experiment_id=None):
    
    # Function to apply a single mutation
    def apply_mutation(sequence, mutation):
        sequence= sequence.replace('U', 'T')
        possible_bases = ['A', 'T', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        offset = 1  # Adjust for 1-based indexing in the mutation string
        
        pos = int(mutation[1:-1]) - offset
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
        for mutation in mutations.split(','):
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


def evaluate(embeddings: np.ndarray, scores: np.ndarray, cv=False, few_shot_k=None, few_shot_repeat=5, seed=42):
    np.random.seed(seed)
    
    mask = ~np.isnan(scores)
    if not mask.all():
        num_nan = len(scores) - mask.sum()
        print(f"Warning: {num_nan} samples have NaN scores and will be excluded from evaluation")

    emb = embeddings[mask]
    sc = scores[mask]
    
    # Few-shot 模式
    if few_shot_k is not None:
        print(f"Running few-shot evaluation with k={few_shot_k}, repeated {few_shot_repeat} times")
        corrs = []
        best_model = None
        for r in range(few_shot_repeat):
            indices = np.random.choice(len(sc), size=few_shot_k, replace=False)
            # flatten embeddings if they are 3D
            if emb.ndim == 3:
                emb = emb.reshape(emb.shape[0], -1)
            emb_train = emb[indices]
            sc_train = sc[indices]

            emb_test = np.delete(emb, indices, axis=0)
            sc_test = np.delete(sc, indices)

            model = Ridge(alpha=1.0)
            model.fit(emb_train, sc_train)
            preds = model.predict(emb_test)

            corr, pval = spearmanr(preds, sc_test)
            corrs.append(corr)
            if best_model is None or corr > np.mean(corrs):
                best_model = model
        avg_emb = best_model.predict(emb)
        print(f"Average correlation over {few_shot_repeat} repeats: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
        print("Shape of average embedding:", avg_emb.shape)

        return np.mean(corrs), np.std(corrs), avg_emb
    
    if cv:
        if emb.ndim > 2:
            emb = emb.mean(axis=1)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        preds = np.zeros(len(sc))
        for train_index, test_index in kf.split(emb):
            model = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_results=True)
            model.fit(emb[train_index], sc[train_index])
            preds[test_index] = model.predict(emb[test_index])
        corr, pval = spearmanr(preds, sc)
        avg_emb = preds
    else:
        avg_emb = embeddings[:,0]
        corr, pval = spearmanr(avg_emb, scores)
    
    return corr, pval, avg_emb
    

    
def score_variants(assay, model, tokenizer, base_dir, results_dir, score_column, batch_size):
    cache_file = os.path.join(args.cache_dir, f"{assay['DMS_ID']}_embeddings.npy")
    dataset = assay['DMS_ID']   
    output_file = os.path.join(results_dir, f"{dataset}.csv")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = assay['DMS_ID']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    df = df[~df[mutation_column].isna()]
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = assay['Raw_Construct_Seq'.upper()].upper()

    try:
        df = get_sequences(wt_seq, df, experiment_id=dataset)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return

    output_file = os.path.join(results_dir, f"{dataset}.csv")

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        scores = np.load(cache_file)
    else:

        max_length = tokenizer.model_max_length
        scores = []


        sequences = df.mutated_sequence
        # Process sequences in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
            batch = sequences[i:i + batch_size]
            # Check and truncate sequences if they exceed the model's maximum length
            batch = [seq[:max_length] for seq in batch]

            # Tokenize the batch of sequences
            tokens = tokenizer(batch, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True).to(device)

            # Process the batch
            with torch.no_grad():
                logits = model(input_ids=tokens['input_ids']).logits
                normalized_logits = F.log_softmax(logits, dim=-1)
                normalized_logits = normalized_logits.cpu().numpy()
            valid_lengths = np.array([len(seq) for seq in batch])
            # mean pooling the logits with respect to the valid lengths
            avg_log_prob = np.array([np.mean(logit[:length], axis=0) for logit, length in zip(normalized_logits, valid_lengths)])
            scores.append(avg_log_prob)
        scores = np.concatenate(scores, axis=0)
        # Save the scores to a cache file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, scores)
        print(f"Scores saved to {cache_file}")
    corr,pval,pred_scores = evaluate(scores, df["dms_score"].values, cv=args.cv,
                                        few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)

    df[score_column] = pred_scores
    df.to_csv(output_file, index=False)

    corr = df[score_column].corr(df["dms_score"])
    print("Correlation between logit scores and DMS scores:", corr)
    summary_path = os.path.join(results_dir, f"summary.txt")

    with open(summary_path,"a") as f:
        f.write(f"{dataset},{corr},{pval}\n")



def main(args):

    print('Loading model...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_location,trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_location,trust_remote_code=True).to(device)

    base_dir = args.dms_directory
    results_dir = args.output_directory
    summary_path = os.path.join(results_dir, f"summary.txt")

    with open(summary_path, "w") as f:
        f.write("Experiment_ID\tCorrelation\n")
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs['YEAR'] = wt_seqs['YEAR'].astype(int)
    wt_seqs['AUTHOR'] = wt_seqs['AUTHOR'].astype(str)
    for curr_task_id in range(args.task_id):
        # wt_seqs['Molecule Type'] = wt_seqs['Molecule Type'].astype(str)
        print(f"Processing task ID: {curr_task_id} from {len(wt_seqs)} total tasks.")
        # Select the row corresponding to the task ID
        #change the first column name to "DMS_ID"
        wt_seqs.rename(columns={wt_seqs.columns[0]: 'DMS_ID'}, inplace=True)
        row = wt_seqs.iloc[curr_task_id]
        
        # Select the row corresponding to the task ID
        assay = wt_seqs.iloc[curr_task_id]
        batch_size = 8
        score_variants(assay, model, tokenizer, base_dir, results_dir, score_column, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--model_location", type=str, help="NT Huggingface model to use")
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    parser.add_argument("--cv", action='store_true',
                        help="Use cross-validation for evaluation")
    parser.add_argument("--few_shot_k", type=int, default=None,
                        help="Number of samples for few-shot evaluation (default: None, use full data)")
    parser.add_argument("--few_shot_repeat", type=int, default=5,
                        help="Number of repeats for few-shot evaluation (default: 5)")
    parser.add_argument("--cache_dir", type=str, default="/data4/marunze/nt/cache/",
                        help="Directory to cache embeddings (optional, for large datasets)")
    args = parser.parse_args()
    main(args)



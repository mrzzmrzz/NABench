from transformers import AutoTokenizer, AutoModel
import sys

sys.path.append('./SPACE')
from SPACE.model.modeling_space import Space, SpaceConfig
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
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge

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
        model = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_results=True)
        model.fit(emb, sc)
        preds = model.predict(emb)
        corr, pval = spearmanr(preds, sc)
        avg_emb = preds
    else:
        avg_emb = embeddings[:,0]
        corr, pval = spearmanr(avg_emb, scores)
    
    return corr, pval, avg_emb
    



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
        if isinstance(mutations, float) or pd.isna(mutations):
            return sequence
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

def score_variants(assay, model,  base_dir, results_dir, score_column, batch_size=1):
    cache_file = os.path.join(args.cache_dir, f"{assay['DMS_ID']}_embeddings.npy")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = assay['DMS_ID']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    mask = df['dms_score'].notna() & df["dms_score"].notna()
    df = df[mask].copy()

    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = assay['Raw_Construct_Seq'.upper()].upper()

    try:
        df = get_sequences(wt_seq, df, experiment_id=dataset)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        scores = np.load(cache_file)
        df[score_column] = scores
        df.to_csv(os.path.join(results_dir, f"{dataset}.csv"), index=False)
    else:
        output_file = os.path.join(results_dir, f"{dataset}.csv")

        max_length = 131072
        scores = []


        sequences = df.mutated_sequence
        print("Processing total sequence of:", len(sequences))
        # Process sequences in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
            # batch = sequences[i:i + batch_size]
            # Check and truncate sequences if they exceed the model's maximum length
            # Pad the sequences to the maximum length
            seq = sequences.iloc[i]
            if len(seq) > max_length:
                seq = seq[:max_length]
            elif len(seq) < max_length:
                seq = seq.ljust(max_length, '-')
            # Map ACGTN to 01234, -1 for padding
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': -1 , 'X': 4, "U": 3}
            tokens = torch.tensor([mapping[base] for base in seq], dtype=torch.long)
            # Move tensors to the appropriate device
            tokens = tokens.to(device)

            with torch.no_grad():
                logits = model(tokens)["out"]
                normalized_logits = F.log_softmax(logits, dim=-1)

            log_prob = logits.squeeze(dim=0)
            avg_log_prob = torch.mean(log_prob,dim=0)
            avg_log_prob = avg_log_prob.cpu().numpy()


            scores.append(avg_log_prob)
        scores = np.vstack(scores)
        print("Scores shape:", scores.shape)
        # Save the scores to a cache file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, scores)
        print(f"Scores saved to {cache_file}")
    corr,pval,scores = evaluate(scores, df["dms_score"].values, cv=args.cv,
                                        few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)
    df[score_column] = scores
    df.to_csv(output_file, index=False)

    corr = df[score_column].corr(df["dms_score"])
    print("Correlation between logit scores and DMS scores:", corr)
    summary_path = os.path.join(results_dir, f"summary.txt")

    with open(summary_path,"a") as f:
        f.write(f"{dataset},{corr},{pval}\n")



def main(args):

    print('Loading model...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    base_dir = args.dms_directory
    results_dir = args.output_directory
    summary_path = os.path.join(results_dir, f"summary.txt")
    if not os.path.exists(summary_path):
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
        
        # Select the row corresponding to the task ID
        assay = wt_seqs.iloc[curr_task_id]
        batch_size = 1
        config = SpaceConfig.from_pretrained('yangyz1230/space')
        # add input_file to config
        dataset = assay['DMS_ID']

        df_path = os.path.join(base_dir, f'{dataset}.csv')
        try:
            with open(summary_path, "r") as f:
                scored_df = pd.read_csv(f, sep=",")
            if assay['DMS_ID'] in scored_df['Experiment_ID'].values:
                print(f"Skipping {assay['DMS_ID']} as it has already been scored.")
                continue
        except:
            print(f"Error reading summary file {summary_path}. It may not exist yet.")
        config.input_file = df_path
        model = Space.from_pretrained('yangyz1230/space',config=config).to(device)
        model.eval()
        score_variants(assay, model, base_dir, results_dir, score_column, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    parser.add_argument("--cv", action='store_true',
                        help="Use cross-validation for evaluation")
    parser.add_argument("--few_shot_k", type=int, default=None,
                        help="Number of samples for few-shot evaluation (default: None, use full data)")
    parser.add_argument("--few_shot_repeat", type=int, default=5,
                        help="Number of repeats for few-shot evaluation (default: 5)")
    parser.add_argument("--cache_dir", type=str, default="/data4/marunze/space/cache/",
                        help="Directory to cache embeddings (optional, for large datasets)")
    args = parser.parse_args()
    main(args)



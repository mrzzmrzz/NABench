from alphagenome.models import dna_client
from numpy import dot
from numpy.linalg import norm
import argparse
import pandas as pd
import os
import copy
from tqdm import tqdm
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

def evaluate(embeddings: np.ndarray, scores: np.ndarray, cv=False):
    if cv:
        mask = ~np.isnan(scores)
        if not mask.all():
            num_nan = len(scores) - mask.sum()
            print(f"Warning: {num_nan} samples have NaN scores and will be excluded from evaluation")
        emb = embeddings[mask]
        sc = scores[mask]
        # Using RidgeCV for cross-validation
        alphas = np.logspace(-3, 3, 7)
        model = RidgeCV(alphas=alphas, store_cv_results=True)
        model.fit(emb, sc)
        preds = model.predict(emb)
        corr, pval = spearmanr(preds, sc)
        return corr, pval, preds
    else:
        emb = embeddings.mean(axis=1)  # Average over sequence length
        sc = scores
        corr, pval = spearmanr(emb, sc)
        return corr, pval, emb


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
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import sleep
# Assume 'dna_client', 'get_sequences', and 'evaluate' are defined elsewhere

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. DEFINE THE HELPER FUNCTION (from above)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def process_sequence(sequence, model):
    """
    Processes a single sequence: pads/truncates it, gets a prediction from the model,
    and returns the calculated score.
    """
    SEQUENCE_LENGTH = 2048
    if len(sequence) > SEQUENCE_LENGTH:
        seq = sequence[:SEQUENCE_LENGTH]
    elif len(sequence) < SEQUENCE_LENGTH:
        seq = sequence.ljust(SEQUENCE_LENGTH, 'N')
    else:
        seq = sequence
    
    output = model.predict_sequence(
        sequence=seq,
        requested_outputs=[
            dna_client.OutputType.CAGE,
            dna_client.OutputType.RNA_SEQ,
        ],
        ontology_terms=['UBERON:0000955']
    )
    prediction = output.rna_seq.values + output.cage.values
    avg_score = np.mean(prediction, axis=1)
    sleep(0.18)  # Sleep to avoid hitting API rate limits
    return avg_score

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. REFACTOR THE MAIN FUNCTION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def score_variants(assay, model, base_dir, results_dir, score_column, max_workers=1):
    dataset = assay['DMS_ID']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None

    mask = df[mutation_column].notna() & df["dms_score"].notna()
    df = df[mask].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = assay['Raw_Construct_Seq'.upper()].upper()

    try:
        df = get_sequences(wt_seq, df, experiment_id=dataset)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return

    output_file = os.path.join(results_dir, f"{dataset}.csv")
    
    sequences = df.mutated_sequence
    print(f"Processing a total of {len(sequences)} sequences for {dataset}...")

    scores = []
    # Use a ThreadPoolExecutor to run API calls in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # We use partial to "fix" the 'model' argument for process_sequence
        task = partial(process_sequence, model=model)
        
        # executor.map applies the task to each sequence and returns results in order.
        # Wrap it in tqdm for a progress bar.
        # after processing, wait for 0.5 seconds to avoid hitting API rate limits

        results_iterator = executor.map(task, sequences)
        scores = list(tqdm(results_iterator, total=len(sequences), desc="Processing sequences"))
        sleep(0.5)

    scores_array = np.vstack(scores)
    print("Scores shape:", scores_array.shape)
    
    corr, pval, final_scores = evaluate(scores_array, df["dms_score"].values)
    df[score_column] = final_scores
    df.to_csv(output_file, index=False)

    final_corr = df[score_column].corr(df["dms_score"])
    print(f"Correlation for {dataset}: {final_corr}")
    summary_path = os.path.join(results_dir, f"summary.txt")

    with open(summary_path, "a") as f:
        f.write(f"{dataset},{final_corr}\n")
# def score_variants(assay, model,  base_dir, results_dir, score_column, batch_size=1):
#     dataset = assay['DMS_ID']
#     if 'snoRNA' in dataset:
#         return
#     df_path = os.path.join(base_dir, f'{dataset}.csv')
#     df = pd.read_csv(df_path)
#     df.columns = df.columns.str.lower()
#     mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None

#     mask = df[mutation_column].notna() & df["dms_score"].notna()
#     df = df[mask].copy()
#     df = df.loc[:, ~df.columns.duplicated()]
#     wt_seq = assay['Raw_Construct_Seq'.upper()].upper()

#     try:
#         df = get_sequences(wt_seq, df, experiment_id=dataset)
#     except AssertionError as e:
#         print("assertion error", e, "in", dataset)
#         return

#     output_file = os.path.join(results_dir, f"{dataset}.csv")

#     # AlphaGenome has specific supported sequence lengths.
#     # Choose one that is appropriate for your data.
#     # Supported lengths: 2048, 16384, 100352, 500224, 1000448
#     SEQUENCE_LENGTH = 2048
#     scores = []


#     sequences = df.mutated_sequence
#     print("Processing total sequence of:", len(sequences))
#     # Process sequences in batches
#     for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
#         seq = sequences.iloc[i]
#         # Truncate or pad sequence to the chosen length
#         if len(seq) > SEQUENCE_LENGTH:
#             seq = seq[:SEQUENCE_LENGTH]
#         elif len(seq) < SEQUENCE_LENGTH:
#             seq = seq.ljust(SEQUENCE_LENGTH, 'N') # Pad with 'N' for neutral base

#         # AlphaGenome prediction
#         # You need to choose which output to use. Here we use DNASE as an example.
#         # Other options include: CAGE, RNA_SEQ, CHIP_HISTONE, etc.
#         # You can also specify tissues or cell types using `ontology_terms`
#         output = model.predict_sequence(
#             sequence=seq,
#             requested_outputs=[
#                 dna_client.OutputType.CAGE,
#                 dna_client.OutputType.RNA_SEQ,
#             ],
#             ontology_terms=['UBERON:0000955']  # Optional, specify if needed
#         )
#         # Extract and average the scores. You may want to use a different aggregation.
#         # This part will depend on which output_type you choose and how you want to
#         # convert the predictions to a single score.
#         prediction = output.rna_seq.values + output.cage.values
#         print("Prediction shape:", prediction.shape)
#         avg_score = np.mean(prediction, axis=1) # Average over sequence length

#         scores.append(avg_score)

#     scores = np.vstack(scores)
#     print("Scores shape:", scores.shape)
#     corr,pval,scores = evaluate(scores, df["dms_score"].values)
#     df[score_column] = scores
#     df.to_csv(output_file, index=False)

#     corr = df[score_column].corr(df["dms_score"])
#     print("Correlation between logit scores and DMS scores:", corr)
#     summary_path = os.path.join(results_dir, f"summary.txt")

#     with open(summary_path,"a") as f:
#         f.write(f"{dataset},{corr}\n")



def main(args):

    print('Loading model...')
    # Load AlphaGenome model.
    # You need to provide your API key.
    # It is recommended to set this as an environment variable or use a secret manager.
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise ValueError("Please set the ALPHAGENOME_API_KEY environment variable.")
    model = dna_client.create(api_key)

    base_dir = args.dms_directory
    results_dir = args.output_directory
    summary_path = os.path.join(results_dir, f"summary.txt")
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("Experiment_ID,Correlation\n")
    score_column = 'alphagenome_scores' # Changed score column name
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs['YEAR'] = wt_seqs['YEAR'].astype(int)
    wt_seqs['AUTHOR'] = wt_seqs['AUTHOR'].astype(str)
    for curr_task_id in range(args.task_id):
        if curr_task_id <13:
            continue
        # wt_seqs['Molecule Type'] = wt_seqs['Molecule Type'].astype(str)
        print(f"Processing task ID: {curr_task_id} from {len(wt_seqs)} total tasks.")
        # Select the row corresponding to the task ID
        #change the first column name to "DMS_ID"
        wt_seqs.rename(columns={wt_seqs.columns[0]: 'DMS_ID'}, inplace=True)

        # Select the row corresponding to the task ID
        assay = wt_seqs.iloc[curr_task_id]
        try:
            scored_df=pd.read_csv(summary_path, sep=",")

            if assay['DMS_ID'] in scored_df['Experiment_ID'].values:
                print(f"Skipping {assay['DMS_ID']} as it has already been scored.")
                continue
        except:
            print(f"Error reading summary file {summary_path}. It may not exist yet.")
        batch_size = 1
        score_variants(assay, model, base_dir, results_dir, score_column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    parser.add_argument('--api_key', type=str, help="Your AlphaGenome API key")
    args = parser.parse_args()
    main(args)
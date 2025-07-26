import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from paddlenlp.transformers import ErnieForMaskedLM
import paddle
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from src.rna_ernie import BatchConverter  
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
        model = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_values=True)
        model.fit(emb, sc)
        preds = model.predict(emb)
        corr, pval = spearmanr(preds, sc)
        avg_emb = preds
    else:
        avg_emb = embeddings[:,0]
        corr, pval = spearmanr(avg_emb, scores)
    
    return corr, pval, avg_emb
# Function Definitions
def construct_file_path(directory, filename, extension=".csv"):
    """Constructs a full file path given a directory, filename, and extension."""
    return f"{directory}{filename}{extension}"

def calculate_mutation_score(mutation, sequence, token_probs, alphabet, offset):

    """Calculates the mutation score for a given mutation."""
    score = 0
    for mut in mutation.split(","):
        mut = mut.strip()
        wt, idx, mt = mut[0], int(mut[1:-1]) - offset, mut[-1]
        
        assert wt=="N" or sequence[idx] == wt , "Mismatch between sequence and wildtype."

        wt_encoded, mt_encoded = alphabet[wt], alphabet[mt]
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score

def main(args):
    with open(f"{args.output_directory}/summary.txt", "w") as f:
                    f.write("Experiment_ID\tCorrelation\n")
    # Load Model
    language_model = ErnieForMaskedLM.from_pretrained(args.model_checkpoint)
    language_model.eval()

    # Initialize BatchConverter
    batch_converter = BatchConverter(
        k_mer=1,
        vocab_path=args.vocab_path,
        batch_size=256,
        max_seq_len=512,
    )

    # Load Reference Data
    reference_data = pd.read_csv(args.reference_sequences)
    reference_data['path'] = reference_data['DMS_ID'].apply(lambda x: construct_file_path(args.dms_directory, x))
    reference_data['RAW_CONSTRUCT_SEQ'] = reference_data['RAW_CONSTRUCT_SEQ'].str.replace('U', 'T')
    alphabet = batch_converter.tokenizer.vocab.token_to_idx

    # Processing Loop
    for idx, row in reference_data.iterrows():
        experiment_id = row['DMS_ID']
        wildtype_sequence = row['RAW_CONSTRUCT_SEQ'].upper()
        mutation_file_path = row['path']
        if len(wildtype_sequence) > 512:
            print(f"Skipping {experiment_id} due to sequence length > 512.")
            continue
        # Prepare Data for Model
        data_batch = [(experiment_id, wildtype_sequence)]

        for _, _, input_ids in batch_converter(data_batch):
            with paddle.no_grad():
                logits = language_model(input_ids).detach()
                logits_tensor = torch.tensor(logits.numpy())  # Convert Paddle tensor to PyTorch
                probabilities = F.softmax(logits_tensor, dim=2)
            try:
            # Load Mutations and Format
                mutation_data = pd.read_csv(mutation_file_path).dropna(subset=['mutant'])
            except FileNotFoundError:
                print(f"File not found: {mutation_file_path}. Skipping this dataset.")
                continue
            mutation_data['mutant'] = mutation_data['mutant'].str.replace('U', 'T')
            mutation_list = mutation_data['mutant'].tolist()
            print(f"Processing dataset {experiment_id}")
            oft = 1
            # Score Mutations
            mutation_scores = [
                calculate_mutation_score(mut, wildtype_sequence, probabilities, alphabet, offset=oft)
                for mut in mutation_list
            ]

            # Append Scores and Save
            mutation_data['Mutation_Scores'] = mutation_scores
            output_path = construct_file_path(args.output_directory, experiment_id)
            mutation_data.to_csv(output_path, index=False)
            print(f"Scores saved to {output_path}")
            #calculate correlation 
            corr = mutation_data['Mutation_Scores'].corr(mutation_data['DMS_score'])
            print(f"Correlation for {experiment_id}: {corr:.4f}")
            # Save correlation to a file
            import os
            if not os.path.exists(args.output_directory):
                os.makedirs(args.output_directory)
            if not os.path.exists(f"{args.output_directory}/summary.txt"):
                with open(f"{args.output_directory}/summary.txt", "w") as f:
                    f.write("Experiment_ID\tCorrelation\n")
            with open(f"{args.output_directory}/summary.txt", "a") as f:
                f.write(f"{experiment_id}\t{corr:.4f}\n")

def score_few_shot(args):
    with open(f"{args.output_directory}/summary.txt", "w") as f:
                    f.write("Experiment_ID\tCorrelation\n")
    # Load Model
    language_model = ErnieForMaskedLM.from_pretrained(args.model_checkpoint)
    language_model.eval()

    # Initialize BatchConverter
    batch_converter = BatchConverter(
        k_mer=1,
        vocab_path=args.vocab_path,
        batch_size=256,
        max_seq_len=512,
    )

    # Load Reference Data
    reference_data = pd.read_csv(args.reference_sequences)
    reference_data['path'] = reference_data['DMS_ID'].apply(lambda x: construct_file_path(args.dms_directory, x))
    reference_data['RAW_CONSTRUCT_SEQ'] = reference_data['RAW_CONSTRUCT_SEQ'].str.replace('U', 'T')
    alphabet = batch_converter.tokenizer.vocab.token_to_idx
    valid_seqlen = len(reference_data['RAW_CONSTRUCT_SEQ'].iloc[0])
    # Processing Loop
    for idx, row in reference_data.iterrows():
        experiment_id = row['DMS_ID']
        wildtype_sequence = row['RAW_CONSTRUCT_SEQ'].upper()
        mutation_file_path = row['path']
        if len(wildtype_sequence) > 512:
            print(f"Skipping {experiment_id} due to sequence length > 512.")
            continue
        # Prepare Data for Model
        df = pd.read_csv(mutation_file_path)
        data_batch = [(mut,seq) for mut, seq in zip(df['mutant'], df['sequence'])]
        scores = []
        for _, _, input_ids in batch_converter(data_batch):
            with paddle.no_grad():
                embeddings = language_model(input_ids).detach()
                embeddings = torch.tensor(embeddings.numpy())  # Convert Paddle tensor to PyTorch
                embeddings = F.softmax(embeddings, dim=1)
                print("Shape of embeddings:", embeddings.shape)
                embeddings = embeddings[:, 0:valid_seqlen + 1, :]  # Exclude CLS token and pad to valid sequence length
                embeddings = embeddings.mean(dim=1)
                scores.append(embeddings.cpu().numpy())
        scores = np.vstack(scores)  # Stack all scores into a single array
        print("Shape of scores:", scores.shape)
        corr, pval, pred_scores = evaluate(scores, df["DMS_score"].values, cv=False,
                                           few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)
        try:
            df['predicted_scores'] = pred_scores
        except ValueError:
            print(f"Error: Predicted scores shape {pred_scores.shape} does not match DataFrame length {len(df)}. Skipping {experiment_id}.")
            continue
        output_path = construct_file_path(args.output_directory, experiment_id)
        df.to_csv(output_path, index=False)
        print(f"Scores saved to {output_path}")    
        print(f"Correlation for {experiment_id}: {corr:.4f}")
        # Save correlation to a file
        import os
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
        if not os.path.exists(f"{args.output_directory}/summary.txt"):
            with open(f"{args.output_directory}/summary.txt", "w") as f:
                f.write("Experiment_ID,Correlation,P-Value\n")
        with open(f"{args.output_directory}/summary.txt", "a") as f:
            f.write(f"{experiment_id},{corr:.4f},{pval:.4f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mutation scores using an RNA language model.")
    parser.add_argument("--reference_sequences", type=str, required=True, help="Path to the reference sequences CSV file.")
    parser.add_argument("--dms_directory", type=str, required=True, help="Directory containing DMS files.")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--few_shot_k", type=int, default=None, help="Number of few-shot examples to use. If set, will run few-shot evaluation.")
    parser.add_argument("--few_shot_repeat", type=int, default=5, help="Number of times to repeat the few-shot evaluation. Default is 5.")
    args = parser.parse_args()
    if args.few_shot_k is not None and args.few_shot_repeat is not None:
        score_few_shot(args)
    else:
        main(args)

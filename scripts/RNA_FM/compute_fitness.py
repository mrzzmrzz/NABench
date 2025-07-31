import os
import torch
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
def create_parser():
    parser = argparse.ArgumentParser(description='Label an RNA mutation dataset with zero-shot predictions from RNA-FM')
    parser.add_argument("--model_location", type=str, help="RNA-FM model directory")
    parser.add_argument("--reference_sequences", type=str, help="CSV file with reference sequences")
    parser.add_argument("--dms_directory", type=str, help="Directory of mutational datasets")
    parser.add_argument("--output_directory", type=str, help="Directory to save scored fitness files")
    parser.add_argument("--scoring-strategy", type=str, default="masked-marginals", choices=["wt-marginals", "masked-marginals"], help="Scoring strategy")
    parser.add_argument("--few_shot_k", type=int, default=None, help="Number of samples for few-shot evaluation")
    parser.add_argument("--few_shot_repeat", type=int, default=5, help="Number of repeats for few-shot evaluation")
    parser.add_argument("--cv", action='store_true', help="Use cross-validation for evaluation")
    parser.add_argument("--cache_dir", type=str, default="/data4/marunze/rnafm/cache/", help="Directory to cache embeddings (optional, for large datasets)")
    return parser
def evaluate(embeddings: np.ndarray, scores: np.ndarray, cv=False, few_shot_k=None, few_shot_repeat=5, seed=42):
    np.random.seed(seed)
    
    mask = ~np.isnan(scores)
    if not mask.all():
        num_nan = len(scores) - mask.sum()
        print(f"Warning: {num_nan} samples have NaN scores and will be excluded from evaluation")
    print("Shape of embeddings:", embeddings.shape)
    emb = embeddings[mask]
    sc = scores[mask]
    
    # Few-shot 模式
    if few_shot_k is not None:
        print(f"Running few-shot evaluation with k={few_shot_k}, repeated {few_shot_repeat} times")
        corrs = []
        best_model = None
        for r in range(few_shot_repeat):
            print("Length of sc:", len(sc))
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
            model = RidgeCV(alphas=np.logspace(-3, 3, 7), store_cv_values=True)
            model.fit(emb[train_index], sc[train_index])
            preds[test_index] = model.predict(emb[test_index])
        corr, pval = spearmanr(preds, sc)
        avg_emb = preds
    else:
        avg_emb = embeddings[:,0]
        mask = ~np.isnan(avg_emb) & ~np.isnan(scores)
        avg_emb = avg_emb[mask]
        scores = scores[mask]
        corr, pval = spearmanr(avg_emb, scores)
    
    return corr, pval, avg_emb

def label_row_wt(row, full_sequence, full_token_probs, alphabet, offset_idx, assay_name, model, max_len=1024): # scoring with wt-marginals strategy
    seq_len = len(full_sequence)
    device = next(model.parameters()).device

    # Parse all mutations in the row
    try:
        mutations = [m.strip() for m in row.split(",")]
        mutation_info = [(m[0], int(m[1:-1]) - offset_idx, m[-1]) for m in mutations]
    except Exception as parse_error:
        print(f"[EXCEPTION] Failed to parse mutation row '{row}' in assay '{assay_name}'")
        print(parse_error)
        return np.nan

    try:
        # Sanity check: WT matches sequence
        for wt, idx, mt in mutation_info:
            if idx < 0 or idx > len(full_sequence):
                raise IndexError(
                    f"\n[ERROR] Mutation index {idx} out of bounds for sequence length {len(full_sequence)} in assay: {assay_name}"
                )
            if idx == len(full_sequence) and wt in {"N", "-"} and mt not in {"N", "-"}:
                full_sequence += mt
                continue
            if full_sequence[idx] != wt and wt not in {"N", "-"}:
                raise ValueError(
                    f"\n[ERROR] Wildtype base mismatch in assay: {assay_name}\n"
                    f"  Mutation: {wt}{idx + offset_idx}{mt}\n"
                    f"  Expected WT: {wt} at position {idx}\n"
                    f"  Actual base in sequence: {full_sequence[idx]}\n"
                    f"  Full sequence: {full_sequence}\n"
                    f"  Problematic mutation row: {row}\n"
                )
            # if wt in {"N", "-"} and mt not in {"N", "-"}:
                # assert full_sequence[idx] in {"N", "-", mt}, \
                #     f"[ERROR] Invalid mutation {wt}{idx + offset_idx}{mt} at position {idx} in sequence {full_sequence}"
        # If the full sequence fits in the model
        if seq_len + 2 <= max_len and full_token_probs is not None:
            for wt, idx, mt in mutation_info:
                wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
                seq_encoded = [alphabet.get_idx(c) for c in full_sequence]
                mut_token_probs = full_token_probs.squeeze(0)  # Remove batch dimension
                mut_token_probs = mut_token_probs[np.arange(1,mut_token_probs.shape[0]-1), seq_encoded]
                mut_token_probs[idx] = full_token_probs[0, idx, mt_encoded]
                return mut_token_probs.cpu().numpy()
        else:
            # Use windowing for sequences longer than maximum input length centered on mutations
            idxs = [idx for _, idx, _ in mutation_info]
            min_idx, max_idx = min(idxs), max(idxs)
            window_size = max_len
            window_half = (window_size - 2) // 2  # excluding BOS and EOS

            center = (min_idx + max_idx) // 2
            start_pos = max(0, center - window_half)
            end_pos = min(seq_len, start_pos + (window_size - 2))
            if end_pos == seq_len:
                start_pos = max(0, seq_len - (window_size - 2))
                end_pos = seq_len

            if not all(start_pos <= idx < end_pos for idx in idxs):
                raise ValueError(
                    f"[ERROR] Window does not include all mutations.\n"
                    f"  Mutations: {mutation_info}\n"
                    f"  Window range: {start_pos}-{end_pos - 1}\n"
                    f"  Assay: {assay_name}"
                )

            window_seq = full_sequence[start_pos:end_pos]
            data = [(assay_name, window_seq)]
            batch_labels, batch_strs, batch_tokens = alphabet.get_batch_converter()(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)

            for wt, idx, mt in mutation_info:
                window_idx = idx - start_pos
                if not (0 <= window_idx < end_pos - start_pos):
                    raise IndexError(
                        f"[IndexError] window_idx={window_idx} is out of bounds for mutation {wt}{idx+offset_idx}{mt} "
                        f"in window [{start_pos}, {end_pos})"
                    )
                wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
                print((token_probs[0, 1 + window_idx, mt_encoded] - token_probs[0, 1 + window_idx, wt_encoded]).item())
                score.append(
                    (token_probs[0, 1 + window_idx, mt_encoded] - token_probs[0, 1 + window_idx, wt_encoded]).item()
                )

    except Exception as e:
        print(f"[EXCEPTION] Assay '{assay_name}', Mutation row: {row}")
        raise e
        print(e)
        return np.nan

    return np.array(score)


def compute_scores_masked(model, alphabet, batch_converter, reference, args, batch_size=32): # scoring with masked-marginals strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    max_len = 1024
    total_score = 0.0
    for _, row in reference.iterrows():
        name = row['DMS_ID']
        wt_rna = row['RAW_CONSTRUCT_SEQ'].upper().replace('T', 'U')
        wt_rna = clean_sequence(wt_rna, name)
        csv_path = row['PATH']

        try:
            mut_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read mutation CSV for {name}: {e}")
            continue

        mut_df = mut_df[~mut_df['mutant'].isna()].copy()
        mut_df['mutant'] = (
            mut_df['mutant'].astype(str).str.upper().str.replace('T', 'U').str.replace('N', '-').str.strip()
        )

        mut_list = mut_df['mutant'].tolist()
        mutation_info = []

        mask_token = alphabet.mask_idx

        for i, mut in enumerate(tqdm(mut_list, desc=f"{name} mutations")):
            try:
                mutations = [m.strip() for m in mut.split(",")]
                idxs = []
                for m in mutations:
                    wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
                    if idx >= len(wt_rna) or (wt_rna[idx] != wt and wt != "N" and wt != "-"):
                        raise ValueError(f"[SKIP] WT mismatch at {m} in {name}, current length is {len(wt_rna)}")
                    idxs.append(idx)

                if len(wt_rna) + 2 <= max_len:
                    window_seq = wt_rna
                    start_pos = 0
                else:
                    min_idx, max_idx = min(idxs), max(idxs)
                    center = (min_idx + max_idx) // 2
                    window_half = (max_len - 2) // 2
                    start_pos = max(0, center - window_half)
                    end_pos = min(len(wt_rna), start_pos + (max_len - 2))
                    if end_pos == len(wt_rna):
                        start_pos = max(0, len(wt_rna) - (max_len - 2))
                    window_seq = wt_rna[start_pos:end_pos]
                    if not all(start_pos <= idx < end_pos for idx in idxs):
                        raise ValueError(f"[SKIP] Not all mutation sites are within model window for {name}: {mut}")

                data = [(name, window_seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                modified_token = batch_tokens.clone()
                for m in mutations:
                    wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
                    window_idx = idx - start_pos
                    modified_token[0, window_idx + 1] = mask_token

                mutation_info.append((i, mutations, modified_token.clone(), start_pos))
            except Exception as e:
                print(f"[ERROR] Skipping mutation '{mut}' in assay '{name}': {e}")
                mutation_info.append((i, None, None, None))

        valid_entries = [entry for entry in mutation_info if entry[1] is not None]
        scores = [np.nan] * len(mut_list)

        for i in range(0, len(valid_entries), batch_size):
            batch = valid_entries[i:i + batch_size]
            token_batch = torch.cat([x[2] for x in batch], dim=0).to(device)

            with torch.no_grad():
                logits = model(token_batch)["logits"]
                log_probs = torch.log_softmax(logits, dim=-1)

            for j, (row_idx, mutations, _, start_pos) in enumerate(batch):
                try:
                    score = 0
                    for m in mutations:
                        wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
                        window_idx = idx - start_pos
                        wt_idx = alphabet.get_idx(wt)
                        mt_idx = alphabet.get_idx(mt)
                        score += (log_probs[j, window_idx + 1, mt_idx] - log_probs[j, window_idx + 1, wt_idx]).item()
                    scores[row_idx] = score
                except Exception as e:
                    print(f"[ERROR] Failed scoring mutation {mut_list[row_idx]} in assay '{name}': {e}")
                    scores[row_idx] = np.nan

        mut_df['RNA_FM_score'] = scores
        score_path = f'{args.output_directory}/{name}.csv'
        #calculate correlation with DMS scores
        if 'DMS_score' in mut_df.columns:
            correlation = mut_df['RNA_FM_scores'].corr(mut_df['DMS_score'])
            print(f"[SUMMARY] Correlation with DMS scores for assay '{name}': {correlation:.4f}")
            summary_path = f'{args.output_directory}/summary.txt'

            if not os.path.exists(summary_path):
                with open(summary_path, 'w') as summary_file:
                    summary_file.write("Assay\tCorrelation\n")
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"{name}\t{correlation:.4f}\n")
        else:
            print(f"[WARNING] No DMS scores found for assay '{name}'")

        try:
            mut_df.to_csv(score_path, index=False, encoding='utf-8')
            print("File saved at", score_path)
        except Exception as e:
            print(f"[ERROR] Failed to save scored file for {name}: {e}")


def clean_sequence(seq, assay_name): # replaces invalid bases with N
    cleaned = ''.join([c if c in {'A', 'U', 'C', 'G'} else 'N' for c in seq])
    if cleaned != seq:
        print(f"[WARNING] Cleaned WT sequence for assay '{assay_name}' due to invalid characters.")
        print(f"Original: {seq}")
        print(f"Cleaned:  {cleaned}")
    return cleaned

def compute_scores_wt(model, alphabet, batch_converter, reference, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for index, row in reference.iterrows():
        name = row['DMS_ID']
        wt_rna = row['RAW_CONSTRUCT_SEQ'].upper().replace('T', 'U')
        wt_rna_clean = clean_sequence(wt_rna, name)
        csv_path = row['PATH']
        seq_len = len(wt_rna)

        try:
            if seq_len + 2 <= 1024:
                # Full sequence fits
                data = [(name, wt_rna_clean)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                max_token_idx = batch_tokens.max().item()
                if max_token_idx >= len(alphabet):
                    print(f"[ERROR] Token index {max_token_idx} out of range (alphabet size: {len(alphabet)}) for assay '{name}'")
                    continue

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[12])
                    token_probs = torch.log_softmax(results["logits"], dim=-1)
                    print("Shape of token_probs:", token_probs.shape)
            else:
                token_probs = None  # Will trigger windowing

        except Exception as model_error:
            print(f"\n[ERROR] Model inference failed for assay '{name}'")
            print(f"  Exception type : {type(model_error).__name__}")
            print(f"  Error message  : {model_error}")
            print(f"  Sequence length: {seq_len}")
            continue

        try:
            mut_df = pd.read_csv(csv_path)
            mut_df = mut_df[~mut_df['mutant'].isna()]
            mut_df['mutant'] = (
                mut_df['mutant']
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace("T", "U")
                .str.replace(" ", "")
            )

            valid_mask = mut_df['mutant'].str.match(r'^([AUCGN][0-9]+[AUCG])(,[AUCGN][0-9]+[AUCG])*$')
            if not valid_mask.all():
                print(f"[WARNING] Invalid mutation format found in assay '{name}':")
                print(mut_df[~valid_mask])
            mut_df = mut_df[valid_mask]
            mut_list = mut_df['mutant'].to_list()

            scores = []
            failed_mutations = []

            for mut in tqdm(mut_list, desc=f"Scoring mutations ({name})"):
                try:
                    score = label_row_wt(
                        row=mut,
                        full_sequence=wt_rna,
                        full_token_probs=token_probs,
                        alphabet=alphabet,
                        offset_idx=1,
                        assay_name=name,
                        model=model,
                        max_len=1024
                    )
                    scores.append(score)
                except Exception as mutation_error:
                    print(f"[ERROR] Failed to score mutation '{mut}' in assay '{name}':")
                    print(mutation_error)
                    failed_mutations.append(mut)
                    scores.append(np.nan)
                    raise mutation_error

            # if failed_mutations:
            #     print(f"\n[SUMMARY] Assay '{name}' had {len(failed_mutations)} failed mutations:")
            #     for fm in failed_mutations:
            #         print(f"  ➤ {fm}")
            #     print("-" * 60)
            print("Shape of scores:", scores[0].shape)
            print("Shape of scores:", scores[1].shape)
            print("Shape of scores:", scores[2].shape)
            scores = np.array(scores) 
            # mask = ~np.isnan(scores).any(axis=1) 
            # if not mask.all():
            #     num_nan = len(scores) - mask.sum()
            #     print(f"[WARNING] {num_nan} samples have NaN scores and will be excluded from evaluation")
            # scores = scores[mask]
            # mut_df = mut_df[mask].reset_index(drop=True)
            corr,pval,pred_scores = evaluate(scores, mut_df['DMS_score'].values, cv=False,
                                        few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)
            mut_df['RNA_FM_scores'] = pred_scores
            score_path = f'{args.output_directory}/{name}.csv'
            mut_df.to_csv(score_path, index=False)
            print(f"File saved at {score_path}")

            # Calculate correlation with DMS scores
            print(f"[SUMMARY] Correlation with DMS scores for assay '{name}': {corr:.4f}")
            summary_path = f'{args.output_directory}/summary.txt'
            if not os.path.exists(summary_path):
                with open(summary_path, 'w') as summary_file:
                    summary_file.write("Assay,Correlation\n")
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"{name},{corr:.4f},{pval:.4f}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed during mutation processing for assay '{name}'")
            print(f"  Exception type : {type(e).__name__}")
            print(f"  Error message  : {e}")
            print("-" * 80)
            raise e

def score_variants(model, alphabet, batch_converter, reference, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for index, row in reference.iterrows():

        name = row['DMS_ID']
        cache_file = f"{args.cache_dir}/{name}_embeddings.npy"
        wt_rna = row['RAW_CONSTRUCT_SEQ'].upper().replace('T', 'U')
        wt_rna_clean = clean_sequence(wt_rna, name)
        csv_path = row['PATH']
        seq_len = len(wt_rna)
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            scores = np.load(cache_file)
        else:
            try:
                if seq_len + 2 <= 1024:
                    # Full sequence fits
                    data = [(name, wt_rna_clean)]
                    batch_labels, batch_strs, batch_tokens = batch_converter(data)
                    batch_tokens = batch_tokens.to(device)

                    max_token_idx = batch_tokens.max().item()
                    if max_token_idx >= len(alphabet):
                        print(f"[ERROR] Token index {max_token_idx} out of range (alphabet size: {len(alphabet)}) for assay '{name}'")
                        continue

                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[12])
                        token_probs = torch.log_softmax(results["logits"], dim=-1)
                        print("Shape of token_probs:", token_probs.shape)
                else:
                    token_probs = None  # Will trigger windowing

            except Exception as model_error:
                print(f"\n[ERROR] Model inference failed for assay '{name}'")
                print(f"  Exception type : {type(model_error).__name__}")
                print(f"  Error message  : {model_error}")
                print(f"  Sequence length: {seq_len}")
                continue

            try:
                mut_df = pd.read_csv(csv_path)
                mut_df = mut_df[~mut_df['mutant'].isna()]
                mut_df['mutant'] = (
                    mut_df['mutant']
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .str.replace("T", "U")
                    .str.replace(" ", "")
                )

                valid_mask = mut_df['mutant'].str.match(r'^([AUCGN][0-9]+[AUCG])(,[AUCGN][0-9]+[AUCG])*$')
                if not valid_mask.all():
                    print(f"[WARNING] Invalid mutation format found in assay '{name}':")
                    print(mut_df[~valid_mask])
                mut_df = mut_df[valid_mask]
                mut_list = mut_df['mutant'].to_list()

                scores = []
                failed_mutations = []
                sequences = mut_df["sequence"].tolist() 
                batch_size = 8
                for batch in tqdm(range(0, len(mut_list), batch_size), desc=f"Processing batches ({name})"):
                    try:
                        batch_idx = mut_list[batch:batch + batch_size]
                        batch_seqs = sequences[batch:batch + batch_size]
                        data = [(name,seq) for name,seq in zip(batch_idx, batch_seqs)]
                        _,_,tokens = batch_converter(data)
                        tokens = tokens.to(device)
                        results = model(tokens, repr_layers=[12])

                        token_embeddings = results["representations"][12]
                        # pooling on the last dimension (token dimension)
                        token_embeddings = token_embeddings.mean(dim=1)

                        scores.append(token_embeddings.detach().cpu().numpy())
                    except Exception as mutation_error:
                        raise mutation_error

                # if failed_mutations:
                #     print(f"\n[SUMMARY] Assay '{name}' had {len(failed_mutations)} failed mutations:")
                #     for fm in failed_mutations:
                #         print(f"  ➤ {fm}")
                #     print("-" * 60)
                scores = np.vstack(scores) 
                # Save the scores to a cache file
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file, scores)
                print(f"Scores saved to {cache_file}")
            # mask = ~np.isnan(scores).any(axis=1) 
            # if not mask.all():
            #     num_nan = len(scores) - mask.sum()
            #     print(f"[WARNING] {num_nan} samples have NaN scores and will be excluded from evaluation")
            # scores = scores[mask]
            # mut_df = mut_df[mask].reset_index(drop=True)
            except Exception as e:
                print(f"\n[ERROR] Failed to process mutations for assay '{name}'")
                print(f"  Exception type : {type(e).__name__}")
                print(f"  Error message  : {e}")
                print("-" * 80)
                continue
            mask = ~np.isnan(mut_df['DMS_score'].values)
            corr,pval,pred_scores = evaluate(scores, mut_df['DMS_score'].values, cv=args.cv,
                                        few_shot_k=args.few_shot_k, few_shot_repeat=args.few_shot_repeat)
            mut_df['RNA_FM_scores'] = pred_scores
            score_path = f'{args.output_directory}/{name}.csv'
            mut_df.to_csv(score_path, index=False)
            print(f"File saved at {score_path}")

            # Calculate correlation with DMS scores
            print(f"[SUMMARY] Correlation with DMS scores for assay '{name}': {corr:.4f}")
            summary_path = f'{args.output_directory}/summary.txt'
            if not os.path.exists(summary_path):
                with open(summary_path, 'w') as summary_file:
                    summary_file.write("Assay,Correlation\n")
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"{name},{corr:.4f},{pval:.4f}\n")

            

def main(args):

    sys.path.insert(1, args.model_location)
    sys.path.append("/home/ma_run_ze/lzm/rnagym/fitness/scripts/RNA_FM/RNA-FM")
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    summary_path = f'{args.output_directory}/summary.txt'
    
    with open(summary_path, 'w') as summary_file:
        summary_file.write("Assay\tCorrelation\n")
    reference = pd.read_csv(args.reference_sequences)
    reference['PATH'] = reference['DMS_ID'].apply(lambda x: f"{args.dms_directory}/{x}.csv")

    if args.scoring_strategy == "masked-marginals":
        compute_scores_masked(model, alphabet, batch_converter, reference, args)
    elif args.scoring_strategy == "wt-marginals":
        score_variants(model, alphabet, batch_converter, reference, args)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

from qaoa_gpt_src.custom_feather import CustomFeatherGraph as FeatherGraph
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
from collections import Counter
import random
import ast
from joblib import Parallel, delayed
from itertools import islice
from networkx import convert_node_labels_to_integers

tqdm.pandas()

def _open_df(fname):
    try:
        df = pd.read_csv(fname)
        df['worker_id'] = fname.stem
        return df
    except Exception as e:
        print(f'{e}  (file: {fname})')
        return None

def generate_tokens_hubo(results_fpath_str: str,
                          save_path_str: str,
                          config_path_temp: str,
                          n_nodes: int,
                          rounding_digits=2,
                          min_block_size=128,
                          max_block_size=256,
                          val_frac=0.1,
                          test_frac=0.1,
                          max_abs_param_val=10,
                          perform_coef_mod_range=True,
                          apply_sliding_window=True,
                          n_workers=1,
                          skip_only_qaoa_circ=False,
                          approx_ratio_thr=0.97,
                          debug_limit=0,
                          target_val_size=10,
                          verbose=True):
    results_fpath_list = [Path(p) for p in results_fpath_str.split(';')]
    save_path = Path(save_path_str)

    for p in results_fpath_list:
        assert p.exists() and p.is_dir(), f"Invalid path: {p}"

    if verbose:
        print("Loading ADAPT-QAOA results...")

    df_list_all = []
    for cur_path in results_fpath_list:
        flist = sorted(cur_path.joinpath('res').glob('*.csv'))
        if debug_limit:
            flist = flist[:debug_limit]
        chunk = Parallel(n_jobs=n_workers)(
            delayed(_open_df)(f)
            for f in tqdm(flist, desc=f'res ({cur_path.stem})'))
        df_list_all += chunk

    df_list = [df for df in df_list_all if df is not None]
    if verbose:
        print(f"  Loaded {len(df_list)} result files")

    full_run_df = pd.concat(df_list)
    full_run_df['prefix'] = full_run_df['worker_id'].apply(lambda x: x[:-8])

    if verbose:
        print("Loading HUBO metadata...")

    hubo_list_all = []
    for cur_path in results_fpath_list:
        flist = sorted(cur_path.joinpath('hubo').glob('*.csv'))
        if debug_limit:
            flist = flist[:debug_limit]
        chunk = Parallel(n_jobs=n_workers)(
            delayed(_open_df)(f)
            for f in tqdm(flist, desc=f'hubo ({cur_path.stem})'))
        hubo_list_all += chunk

    hubo_list = [df for df in hubo_list_all if df is not None]
    if verbose:
        print(f"  Loaded {len(hubo_list)} HUBO metadata files")

    full_run_hubo_df = pd.concat(hubo_list)
    full_run_hubo_df['prefix'] = full_run_hubo_df['worker_id'].apply(
        lambda x: x[:-5])

    full_run_hubo_df['linear_list'] = full_run_hubo_df['linear_json'].apply(
        lambda x: [tuple(e) for e in json.loads(x)])
    full_run_hubo_df['quad_list'] = full_run_hubo_df['quad_json'].apply(
        lambda x: [tuple(e) for e in json.loads(x)])
    full_run_hubo_df['cubic_list'] = full_run_hubo_df['cubic_json'].apply(
        lambda x: [tuple(e) for e in json.loads(x)])
    full_run_hubo_df['projected_edgelist_list'] = (
        full_run_hubo_df['projected_edgelist_json'].apply(
            lambda x: [tuple(e) for e in json.loads(x)]))

    if verbose:
        print("Merging ADAPT results with HUBO metadata...")

    combined_res_df = pd.merge(
        left=full_run_df,
        right=full_run_hubo_df,
        left_on=['prefix', 'graph_num'],
        right_on=['prefix', 'graph_num'],
    )

    combined_res_df['energy_list'] = combined_res_df['energy_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    combined_res_df['n_layers'] = combined_res_df['energy_list'].apply(len)

    combined_res_df['graph_id'] = (
        combined_res_df['prefix'] + '_^_' +
        combined_res_df['graph_num'].astype(str))

    combined_res_df['mixer_pool_index'] = combined_res_df[
        'mixer_pool_index'].apply(
            lambda x: [int(i) for i in ast.literal_eval(x)]
            if isinstance(x, str) else x)

    combined_res_df['only_qaoa_circ'] = combined_res_df[
        'mixer_pool_index'].progress_apply(
            lambda x: all(e == n_nodes + 1 for e in x))

    if combined_res_df.empty:
        raise ValueError(
            "Merge of ADAPT results and HUBO metadata produced zero rows.\n"
            "Check that the 'res/' and 'hubo/' CSVs in your results directory "
            "were generated together (same pid and timestamp).")
    combined_res_df['interaction_count'] = combined_res_df.apply(
        lambda row: len(row['linear_list']) + len(row['quad_list']) + len(row['cubic_list']),
        axis=1)

    if verbose:
        print("Computing FEATHER embeddings on projected graphs...")

    combined_unique_hubo_df = (
        combined_res_df[['graph_id', 'projected_edgelist_json']].drop_duplicates())

    def _make_weighted_nx(elist_0indexed):
        G = nx.Graph()
        G.add_weighted_edges_from(elist_0indexed)
        return G

    combined_unique_hubo_df['edgelist_py_list'] = (
        combined_unique_hubo_df['projected_edgelist_json'].progress_apply(
            lambda x: [(e[0] - 1, e[1] - 1, e[2]) for e in json.loads(x)]))

    combined_unique_hubo_df['graph_nx'] = (
        combined_unique_hubo_df['edgelist_py_list'].progress_apply(
            lambda x: convert_node_labels_to_integers(_make_weighted_nx(x))))

    combined_unique_hubo_w_idx_df = combined_unique_hubo_df.set_index('graph_id')
    graphs_nx_dict = combined_unique_hubo_w_idx_df['graph_nx'].to_dict()

    graphs_nx_filt_dict = {
        name: g for name, g in graphs_nx_dict.items()
        if g.number_of_nodes() == n_nodes
    }
    graphs_nx_filt_names = list(graphs_nx_filt_dict.keys())
    graphs_nx_filt_list = list(graphs_nx_filt_dict.values())

    emb_graph_idx_to_id_dict = {k: v for k, v in enumerate(graphs_nx_filt_names)}
    emb_graph_id_to_idx_dict = {v: k for k, v in enumerate(graphs_nx_filt_names)}

    def _get_feather_emb(g_list):
        feather_model = FeatherGraph(
            order=5, eval_points=25, theta_max=2.5, seed=42, pooling="mean")
        feather_model.fit(graphs=g_list)
        return feather_model.get_embedding()

    def _split_list(lst, n):
        it = iter(lst)
        return [list(islice(it, i))
                for i in [len(lst) // n + (1 if x < len(lst) % n else 0)
                          for x in range(n)]]

    def _embed_parallel(graphs_list, n_workers=n_workers):
        chunks = _split_list(graphs_list, n_workers)
        emb_list = Parallel(n_jobs=n_workers)(
            delayed(_get_feather_emb)(chunk) for chunk in chunks)
        return np.vstack(emb_list)

    feather_par_emb = _embed_parallel(graphs_nx_filt_list)
    feather_par_emb = feather_par_emb.round(rounding_digits)

    combined_res_df['has_emb'] = combined_res_df['graph_id'].apply(
        lambda x: x in emb_graph_id_to_idx_dict)

    if verbose:
        print("Filtering results...")

    combined_res_df['gamma_coef'] = combined_res_df['gamma_coef'].apply(
        lambda x: [float(c) for c in ast.literal_eval(x)]
        if isinstance(x, str) else x)

    combined_res_df['approx_ratio'] = pd.to_numeric(
        combined_res_df['approx_ratio'], errors='coerce')

    combined_res_filt_df = combined_res_df[
        (combined_res_df['gamma_coef'].apply(
            lambda x: all(abs(c) < max_abs_param_val for c in x))) &
        (combined_res_df['approx_ratio'] > approx_ratio_thr)
    ].copy()

    if skip_only_qaoa_circ:
        combined_res_filt_df = combined_res_filt_df[
            ~combined_res_filt_df['only_qaoa_circ']]

    combined_res_filt_df['beta_coef'] = combined_res_filt_df['beta_coef'].apply(
        lambda x: [float(c) for c in ast.literal_eval(x)]
        if isinstance(x, str) else x)

    combined_res_filt_df['gamma_coef'] = combined_res_filt_df['gamma_coef'].apply(
        lambda x: [float(c) for c in ast.literal_eval(x)]
        if isinstance(x, str) else x)

    combined_res_filt_df['mixer_pool_index'] = combined_res_filt_df[
        'mixer_pool_index'].apply(
            lambda x: [int(i) for i in ast.literal_eval(x)]
            if isinstance(x, str) else x)

    if verbose:
        print(f"  Rows after filtering: {len(combined_res_filt_df)}")

    if verbose:
        print("Building vocabulary...")

    tokens_list = []

    tokens_list += ['pad', 'bos', 'eos', 'new_layer_p', 'end_of_graph']

    all_linear_tuples = set()
    for row_list in combined_res_filt_df['linear_list']:
        for (i, _w) in row_list:
            all_linear_tuples.add((i,))
    tokens_list += list(all_linear_tuples)
    if verbose:
        print(f"  Linear tokens: {len(all_linear_tuples)}")

    all_quad_tuples = set()
    for row_list in combined_res_filt_df['quad_list']:
        for (i, j, _w) in row_list:
            all_quad_tuples.add((i, j))
    tokens_list += list(all_quad_tuples)
    if verbose:
        print(f"  Quadratic tokens: {len(all_quad_tuples)}")

    all_cubic_tuples = set()
    for row_list in combined_res_filt_df['cubic_list']:
        for (i, j, k, _w) in row_list:
            all_cubic_tuples.add((i, j, k))
    tokens_list += list(all_cubic_tuples)
    if verbose:
        print(f"  Cubic tokens: {len(all_cubic_tuples)}")

    n_steps = int((max_abs_param_val * 2 / (10 ** -rounding_digits)) + 1)
    all_coefs_set = set([
        round(c, rounding_digits)
        for c in np.linspace(-max_abs_param_val, max_abs_param_val, n_steps).tolist()
    ])
    tokens_list += list(all_coefs_set)
    if verbose:
        print(f"  Coefficient tokens: {len(all_coefs_set)}")

    ops_list = list(set([
        idx
        for idxs in combined_res_filt_df['mixer_pool_index']
        for idx in idxs
    ]))
    tokens_list += ops_list
    if verbose:
        print(f"  Operator pool tokens: {len(ops_list)}")
        print(f"  Total vocabulary size: {len(tokens_list)}")

    int_idx_to_token_dict = dict(enumerate(tokens_list))
    token_to_int_idx_dict = {v: k for k, v in int_idx_to_token_dict.items()}
    vocab_size = len(int_idx_to_token_dict)

    def arth_mod(a, b):
        result = a % b
        return result if a >= 0 else result - b

    def tokenize_row(row, coef_mod=True):
        tokens_seq = ['bos']

        for (i, w) in row['linear_list']:
            tokens_seq.append((i,))
            tokens_seq.append(round(w, rounding_digits))

        for (i, j, w) in row['quad_list']:
            tokens_seq.append((i, j))
            tokens_seq.append(round(w, rounding_digits))

        for (i, j, k, w) in row['cubic_list']:
            tokens_seq.append((i, j, k))
            tokens_seq.append(round(w, rounding_digits))

        tokens_seq.append('end_of_graph')

        for p in range(row['n_layers']):
            tokens_seq.append('new_layer_p')
            tokens_seq.append(row['mixer_pool_index'][p])

            cur_gamma = row['gamma_coef'][p]
            if -max_abs_param_val < cur_gamma < max_abs_param_val:
                tokens_seq.append(round(cur_gamma, rounding_digits))
            else:
                return None

            cur_beta = row['beta_coef'][p]
            if coef_mod:
                cur_beta = arth_mod(cur_beta, np.pi)
            if -max_abs_param_val < cur_beta < max_abs_param_val:
                tokens_seq.append(round(cur_beta, rounding_digits))
            else:
                return None

        tokens_seq.append('eos')
        return tokens_seq

    col_tok = f'token_seq_round_d{rounding_digits}'
    combined_res_filt_df[col_tok] = combined_res_filt_df.progress_apply(
        lambda row: tokenize_row(row, coef_mod=perform_coef_mod_range), axis=1)

    combined_res_tok_df = combined_res_filt_df.dropna(subset=[col_tok])

    col_int = f'token_int_seq_round_d{rounding_digits}'
    combined_res_tok_df[col_int] = combined_res_tok_df[col_tok].progress_apply(
        lambda seq: [token_to_int_idx_dict[t] for t in seq])

    if verbose:
        print("Creating train/val/test splits...")

    combined_res_tok_shf_df = combined_res_tok_df.sample(frac=1).reset_index(drop=True)

    graph_ids = combined_res_tok_shf_df['graph_id'].drop_duplicates().tolist()
    n_total = len(graph_ids)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val

    train_ids = set(graph_ids[:n_train])
    val_ids = set(graph_ids[n_train:n_train + n_val])
    test_ids = set(graph_ids[n_train + n_val:])

    combined_res_tok_shf_df['label'] = 'train'
    combined_res_tok_shf_df.loc[
        combined_res_tok_shf_df['graph_id'].isin(val_ids), 'label'] = 'val'
    combined_res_tok_shf_df.loc[
        combined_res_tok_shf_df['graph_id'].isin(test_ids), 'label'] = 'test'

    def pad_zeros(seq, target_len):
        pad_len = target_len - len(seq)
        return seq + [0] * pad_len if pad_len > 0 else seq

    def sliding_window(numbers, min_bs, max_bs):
        block_size = (random.randint(min_bs, max_bs)
                      if min_bs != max_bs else min_bs)
        if block_size >= len(numbers):
            w = numbers[:-1]
            ws = numbers[1:]
            return [[pad_zeros(w, max_bs), pad_zeros(ws, max_bs)]]
        result = [numbers[i:i + block_size]
                  for i in range(len(numbers) - block_size + 1)]
        return [[pad_zeros(x, max_bs), pad_zeros(y, max_bs)]
                for x, y in zip(result, result[1:])]

    if apply_sliding_window:
        if verbose:
            print("Applying sliding window...")
        col_sw = f'{col_int}_sw'
        combined_res_tok_shf_df[col_sw] = (
            combined_res_tok_shf_df[col_int].progress_apply(
                lambda x: sliding_window(x, min_block_size, max_block_size)))

    train_data = combined_res_tok_shf_df[combined_res_tok_shf_df['label'] == 'train']
    val_data = combined_res_tok_shf_df[combined_res_tok_shf_df['label'] == 'val']
    test_data = combined_res_tok_shf_df[combined_res_tok_shf_df['label'] == 'test']

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    def _collect_split(split_df, col_sw):
        tokens_flat = []
        graph_idx_flat = []
        for gid, seq_list in zip(split_df['graph_id'], split_df[col_sw]):
            if gid in emb_graph_id_to_idx_dict:
                tokens_flat += seq_list
                graph_idx_flat += [emb_graph_id_to_idx_dict[gid]] * len(seq_list)
        return np.array(tokens_flat, dtype=np.uint16), graph_idx_flat

    if apply_sliding_window:
        train_np, train_graph_idx = _collect_split(train_data, col_sw)
        val_np, val_graph_idx = _collect_split(val_data, col_sw)
        test_np, test_graph_idx = _collect_split(test_data, col_sw)

    if verbose:
        print(f"Saving to {save_path_str} ...")

    save_path.mkdir(parents=True, exist_ok=True)

    if apply_sliding_window:
        np.save(save_path / 'train.npy', train_np)
        np.save(save_path / 'val.npy', val_np)
        np.save(save_path / 'test.npy', test_np)

    combined_res_df.to_pickle(save_path / 'combined_res_df.pkl')
    combined_res_tok_shf_df.to_pickle(save_path / 'combined_res_tok_shf_df.pkl')

    emb_size = feather_par_emb.shape[1]
    np.save(save_path / f'feather_emb_d{emb_size}.npy', feather_par_emb)

    sample_per_bucket = max(1, int(target_val_size /
                                   max(1, len(combined_res_df['interaction_count'].drop_duplicates()))))
    val_data_sampled = (
        val_data[~val_data[col_tok].isna()]
        .groupby('interaction_count', group_keys=False)
        .filter(lambda g: len(g) > sample_per_bucket)
        .reset_index(drop=True)
    )
    val_data_sampled.to_pickle(save_path / 'combined_res_tok_shf_val_df.pkl')

    meta = {
        'vocab_size': vocab_size,
        'itos': int_idx_to_token_dict,
        'stoi': token_to_int_idx_dict,
        'train_data_graph_idx_list': train_graph_idx if apply_sliding_window else [],
        'val_data_graph_idx_list': val_graph_idx if apply_sliding_window else [],
        'test_data_graph_idx_list': test_graph_idx if apply_sliding_window else [],
        'emb_graph_id_to_idx_dict': emb_graph_id_to_idx_dict,
        'emb_graph_idx_to_id_dict': emb_graph_idx_to_id_dict,
    }
    pd.to_pickle(meta, save_path / 'meta.pkl')

    with open(f'{config_path_temp}/train_adapt_gpt_config_template.py') as f:
        config_template_str = f.read()

    pool_type = combined_res_df['pool_type'].iloc[0]
    dataset_name = save_path.stem
    config_str = config_template_str.format(
        out_dir=f'out-{dataset_name}',
        dataset=dataset_name,
        block_size=max_block_size,
        use_graph_emb="True",
        pool_type=pool_type,
        n_nodes=n_nodes,
        token_seq_round=f'token_seq_round_d{rounding_digits}',
    )
    with open(save_path / 'train_adapt_gpt_config.py', 'w') as f:
        f.write(config_str)

    if verbose:
        print(f"Done! Data saved to {save_path.absolute()}")

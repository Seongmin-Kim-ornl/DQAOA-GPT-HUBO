import os, sys, torch, subprocess, json, random, cudaq

import traceback
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
from itertools import islice

from gurobipy import Model, GRB
import gurobipy as gb

from qaoa_gpt_src.custom_feather import CustomFeatherGraph as FeatherGraph
from joblib import Parallel, delayed

from hamiltonian_graph import term_coefficients, term_words, max_cut_ham
from adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double

os.environ["CUDA_QUANTUM_OPTIMIZATION_LEVEL"] = "0"
os.environ["CUDAQ_DISABLE_LOOP_UNROLL"] = "1"
os.environ["IONQ_API_KEY"] = "yRN1cxbzPTR94Fk84rZ4hZVthUfYV6RV"

cudaq.set_target("nvidia", option="fp64")

def generate_circ_from_df(
    test_run_df,
    graph_emb_np,
    emb_graph_id_to_idx_dict,
    meta,
    model,
    device,
    ctx,
    n_samples_per_batch,
    num_samples=5,
    max_new_tokens=200,
    temperature=0.1,
    top_k=200,
    token_seq_col='token_seq_round_d2',
    normalize_weights_flag=False,
    emb_dtype=torch.bfloat16,
):

    if graph_emb_np is not None and emb_graph_id_to_idx_dict is not None:
        gemb_flag = True
    else:
        gemb_flag = False

    stoi, itos = meta['stoi'], meta['itos']
    _numeric_keys = sorted([k for k in stoi if isinstance(k, (int, float))])
    _numeric_arr  = np.array(_numeric_keys) if _numeric_keys else None
    def _safe_lookup(c):
        if c in stoi:
            return stoi[c]
        if _numeric_arr is not None and isinstance(c, (int, float)):
            nearest = _numeric_keys[int(np.argmin(np.abs(_numeric_arr - c)))]
            return stoi[nearest]
        raise KeyError(c)
    encode = lambda s: [_safe_lookup(c) for c in s]
    decode = lambda l: [itos[i] for i in l]

    n_edges_to_count_dict = test_run_df['edgelist_list_len'].value_counts(
    ).to_dict()

    adapt_gpt_out_list_dict = defaultdict(list)
    x_list_dict = defaultdict(list)
    graph_emb_dict = defaultdict(list)
    y_dict = dict()

    for n_edges, n_graphs in n_edges_to_count_dict.items():
        cur_test_run_df = test_run_df[test_run_df['edgelist_list_len'] ==
                                      n_edges]

        for row_idx, graph_df_row in cur_test_run_df.iterrows():
            start, adapt_seq = extract_graph(graph_df_row[token_seq_col])
            start_ids = encode(start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None,
                                                                          ...])
            x_list_dict[n_edges].append(x)

            if gemb_flag:
                cur_graph_idx = emb_graph_id_to_idx_dict[
                    graph_df_row['graph_id']]
                graph_emb_dict[n_edges].append(
                    torch.tensor(graph_emb_np[cur_graph_idx],
                                 dtype=emb_dtype,
                                 device=device))

            adapt_gpt_out_dict = dict()
            adapt_gpt_out_dict['graph'] = start[1:-1]
            adapt_gpt_out_dict['n_edges'] = graph_df_row['edgelist_list_len']
            adapt_gpt_out_dict['q_circuits'] = []
            adapt_gpt_out_dict['adapt_circuit'] = adapt_seq
            adapt_gpt_out_dict['adapt_full_ar'] = graph_df_row['approx_ratio']
            adapt_gpt_out_dict['graph_prefix'] = graph_df_row['graph_id']
            if 'true_energy' in graph_df_row:
                adapt_gpt_out_dict['true_energy'] = graph_df_row['true_energy']
            if 'energy_gurobi' in graph_df_row:
                adapt_gpt_out_dict['energy_gurobi'] = graph_df_row[
                    'energy_gurobi']
            adapt_gpt_out_dict['label'] = graph_df_row['label']
            adapt_gpt_out_list_dict[n_edges].append(adapt_gpt_out_dict)

        cur_batch_torch = torch.vstack(x_list_dict[n_edges])

        if gemb_flag:
            cur_emb_batch_torch = torch.vstack(graph_emb_dict[n_edges])

        total_samples = cur_batch_torch.size(0)
        n_batches = (total_samples + n_samples_per_batch -
                     1) // n_samples_per_batch  # Ensure ceiling division

        y_list = []

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * n_samples_per_batch
                end_idx = min((i + 1) * n_samples_per_batch, total_samples)

                mini_batch = cur_batch_torch[start_idx:end_idx]
                mini_batch_repeated = mini_batch.repeat(
                    num_samples, 1)  # Repeat the mini-batch for num_samples

                if gemb_flag:
                    mini_emb_batch = cur_emb_batch_torch[start_idx:end_idx]
                    mini_emb_batch_repeated = mini_emb_batch.repeat(
                        num_samples, 1)  # Repeat the mini-batch for num_samples

                with ctx:
                    if gemb_flag:
                        y = model.generate(mini_batch_repeated,
                                           mini_emb_batch_repeated,
                                           max_new_tokens,
                                           temperature=temperature,
                                           top_k=top_k)
                    else:
                        y = model.generate(
                            mini_batch_repeated,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k)

                y_list.append(y.detach().cpu())

        y_dict[n_edges] = torch.cat(y_list, dim=0)

    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        cur_full_y_tensor = y_dict[n_edges]

        for graph_idx in range(len(cur_adapt_gpt_out_list)):

            cur_y_tensor = cur_full_y_tensor[
                graph_idx::len(cur_adapt_gpt_out_list)]

            for k in range(num_samples):
                cur_gen_result = decode(cur_y_tensor[k].tolist())
                cur_circ = []
                circ_flag = 0
                for idx, tok in enumerate(cur_gen_result):
                    if tok == 'end_of_graph':
                        circ_flag = 1
                    if circ_flag:
                        cur_circ.append(tok)
                    if tok == 'eos':
                        break
                cur_adapt_gpt_out_list[graph_idx]['q_circuits'].append(
                    cur_circ[1:-1])

        adapt_gpt_test_samples_list = []
        for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
            adapt_gpt_test_samples_list += cur_adapt_gpt_out_list

    for idx in range(len(adapt_gpt_test_samples_list)):
        q_circ_filt_list = []
        for circ in adapt_gpt_test_samples_list[idx]['q_circuits']:
            filt_flag = circ_sanity_check(circ)
            q_circ_filt_list.append(circ)
        adapt_gpt_test_samples_list[idx]['q_circuits'] = q_circ_filt_list

    adapt_gpt_test_samples_list[idx]['q_circuits'] = q_circ_filt_list

    for gr_dict in adapt_gpt_test_samples_list:
        graph_py_list = []

        graph_edges_list = gr_dict['graph'][::2]
        graph_weights_list = gr_dict['graph'][1::2]

        if normalize_weights_flag:
            graph_w_norm = sum(graph_weights_list)
        else:
            graph_w_norm = 1.0

        for edge_idx, edge in enumerate(graph_edges_list):
            cur_edge = list(edge)
            cur_edge += [graph_weights_list[edge_idx] / graph_w_norm]
            graph_py_list.append(cur_edge)

        gr_dict['graph_w_py'] = graph_py_list
        gr_dict['graph_weight_norm'] = graph_w_norm

    adapt_gpt_test_samples_filt_list = []

    for rec in adapt_gpt_test_samples_list:
        pos_flag = 1

        if pos_flag:
            adapt_gpt_test_samples_filt_list.append(rec)

    adapt_gpt_test_samples_df = pd.DataFrame(adapt_gpt_test_samples_filt_list)

    return adapt_gpt_test_samples_df

def prepare_model_input(
    graphs_container,
    n_nodes,
    calculate_classic_maxcut=True,
    n_workers_feather=1,
):

    if type(graphs_container) == list:
        graphs_edgelist_list_dict = {
            f'graph_{i}': g for i, g in enumerate(graphs_container)
        }
    elif type(graphs_container) == dict:
        graphs_edgelist_list_dict = graphs_container
    else:
        raise ValueError(
            "Only list or dict containers are supported for input graphs!")

    graphs_nx_dict = defaultdict(dict)

    for name, nx_graph in graphs_edgelist_list_dict.items():
        nx_elist_dict = nx_to_elist(nx_graph)

        graphs_nx_dict[name]['elist'] = nx_elist_dict['elist']
        graphs_nx_dict[name]['n_nodes'] = nx_elist_dict['n_nodes']
        if calculate_classic_maxcut:
            graphs_nx_dict[name]['energy_gurobi'] = gurobi_max_cut_val_from_nx(
                nx_graph)

    graphs_nx_df = pd.DataFrame(graphs_nx_dict).T.reset_index(names='graph_id')
    graphs_nx_df['token_seq_round_d2'] = graphs_nx_df['elist'].apply(
        seq_tokenize_graph)
    graphs_nx_df['edgelist_list_len'] = graphs_nx_df['elist'].apply(len)
    graphs_nx_df['approx_ratio'] = None
    graphs_nx_df['label'] = 'test_interactive'
    graphs_nx_df['edgelist_json'] = graphs_nx_df['elist'].apply(
        lambda x: json.dumps(x))

    feather_par_emb, emb_graph_idx_to_id_dict = get_feather_emb(
        graphs_nx_df,
        n_workers=n_workers_feather,
        n_nodes=n_nodes,
    )

    emb_graph_id_to_idx_dict = {
        v: k for k, v in emb_graph_idx_to_id_dict.items()
    }

    graphs_nx_df['has_emb'] = graphs_nx_df['graph_id'].apply(
        lambda x: True if x in emb_graph_id_to_idx_dict else False)

    graphs_nx_df = graphs_nx_df[graphs_nx_df['has_emb']]

    graphs_nx_df['graph_id'].apply(lambda x: x[:2]).value_counts()

    return graphs_nx_df, feather_par_emb, emb_graph_id_to_idx_dict

def extract_graph(token_seq):
    graph_seq = []
    for idx, tok in enumerate(token_seq):
        graph_seq.append(tok)
        if tok == 'end_of_graph':
            break
    adapt_seq = token_seq[idx + 1:-1]
    return graph_seq, adapt_seq

def circ_sanity_check(cur_q_circ):
    lr_sep_list = cur_q_circ[0::4]
    op_idx_list = cur_q_circ[1::4]
    num_vals = cur_q_circ[2::4] + cur_q_circ[3::4]

    if any([type(el) != int for el in op_idx_list]):
        return False
    if any([type(el) != str for el in lr_sep_list]):
        return False
    if len(cur_q_circ) % 4:
        return False
    return True

def generate_circ_from_df(
    test_run_df,
    graph_emb_np,
    emb_graph_id_to_idx_dict,
    meta,
    model,
    device,
    ctx,
    n_samples_per_batch,
    num_samples=5,
    max_new_tokens=200,
    temperature=0.1,
    top_k=200,
    token_seq_col='token_seq_round_d2',
    normalize_weights_flag=False,
    emb_dtype=torch.bfloat16,
):
    if graph_emb_np is not None and emb_graph_id_to_idx_dict is not None:
        gemb_flag = True
    else:
        gemb_flag = False

    stoi, itos = meta['stoi'], meta['itos']
    _numeric_keys = sorted([k for k in stoi if isinstance(k, (int, float))])
    _numeric_arr  = np.array(_numeric_keys) if _numeric_keys else None
    def _safe_lookup(c):
        if c in stoi:
            return stoi[c]
        if _numeric_arr is not None and isinstance(c, (int, float)):
            nearest = _numeric_keys[int(np.argmin(np.abs(_numeric_arr - c)))]
            return stoi[nearest]
        raise KeyError(c)
    encode = lambda s: [_safe_lookup(c) for c in s]
    decode = lambda l: [itos[i] for i in l]

    n_edges_to_count_dict = test_run_df['edgelist_list_len'].value_counts().to_dict()

    adapt_gpt_out_list_dict = defaultdict(list)
    x_list_dict = defaultdict(list)
    graph_emb_dict = defaultdict(list)
    y_dict = dict()

    for n_edges, n_graphs in n_edges_to_count_dict.items():
        cur_test_run_df = test_run_df[test_run_df['edgelist_list_len'] == n_edges]

        for row_idx, graph_df_row in cur_test_run_df.iterrows():
            start, adapt_seq = extract_graph(graph_df_row[token_seq_col])
            start_ids = encode(start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            x_list_dict[n_edges].append(x)

            if gemb_flag:
                cur_graph_idx = emb_graph_id_to_idx_dict[graph_df_row['graph_id']]
                graph_emb_dict[n_edges].append(
                    torch.tensor(graph_emb_np[cur_graph_idx],
                                 dtype=emb_dtype,
                                 device=device))

            adapt_gpt_out_dict = dict()
            adapt_gpt_out_dict['graph'] = start[1:-1]
            adapt_gpt_out_dict['n_edges'] = graph_df_row['edgelist_list_len']
            adapt_gpt_out_dict['q_circuits'] = []
            adapt_gpt_out_dict['adapt_circuit'] = adapt_seq
            adapt_gpt_out_dict['adapt_full_ar'] = graph_df_row['approx_ratio']
            adapt_gpt_out_dict['graph_prefix'] = graph_df_row['graph_id']
            if 'true_energy' in graph_df_row:
                adapt_gpt_out_dict['true_energy'] = graph_df_row['true_energy']
            if 'energy_gurobi' in graph_df_row:
                adapt_gpt_out_dict['energy_gurobi'] = graph_df_row['energy_gurobi']
            adapt_gpt_out_dict['label'] = graph_df_row['label']
            adapt_gpt_out_list_dict[n_edges].append(adapt_gpt_out_dict)

        cur_batch_torch = torch.vstack(x_list_dict[n_edges])

        if gemb_flag:
            cur_emb_batch_torch = torch.vstack(graph_emb_dict[n_edges])

        total_samples = cur_batch_torch.size(0)
        n_batches = (total_samples + n_samples_per_batch - 1) // n_samples_per_batch

        y_list = []

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * n_samples_per_batch
                end_idx = min((i + 1) * n_samples_per_batch, total_samples)

                mini_batch = cur_batch_torch[start_idx:end_idx]
                mini_batch_repeated = mini_batch.repeat(num_samples, 1)

                if gemb_flag:
                    mini_emb_batch = cur_emb_batch_torch[start_idx:end_idx]
                    mini_emb_batch_repeated = mini_emb_batch.repeat(num_samples, 1)

                with ctx:
                    if gemb_flag:
                        y = model.generate(mini_batch_repeated,
                                           mini_emb_batch_repeated,
                                           max_new_tokens,
                                           temperature=temperature,
                                           top_k=top_k)
                    else:
                        y = model.generate(mini_batch_repeated,
                                           max_new_tokens,
                                           temperature=temperature,
                                           top_k=top_k)

                y_list.append(y.detach().cpu())

        y_dict[n_edges] = torch.cat(y_list, dim=0)

    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        cur_full_y_tensor = y_dict[n_edges]

        for graph_idx in range(len(cur_adapt_gpt_out_list)):
            cur_y_tensor = cur_full_y_tensor[graph_idx::len(cur_adapt_gpt_out_list)]

            for k in range(num_samples):
                cur_gen_result = decode(cur_y_tensor[k].tolist())
                cur_circ = []
                circ_flag = 0
                for idx, tok in enumerate(cur_gen_result):
                    if tok == 'end_of_graph':
                        circ_flag = 1
                    if circ_flag:
                        cur_circ.append(tok)
                    if tok == 'eos':
                        break
                cur_adapt_gpt_out_list[graph_idx]['q_circuits'].append(cur_circ[1:-1])

    adapt_gpt_test_samples_list = []
    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        adapt_gpt_test_samples_list += cur_adapt_gpt_out_list

    for idx in range(len(adapt_gpt_test_samples_list)):
        q_circ_filt_list = []
        for circ in adapt_gpt_test_samples_list[idx]['q_circuits']:
            filt_flag = circ_sanity_check(circ)
            q_circ_filt_list.append(circ)
        adapt_gpt_test_samples_list[idx]['q_circuits'] = q_circ_filt_list

    for gr_dict in adapt_gpt_test_samples_list:
        graph_py_list = []
        graph_edges_list = gr_dict['graph'][::2]
        graph_weights_list = gr_dict['graph'][1::2]

        if normalize_weights_flag:
            graph_w_norm = sum(graph_weights_list)
        else:
            graph_w_norm = 1.0

        for edge_idx, edge in enumerate(graph_edges_list):
            cur_edge = list(edge)
            cur_edge += [graph_weights_list[edge_idx] / graph_w_norm]
            graph_py_list.append(cur_edge)

        gr_dict['graph_w_py'] = graph_py_list
        gr_dict['graph_weight_norm'] = graph_w_norm

    adapt_gpt_test_samples_filt_list = []
    for rec in adapt_gpt_test_samples_list:
        pos_flag = 1
        if pos_flag:
            adapt_gpt_test_samples_filt_list.append(rec)

    adapt_gpt_test_samples_df = pd.DataFrame(adapt_gpt_test_samples_filt_list)

    return adapt_gpt_test_samples_df

def get_feather_emb(graphs_nx_df, n_workers, n_nodes, rounding_digits=2):
    combined_unique_graphs_df = (graphs_nx_df[['graph_id', 'edgelist_json']].drop_duplicates())

    def create_weighted_graph_nx(w_elist):
        G = nx.Graph()
        G.add_weighted_edges_from(w_elist)
        return G

    combined_unique_graphs_df['edgelist_py_list'] = combined_unique_graphs_df[
        'edgelist_json'].apply(lambda x: [(e[0] - 1, e[1] - 1, e[2]) for e in json.loads(x)])

    combined_unique_graphs_df['graph_nx'] = (
        combined_unique_graphs_df['edgelist_py_list'].apply(
            lambda x: create_weighted_graph_nx(x)))

    combined_unique_graphs_w_idx_df = combined_unique_graphs_df.set_index('graph_id')
    graphs_nx_dict = combined_unique_graphs_w_idx_df['graph_nx'].to_dict()
    graphs_nx_filt_dict = dict([(name, g) for name, g in graphs_nx_dict.items()
                                if g.number_of_nodes() == n_nodes])
    graphs_nx_filt_names = list(graphs_nx_filt_dict.keys())
    graphs_nx_filt_list = list(graphs_nx_filt_dict.values())

    emb_graph_idx_to_id_dict = {k: v for k, v in enumerate(graphs_nx_filt_names)}
    emb_graph_id_to_idx_dict = {v: k for k, v in enumerate(graphs_nx_filt_names)}

    def get_single_thread_feather_emb(g_list):
        feather_model = FeatherGraph(order=5, eval_points=25, theta_max=2.5,
                                     seed=42, pooling="mean")
        feather_model.fit(graphs=g_list)
        return feather_model.get_embedding()

    def split_list(lst, n):
        it = iter(lst)
        return [list(islice(it, i)) for i in
                [len(lst) // n + (1 if x < len(lst) % n else 0) for x in range(n)]]

    def embed_nx_w_feather_parallel(graphs_list, n_workers=n_workers):
        graphs_chunked_list = split_list(graphs_list, n_workers)
        emb_np_list = Parallel(n_jobs=n_workers)(
            delayed(get_single_thread_feather_emb)(g_chunk)
            for g_chunk in graphs_chunked_list)
        return np.vstack(emb_np_list)

    feather_par_emb = embed_nx_w_feather_parallel(graphs_nx_filt_list[:], n_workers=n_workers)
    feather_par_emb = feather_par_emb.round(rounding_digits)

    return feather_par_emb, emb_graph_idx_to_id_dict

def seq_tokenize_graph(elist):
    tok_list = ['bos']
    for n1, n2, w in elist:
        tok_list += [tuple(sorted([n1, n2])), w]
    tok_list.append('end_of_graph')
    return tok_list

def gurobi_max_cut_val_from_nx(graph_nx):
    model = Model("Max-Cut")
    model.setParam('OutputFlag', False)
    model.setParam(GRB.Param.TimeLimit, 10)
    variables = {}
    for node in graph_nx.nodes:
        variables[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

    objective = 0
    for u, v, w in graph_nx.edges(data="weight"):
        objective -= w * ((2 * variables[v] * variables[u]) -
                          (variables[v] + variables[u]))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    solution = [variables[node].x for node in graph_nx.nodes]

    return -model.ObjVal

def check_if_nx_graph_is_weighted(graph_nx):
    return all('weight' in graph_nx[u][v] for u, v in graph_nx.edges)

def nx_to_elist(nx_graph, idx_shift=True):
    if not isinstance(nx_graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph object.")

    weighted = check_if_nx_graph_is_weighted(nx_graph)
    if not weighted:
        raise ValueError(
            "Current version of QAOA-GPT does not support unweighted graphs. "
            "Weights w are expected to be sampled from U(0,1).")
    shifted_elist = []
    for edge_idx, (n1, n2) in enumerate(nx_graph.edges):
        cur_e_weight = nx_graph[n1][n2]['weight']
        if idx_shift:
            n1 += 1
            n2 += 1
        shifted_elist.append((n1, n2, cur_e_weight))

    graph_nx_from_edges = nx.from_edgelist(nx_graph.edges)
    n_nodes = graph_nx_from_edges.number_of_nodes()

    return {"elist": shifted_elist, "n_nodes": n_nodes}

def prepare_model_input(graphs_container, n_nodes, calculate_classic_maxcut=True,
                       n_workers_feather=1):
    if type(graphs_container) == list:
        graphs_edgelist_list_dict = {f'graph_{i}': g for i, g in enumerate(graphs_container)}
    elif type(graphs_container) == dict:
        graphs_edgelist_list_dict = graphs_container
    else:
        raise ValueError("Only list or dict containers are supported for input graphs!")

    graphs_nx_dict = defaultdict(dict)

    for name, nx_graph in graphs_edgelist_list_dict.items():
        nx_elist_dict = nx_to_elist(nx_graph)
        graphs_nx_dict[name]['elist'] = nx_elist_dict['elist']
        graphs_nx_dict[name]['n_nodes'] = nx_elist_dict['n_nodes']
        if calculate_classic_maxcut:
            graphs_nx_dict[name]['energy_gurobi'] = gurobi_max_cut_val_from_nx(nx_graph)

    graphs_nx_df = pd.DataFrame(graphs_nx_dict).T.reset_index(names='graph_id')
    graphs_nx_df['token_seq_round_d2'] = graphs_nx_df['elist'].apply(seq_tokenize_graph)
    graphs_nx_df['edgelist_list_len'] = graphs_nx_df['elist'].apply(len)
    graphs_nx_df['approx_ratio'] = None
    graphs_nx_df['label'] = 'test_interactive'
    graphs_nx_df['edgelist_json'] = graphs_nx_df['elist'].apply(lambda x: json.dumps(x))

    feather_par_emb, emb_graph_idx_to_id_dict = get_feather_emb(
        graphs_nx_df, n_workers=n_workers_feather, n_nodes=n_nodes)

    emb_graph_id_to_idx_dict = {v: k for k, v in emb_graph_idx_to_id_dict.items()}

    graphs_nx_df['has_emb'] = graphs_nx_df['graph_id'].apply(
        lambda x: True if x in emb_graph_id_to_idx_dict else False)

    graphs_nx_df = graphs_nx_df[graphs_nx_df['has_emb']]

    return graphs_nx_df, feather_par_emb, emb_graph_id_to_idx_dict

def elist_to_nx(input_elist, idx_shift=True):
    elist = []
    if idx_shift:
        for u, v, w in input_elist:
            elist.append((u - 1, v - 1, w))
    else:
        elist = input_elist

    G = nx.Graph()
    G.add_weighted_edges_from(elist)
    return G

def graph_to_edgelist(g):
    return [(u, v, d['weight']) for u, v, d in g.edges(data=True)]

def sample_ansatz_bitstring_only(edgelist, q_circuit, n_nodes, pool_type, verbose=False):

    g = elist_to_nx(edgelist, idx_shift=True)

    e_list = graph_to_edgelist(g)

    spin_ham = max_cut_ham(e_list)
    qubits_num = n_nodes

    ham_coeffs = term_coefficients(spin_ham)
    ham_words = term_words(spin_ham, qubits_num)

    if pool_type == 'all_pool':
        pools = all_pool(qubits_num)
    elif pool_type == 'qaoa_mixer':
        pools = qaoa_mixer(qubits_num)
    elif pool_type == 'qaoa_single_x':
        pools = qaoa_single_x(qubits_num)
    elif pool_type == 'qaoa_double_ops':
        pools = qaoa_double(qubits_num)
    else:
        raise ValueError(
            "Invalid pool name. Choose from 'all_pool', 'qaoa_mixer', 'qaoa_single_x', or 'qaoa_double'."
        )

    op_indeces = []
    beta_coef = []
    gamma_coef = []

    for j in range(0, len(q_circuit), 4):
        op_indeces.append(q_circuit[j + 1] - 1)
        beta_coef.append(q_circuit[j + 2])
        gamma_coef.append(q_circuit[j + 3])

    temp = []
    for idx in op_indeces:
        if idx < len(pools):
            temp.append(pools[idx])
        else:
            raise ValueError(
                f"Operator index {idx} out of range for the pool of size {len(pools)}."
            )

    mixer_pool = []
    mixer_pool_str = []

    for op in temp:
        temp_op = []
        temp_op_str = []
        for term in op:
            temp_op.append(cudaq.pauli_word(term.get_pauli_word(qubits_num)))
            temp_op_str.append(term.get_pauli_word(qubits_num))
        mixer_pool.append(temp_op)
        mixer_pool_str.append(temp_op_str)
    if verbose:
        print(
            f"Using {len(mixer_pool)} operators from the pool: {mixer_pool_str}"
        )

    num_layer = len(mixer_pool)

    @cudaq.kernel
    def kernel_qaoa(qubits_num: int, ham_words: list[cudaq.pauli_word],
                    ham_coeffs: list[complex],
                    mixer_pool: list[list[cudaq.pauli_word]],
                    gamma: list[float], beta: list[float], num_layer: int):

        qubits = cudaq.qvector(qubits_num)

        h(qubits)

        idx = 0
        for p in range(num_layer):

            for i in range(len(ham_coeffs)):
                exp_pauli(gamma[p] * ham_coeffs[i].real, qubits, ham_words[i])

            for word in mixer_pool[p]:
                exp_pauli(beta[p], qubits, word)

    result = cudaq.sample(kernel_qaoa, qubits_num, ham_words, ham_coeffs,
                          mixer_pool, gamma_coef, beta_coef, num_layer, shots_count=1000)
    return result.most_probable()

def process_graph_bitstring_only(graph_idx, adapt_gpt_out_list, n_nodes, pool_type,
                                 num_shots=1024, verbose=False):
    adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]
    edgelist = adapt_gpt_out_dict["graph_w_py"]
    adapt_gpt_bitstrings_list = []

    q_circuits = adapt_gpt_out_dict.get("q_circuits", [])

    for i in range(len(q_circuits) + 1):
        if i < len(q_circuits):
            generated_list = q_circuits[i]
        else:
            if "adapt_circuit" not in adapt_gpt_out_dict:
                raise ValueError("No 'adapt_circuit' found in the output dictionary.")
            else:
                generated_list = adapt_gpt_out_dict["adapt_circuit"]

        bitstring = '0' * n_nodes

        try:
            bitstring = sample_ansatz_bitstring_only(edgelist, generated_list, n_nodes,
                                                    pool_type, verbose=verbose)
        except Exception as e:
            pass

        if i < len(q_circuits):
            adapt_gpt_bitstrings_list.append(bitstring)
        else:
            adapt_gpt_out_dict["ADAPT_bitstring"] = bitstring

    adapt_gpt_out_dict["adapt_gpt_bitstrings"] = adapt_gpt_bitstrings_list

def process_dataframe_bitstring_only(adapt_gpt_res_df, n_nodes, pool_type,
                                     num_shots=1024, verbose=False):
    adapt_gpt_out_list = adapt_gpt_res_df.to_dict('records')

    for graph_idx in range(len(adapt_gpt_out_list)):
        process_graph_bitstring_only(graph_idx, adapt_gpt_out_list, n_nodes,
                                     pool_type, num_shots, verbose)

    result_df = pd.DataFrame(adapt_gpt_out_list)
    return result_df

def solve_qaoa_circuits_with_bitstrings(qaoa_gpt_circ_df, n_nodes,
                                       pool_type="all_pool", num_shots=1024, verbose=False):
    return process_dataframe_bitstring_only(
        qaoa_gpt_circ_df,
        n_nodes,
        pool_type=pool_type,
        num_shots=num_shots,
        verbose=verbose
    )

import os
import sys
import json
import torch
import cudaq
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
from itertools import islice
from joblib import Parallel, delayed

from qaoa_gpt_src.custom_feather import CustomFeatherGraph as FeatherGraph
from qaoa_gpt_src.hubo_hamiltonian import (
    hubo_to_spin_ham,
    hubo_to_projected_graph,
    hubo_to_interaction_lists,
)
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double

os.environ["CUDA_QUANTUM_OPTIMIZATION_LEVEL"] = "0"
os.environ["CUDAQ_DISABLE_LOOP_UNROLL"] = "1"

def _set_cudaq_target():
    _requested = os.environ.get("CUDAQ_TARGET", "auto")
    if _requested not in ("auto", "nvidia"):
        cudaq.set_target(_requested)
        return
    cudaq.set_target("nvidia", option="fp64")
    try:
        @cudaq.kernel
        def _probe():
            q = cudaq.qvector(1)
            h(q[0])
        cudaq.sample(_probe, shots_count=1)
    except Exception:
        cudaq.set_target("qpp-cpu")
_set_cudaq_target()

def seq_tokenize_hubo(linear_list, quad_list, cubic_list):
    tok_list = ['bos']
    for (i, w) in linear_list:
        tok_list += [(i,), w]
    for (i, j, w) in quad_list:
        tok_list += [(i, j), w]
    for (i, j, k, w) in cubic_list:
        tok_list += [(i, j, k), w]
    tok_list.append('end_of_graph')
    return tok_list

def get_feather_emb_hubo(hubo_df, n_nodes, n_workers=1, rounding_digits=2):
    unique_df = hubo_df[['graph_id', 'projected_edgelist_json']].drop_duplicates()

    def _make_nx(elist_json):
        elist_0 = [(e[0] - 1, e[1] - 1, e[2]) for e in json.loads(elist_json)]
        G = nx.Graph()
        G.add_weighted_edges_from(elist_0)
        return G

    unique_df = unique_df.copy()
    unique_df['graph_nx'] = unique_df['projected_edgelist_json'].apply(_make_nx)

    graphs_dict = unique_df.set_index('graph_id')['graph_nx'].to_dict()
    filt_dict = {name: g for name, g in graphs_dict.items()
                 if g.number_of_nodes() == n_nodes}
    filt_names = list(filt_dict.keys())
    filt_list = list(filt_dict.values())

    emb_graph_idx_to_id = {k: v for k, v in enumerate(filt_names)}

    def _feather(g_list):
        model = FeatherGraph(order=5, eval_points=25, theta_max=2.5, seed=42, pooling="mean")
        model.fit(graphs=g_list)
        return model.get_embedding()

    def _split(lst, n):
        it = iter(lst)
        return [list(islice(it, i))
                for i in [len(lst) // n + (1 if x < len(lst) % n else 0)
                          for x in range(n)]]

    if n_workers > 1:
        chunks = _split(filt_list, n_workers)
        embs = Parallel(n_jobs=n_workers)(delayed(_feather)(c) for c in chunks)
        feather_emb = np.vstack(embs)
    else:
        feather_emb = _feather(filt_list)

    return feather_emb.round(rounding_digits), emb_graph_idx_to_id

def prepare_model_input_hubo(sub_hubos_dict, n_nodes,
                               n_workers_feather=1, rounding_digits=1):
    rows = []
    for name, T in sub_hubos_dict.items():
        linear_list, quad_list, cubic_list = hubo_to_interaction_lists(
            T, rounding_digits=rounding_digits)

        proj_G = hubo_to_projected_graph(T)
        proj_elist = [
            (u + 1, v + 1, round(data['weight'], rounding_digits))
            for u, v, data in proj_G.edges(data=True)
        ]
        proj_json = json.dumps(proj_elist)

        tok_seq = seq_tokenize_hubo(linear_list, quad_list, cubic_list)
        n_interactions = len(linear_list) + len(quad_list) + len(cubic_list)

        rows.append({
            'graph_id': name,
            'linear_list': linear_list,
            'quad_list': quad_list,
            'cubic_list': cubic_list,
            'projected_edgelist_json': proj_json,
            'token_seq_round_d2': tok_seq,
            'interaction_count': n_interactions,
            'edgelist_list_len': n_interactions,
            'approx_ratio': None,
            'label': 'test_interactive',
        })

    hubo_nx_df = pd.DataFrame(rows)

    feather_par_emb, emb_graph_idx_to_id = get_feather_emb_hubo(
        hubo_nx_df, n_nodes=n_nodes, n_workers=n_workers_feather,
        rounding_digits=rounding_digits)

    emb_graph_id_to_idx = {v: k for k, v in emb_graph_idx_to_id.items()}

    hubo_nx_df['has_emb'] = hubo_nx_df['graph_id'].apply(
        lambda x: x in emb_graph_id_to_idx)
    hubo_nx_df = hubo_nx_df[hubo_nx_df['has_emb']]

    return hubo_nx_df, feather_par_emb, emb_graph_id_to_idx

def sample_ansatz_bitstring_only_hubo(T, q_circuit, n_nodes, pool_type,
                                       num_shots=1000, verbose=False):
    spin_ham = hubo_to_spin_ham(T)
    ham_coeffs = term_coefficients(spin_ham)
    ham_words = term_words(spin_ham, n_nodes)

    if pool_type == 'all_pool':
        pools = all_pool(n_nodes)
    elif pool_type == 'qaoa_mixer':
        pools = qaoa_mixer(n_nodes)
    elif pool_type == 'qaoa_single_x':
        pools = qaoa_single_x(n_nodes)
    elif pool_type == 'qaoa_double_ops':
        pools = qaoa_double(n_nodes)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

    op_indices = []
    beta_coef = []
    gamma_coef = []
    for j in range(0, len(q_circuit), 4):
        op_indices.append(q_circuit[j + 1] - 1)
        beta_coef.append(q_circuit[j + 2])
        gamma_coef.append(q_circuit[j + 3])

    mixer_pool = []
    for idx in op_indices:
        if idx >= len(pools):
            raise ValueError(
                f"Operator index {idx} out of range for pool size {len(pools)}")
        op = pools[idx]
        layer_ops = [cudaq.pauli_word(term.get_pauli_word(n_nodes)) for term in op]
        mixer_pool.append(layer_ops)

    num_layer = len(mixer_pool)

    if verbose:
        print(f"  num_layer={num_layer}, beta={beta_coef}, gamma={gamma_coef}")

    @cudaq.kernel
    def kernel_qaoa_hubo(qubits_num: int,
                         ham_words: list[cudaq.pauli_word],
                         ham_coeffs: list[complex],
                         mixer_pool: list[list[cudaq.pauli_word]],
                         gamma: list[float],
                         beta: list[float],
                         num_layer: int):
        qubits = cudaq.qvector(qubits_num)
        h(qubits)
        for p in range(num_layer):
            for i in range(len(ham_coeffs)):
                exp_pauli(gamma[p] * ham_coeffs[i].real, qubits, ham_words[i])
            for word in mixer_pool[p]:
                exp_pauli(beta[p], qubits, word)

    result = cudaq.sample(
        kernel_qaoa_hubo,
        n_nodes, ham_words, ham_coeffs, mixer_pool,
        gamma_coef, beta_coef, num_layer,
        shots_count=num_shots,
    )
    return result.most_probable()

def _process_graph_bitstring_hubo(graph_idx, adapt_gpt_out_list, T_dict,
                                    n_nodes, pool_type, num_shots=1024,
                                    verbose=False):
    rec = adapt_gpt_out_list[graph_idx]
    graph_prefix = rec['graph_prefix']

    T = T_dict.get(graph_prefix)
    if T is None:
        rec['ADAPT_bitstring'] = '0' * n_nodes
        rec['adapt_gpt_bitstrings'] = []
        return

    bitstrings = []
    q_circuits = rec.get('q_circuits', [])

    for i in range(len(q_circuits) + 1):
        circ = q_circuits[i] if i < len(q_circuits) else rec.get('adapt_circuit', [])
        bitstring = '0' * n_nodes
        try:
            bitstring = sample_ansatz_bitstring_only_hubo(
                T, circ, n_nodes, pool_type,
                num_shots=num_shots, verbose=verbose)
        except Exception:
            pass

        if i < len(q_circuits):
            bitstrings.append(bitstring)
        else:
            rec['ADAPT_bitstring'] = bitstring

    rec['adapt_gpt_bitstrings'] = bitstrings

def solve_hubo_circuits_with_bitstrings(qaoa_gpt_circ_df, T_dict,
                                         n_nodes, pool_type='all_pool',
                                         num_shots=1024, verbose=False):
    records = qaoa_gpt_circ_df.to_dict('records')
    for idx in range(len(records)):
        _process_graph_bitstring_hubo(
            idx, records, T_dict, n_nodes, pool_type, num_shots, verbose)
    return pd.DataFrame(records)

def extract_graph_hubo(token_seq):
    graph_seq = []
    for idx, tok in enumerate(token_seq):
        graph_seq.append(tok)
        if tok == 'end_of_graph':
            break
    adapt_seq = token_seq[idx + 1:-1]
    return graph_seq, adapt_seq

def circ_sanity_check(cur_q_circ):
    if len(cur_q_circ) % 4:
        return False
    op_idx_list = cur_q_circ[1::4]
    lr_sep_list = cur_q_circ[0::4]
    if any(type(el) != int for el in op_idx_list):
        return False
    if any(type(el) != str for el in lr_sep_list):
        return False
    return True

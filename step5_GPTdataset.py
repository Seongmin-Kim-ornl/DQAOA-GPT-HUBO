"""
STEP 5 (v2) — Active-learning data generation via GPT inference (multigamma).

Multigamma setup: one gamma per QAOA layer (no per-block grouping).
Generates CSV data compatible with step2_tokenize.py (v2).

GPT-predicted thetas and gammas are used directly (no re-optimization).
AR is the Ising-space expectation value vs brute-force minimum.
Only circuits whose AR >= ar_threshold are saved.

Usage
-----
    python step5_GPTdataset.py \\
        --ckpt /path/to/ckpt.pt --data_dir /path/to/data_dir \\
        --output_dir adapt_results_hubo_10nodes \\
        --n_nodes 10 --hubo_number 500 \\
        --ar_threshold 0.80
"""

import os
import sys
import json
import time
import argparse
import pickle
import re
import contextlib
import io
import random

import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cudaq

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    brute_force_hubo,
    simulated_annealing_hubo,
    hubo_to_spin_ham,
    hubo_to_projected_graph,
    hubo_to_interaction_lists,
    evaluate_hubo_energy,
)
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double
from qaoa_gpt_src.util_hubo import circ_sanity_check
from utils.solve_HUBO_GPT import _get_hubo_gpt_model

_POOL_MAP = {
    'all_pool':        all_pool,
    'qaoa_mixer':      qaoa_mixer,
    'qaoa_single_x':   qaoa_single_x,
    'qaoa_double_ops': qaoa_double,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(output_dir):
    for sub in ['hubo', 'res']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


def _projected_elist_json(T, rounding_digits):
    G = hubo_to_projected_graph(T)
    elist = [
        (u + 1, v + 1, round(data['weight'], rounding_digits))
        for u, v, data in G.edges(data=True)
    ]
    return json.dumps(elist)


def _get_pool(pool_type, n_nodes):
    return _POOL_MAP.get(pool_type, all_pool)(n_nodes)


def _get_ham_terms(T, n_nodes):
    """Return (pauli_words, coeffs, spin_ham) for cudaq kernels."""
    spin_ham   = hubo_to_spin_ham(T)
    ham_coeffs = term_coefficients(spin_ham)
    ham_words  = term_words(spin_ham, n_nodes)
    cw, cc = [], []
    for c, w in zip(ham_coeffs, ham_words):
        if set(w) == {'I'}:
            continue
        cw.append(cudaq.pauli_word(w))
        cc.append(float(c.real))
    return cw, cc, spin_ham


def _build_mixer_pool_str(op_indices_0indexed, n_nodes, pool_type='all_pool'):
    """Return list-of-lists of Pauli word strings (for CSV mixer_pool_pauli_word)."""
    pools = _get_pool(pool_type, n_nodes)
    return [
        [term.get_pauli_word(n_nodes) for term in pools[idx]]
        for idx in op_indices_0indexed
    ]


def _parse_inline_circuit(q_circuit):
    """Parse v2 inline format: [new_layer_p, op_idx_1indexed, gamma, theta] * L."""
    L            = len(q_circuit) // 4
    op_indices_0 = [q_circuit[j * 4 + 1] - 1 for j in range(L)]
    gammas       = [q_circuit[j * 4 + 2]     for j in range(L)]
    thetas       = [q_circuit[j * 4 + 3]     for j in range(L)]
    return op_indices_0, gammas, thetas


def _make_mixer_pool_cudaq(op_indices_0, pool_type, n_nodes):
    pools = _get_pool(pool_type, n_nodes)
    return [
        [cudaq.pauli_word(term.get_pauli_word(n_nodes)) for term in pools[idx]]
        for idx in op_indices_0
    ]


def _observe_energy(T, op_indices_0, thetas, gammas, n_nodes, pool_type='all_pool'):
    """Compute Ising-space expectation value for circuit screening."""
    ham_words_cq, ham_coeffs_cq, spin_ham = _get_ham_terms(T, n_nodes)
    mixer_pool = _make_mixer_pool_cudaq(op_indices_0, pool_type, n_nodes)
    num_layer  = len(op_indices_0)

    @cudaq.kernel
    def _kernel_obs(qubits_num: int,
                    ham_words: list[cudaq.pauli_word],
                    ham_coeffs: list[float],
                    mp: list[list[cudaq.pauli_word]],
                    gamma: list[float],
                    beta: list[float],
                    nl: int):
        qubits = cudaq.qvector(qubits_num)
        h(qubits)
        for p in range(nl):
            for i in range(len(ham_coeffs)):
                exp_pauli(gamma[p] * ham_coeffs[i], qubits, ham_words[i])
            for word in mp[p]:
                exp_pauli(beta[p], qubits, word)

    return cudaq.observe(
        _kernel_obs, spin_ham,
        n_nodes, ham_words_cq, ham_coeffs_cq,
        mixer_pool, list(gammas), list(thetas), num_layer,
    ).expectation()


def _sample_bitstring(T, op_indices_0, thetas, gammas, n_nodes,
                      num_shots=5000, pool_type='all_pool'):
    """Sample most-probable bitstring from the per-layer-gamma ansatz."""
    ham_words_cq, ham_coeffs_cq, _ = _get_ham_terms(T, n_nodes)
    mixer_pool = _make_mixer_pool_cudaq(op_indices_0, pool_type, n_nodes)
    num_layer  = len(op_indices_0)

    @cudaq.kernel
    def _kernel_sample(qubits_num: int,
                       ham_words: list[cudaq.pauli_word],
                       ham_coeffs: list[float],
                       mp: list[list[cudaq.pauli_word]],
                       gamma: list[float],
                       beta: list[float],
                       nl: int):
        qubits = cudaq.qvector(qubits_num)
        h(qubits)
        for p in range(nl):
            for i in range(len(ham_coeffs)):
                exp_pauli(gamma[p] * ham_coeffs[i], qubits, ham_words[i])
            for word in mp[p]:
                exp_pauli(beta[p], qubits, word)

    count = cudaq.sample(
        _kernel_sample,
        n_nodes, ham_words_cq, ham_coeffs_cq,
        mixer_pool, list(gammas), list(thetas), num_layer,
        shots_count=num_shots,
    )
    return str(count.most_probable())


@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# GPT inference
# ---------------------------------------------------------------------------

def _run_gpt_inference(gpt_model, hubo_name, T_rounded, rounding_digits,
                       num_gpt_samples, temperature):
    from tqdm import tqdm
    tqdm.disable = True
    with _silence():
        circ_df = gpt_model.generate_circ_from_sub_hubos(
            {hubo_name: T_rounded},
            n_samples_per_batch=50,
            num_samples=num_gpt_samples,
            max_new_tokens=150,
            temperature=temperature,
            top_k=200,
            rounding_digits=rounding_digits,
        )

    valid_circs = []
    for _, row in circ_df.iterrows():
        for c in row.get('q_circuits', []):
            if circ_sanity_check(c):
                valid_circs.append(c)
        ac = row.get('adapt_circuit', [])
        if ac and circ_sanity_check(ac):
            valid_circs.append(ac)
    return valid_circs


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run_active_learning(args, gpt_model, rounding_digits, tensor_list, snap_fn,
                        pool_type='all_pool'):
    _ensure_dirs(args.output_dir)

    pid = os.getpid()
    ts  = datetime.now().strftime("%y-%m-%d__%H_%M")

    hubo_csv_path    = os.path.join(args.output_dir, 'hubo',
                                    f'step5_pid{pid}_{ts}_hubo.csv')
    results_csv_path = os.path.join(args.output_dir, 'res',
                                    f'step5_pid{pid}_{ts}_results.csv')

    hubo_rows   = []
    result_rows = []
    n_added     = 0
    n_total     = len(tensor_list)

    for hubo_num, T in enumerate(tensor_list):
        cur_hubo_name = f"HUBO_{hubo_num + 1}"
        t_start = time.time()
        print(f"\n[{hubo_num + 1}/{n_total}] {cur_hubo_name}", flush=True)

        # ---- Reference energy: Ising space, consistent with v2 training data ----
        true_energy         = -1e-8
        classical_bitstring = '0' * args.n_nodes
        e_bf                = -1e-8

        if not args.no_brute_force and args.n_nodes <= 20:
            best_ising_energy, best_hubo_energy, classical_bitstring = brute_force_hubo(T)
            true_energy = best_ising_energy if abs(best_ising_energy) >= 1e-8 else -1e-8
            e_bf = best_hubo_energy
            print(f"  BF min (HUBO): {best_hubo_energy:.4f}")
        elif not args.no_sa:
            best_ising_energy, best_sa_energy, classical_bitstring = simulated_annealing_hubo(T)
            true_energy = best_ising_energy if abs(best_ising_energy) >= 1e-8 else -1e-8
            e_bf = best_sa_energy
            print(f"  SA min (HUBO): {best_sa_energy:.4f}")

        # ---- GPT inference ----
        T_rounded   = np.vectorize(snap_fn)(np.round(T.copy(), rounding_digits))
        valid_circs = _run_gpt_inference(
            gpt_model, cur_hubo_name, T_rounded, rounding_digits,
            args.num_gpt_samples, args.temperature)

        if not valid_circs:
            print(f"  No valid GPT circuits — skipping.")
            continue

        print(f"  Valid GPT circuits: {len(valid_circs)}")

        # ---- Screen circuits by Ising expectation value ----
        if len(valid_circs) == 1:
            best_circ = valid_circs[0]
        else:
            screened = []
            for circ in valid_circs:
                try:
                    op_idx0, gm, th = _parse_inline_circuit(circ)
                    e_obs = _observe_energy(T, op_idx0, th, gm, args.n_nodes, pool_type)
                except Exception:
                    e_obs = float('inf')
                screened.append((e_obs, circ))
            screened.sort(key=lambda x: x[0])
            best_circ = screened[0][1]

        op_indices_0, gammas_gpt, thetas_gpt = _parse_inline_circuit(best_circ)
        num_layers = len(op_indices_0)

        opt_thetas = thetas_gpt
        opt_gammas = gammas_gpt

        # ---- Sample bitstring ----
        try:
            bitstring = _sample_bitstring(
                T, op_indices_0, opt_thetas, opt_gammas,
                args.n_nodes, num_shots=args.num_shots, pool_type=pool_type)
        except Exception:
            bitstring = '0' * args.n_nodes

        # ---- HUBO energy of sampled bitstring ----
        x_bits = np.array([int(b) for b in bitstring], dtype=float)
        e_gpt  = evaluate_hubo_energy(T, x_bits)
        ar     = e_gpt / e_bf if abs(e_bf) > 1e-8 else float('nan')

        elapsed_time = time.time() - t_start
        print(f"  e_gpt={e_gpt:.4f}  e_bf={e_bf:.4f}"
              f"  ar={ar:.4f}  threshold={args.ar_threshold}"
              f"  layers={num_layers}  gammas={[round(g, 3) for g in opt_gammas]}"
              f"  ({elapsed_time:.1f}s)")

        if ar < args.ar_threshold:
            print(f"  AR below threshold — skipping.")
            continue

        energy_list              = [e_gpt] * num_layers
        pool_list_index_1indexed = [i + 1 for i in op_indices_0]
        mixer_pool_pauli_word    = _build_mixer_pool_str(op_indices_0, args.n_nodes, pool_type)

        linear_list, quad_list, cubic_list = hubo_to_interaction_lists(
            T, rounding_digits=args.rounding_digits)
        projected_edgelist_json = _projected_elist_json(T, args.rounding_digits)
        n_interactions = len(linear_list) + len(quad_list) + len(cubic_list)

        seq_num = n_added + 1
        hubo_rows.append({
            'graph_num':               seq_num,
            'hubo_name':               cur_hubo_name,
            'n_nodes':                 args.n_nodes,
            'linear_json':             json.dumps(linear_list),
            'quad_json':               json.dumps(quad_list),
            'cubic_json':              json.dumps(cubic_list),
            'projected_edgelist_json': projected_edgelist_json,
            'n_interactions':          n_interactions,
        })

        result_rows.append({
            'method':                   'step5-direct',
            'graph_name':               cur_hubo_name,
            'graph_num':                seq_num,
            'trial_num':                1,
            'n_nodes':                  args.n_nodes,
            'init_gamma':               opt_gammas[0],
            'optimizer':                'GPT-direct',
            'pool_type':                pool_type,
            'edge_weight_scaling_coef': 1.0,
            'edge_weight_norm_coef':    1.0,
            'energy_list':              energy_list,
            'true_energy':              true_energy,
            'mixer_pool_pauli_word':    mixer_pool_pauli_word,
            'mixer_pool_index':         pool_list_index_1indexed,
            'gamma_coef':               opt_gammas,
            'beta_coef':                opt_thetas,
            'approx_ratio':             ar,
            'cut_adapt':                bitstring.zfill(args.n_nodes),
            'cut_classical':            str(classical_bitstring).zfill(args.n_nodes),
            'num_layers':               num_layers,
            'optimizer_success_flag':   True,
            'elapsed_time':             elapsed_time,
        })

        n_added += 1
        print(f"  Added. Total added so far: {n_added}")

        if args.checkpoint_every and n_added % args.checkpoint_every == 0:
            pd.DataFrame(hubo_rows).to_csv(hubo_csv_path, index=False)
            pd.DataFrame(result_rows).to_csv(results_csv_path, index=False)
            print(f"  [checkpoint] {n_added} valid circuits saved.")

    # ---- Final save ----
    pd.DataFrame(hubo_rows).to_csv(hubo_csv_path, index=False)
    pd.DataFrame(result_rows).to_csv(results_csv_path, index=False)

    accept_rate = n_added / max(n_total, 1) * 100
    print(f"\n{'=' * 60}")
    print(f"Step 5 complete.")
    print(f"  HUBOs processed  : {n_total}")
    print(f"  Circuits added   : {n_added}  ({accept_rate:.1f}%)")
    print(f"  HUBO metadata    -> {hubo_csv_path}")
    print(f"  ADAPT results    -> {results_csv_path}")
    print(f"  Next: run step2 with --results_dir pointing to {args.output_dir}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 5 (v2): active-learning data generation via GPT inference (multigamma)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # GPT model
    p.add_argument("--ckpt",            type=str, required=True)
    p.add_argument("--data_dir",        type=str, default=None)
    p.add_argument("--config_file",     type=str, default=None)
    p.add_argument("--device",          type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--num_gpt_samples", type=int,   default=5)
    p.add_argument("--temperature",     type=float, default=0.8)
    # HUBO generation
    p.add_argument("--n_nodes",         type=int,   default=6)
    p.add_argument("--hubo_number",     type=int,   default=500)
    p.add_argument("--density",         type=float, default=0.5)
    p.add_argument("--weight_range",    type=float, default=1.0)
    p.add_argument("--rounding_digits", type=int,   default=1)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--from_formulated", action="store_true")
    p.add_argument("--hubo_size",       type=int,   default=None)
    p.add_argument("--formulated_dir",  type=str,   default=None)
    p.add_argument("--hubo_npy_dir",    type=str,   default=None)
    # Active-learning strategy
    p.add_argument("--ar_threshold",    type=float, default=0.80)
    p.add_argument("--num_shots",       type=int,   default=1024)
    # Output
    p.add_argument("--output_dir",      type=str,   default="adapt_results_hubo")
    p.add_argument("--checkpoint_every", type=int,  default=20)
    p.add_argument("--no_brute_force",  action="store_true")
    p.add_argument("--no_sa",           action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    ckpt     = args.ckpt
    data_dir = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
    config_file = args.config_file or os.path.join(data_dir,
                                                    'train_adapt_gpt_config.py')

    for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                        (config_file, "config_file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("=" * 60)
    print("STEP 5 (v2): Active-learning via GPT (multigamma, per-layer gamma)")
    print("=" * 60)
    print(f"  ckpt           : {ckpt}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  n_nodes        : {args.n_nodes}")
    print(f"  hubo_number    : {args.hubo_number}")
    print(f"  ar_threshold   : {args.ar_threshold}")
    print(f"  num_gpt_samples: {args.num_gpt_samples}")
    print("=" * 60)

    with open(config_file, 'r') as cf:
        cfg_text = cf.read()
    _rd = re.search(r"token_seq_round\s*=\s*['\"].*_d(\d+)['\"]", cfg_text)
    rounding_digits = int(_rd.group(1)) if _rd else args.rounding_digits
    _pt = re.search(r"pool_type\s*=\s*['\"](\w+)['\"]", cfg_text)
    pool_type = _pt.group(1) if _pt else 'all_pool'
    print(f"  pool_type      : {pool_type}  (from config)")
    print(f"  rounding_digits: {rounding_digits}  (from config)")

    _meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(_meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    vocab_nums = np.array(sorted(
        [k for k in stoi if isinstance(k, (int, float)) and not isinstance(k, bool)]))

    def _snap(val):
        if val == 0.0:
            return 0.0
        if len(vocab_nums) == 0:
            return val
        vf_min, vf_max = vocab_nums[0], vocab_nums[-1]
        clipped = float(np.clip(val, vf_min, vf_max))
        return float(vocab_nums[np.argmin(np.abs(vocab_nums - clipped))])

    print("Loading GPT model...")
    with _silence():
        gpt_model = _get_hubo_gpt_model(ckpt, data_dir, config_file, args.device)
    print("GPT model loaded.")

    import importlib
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Using seed: {seed}")

    if args.from_formulated:
        if args.hubo_size is None:
            print("ERROR: --hubo_size required with --from_formulated")
            sys.exit(1)
        subhubo_gen = importlib.import_module('subHUBO_gen')
        base = args.formulated_dir or os.getcwd()
        T_full = subhubo_gen.formulate_HUBO(args.hubo_size, data_dir=base)
        random.seed(seed)
        tensor_list = []
        for _ in range(args.hubo_number):
            sub_T, _ = subhubo_gen.make_sub_HUBO(T_full, args.n_nodes)
            tensor_list.append(sub_T)
        print(f"HUBO: {args.hubo_number} sub-HUBOs from formulated size={args.hubo_size}")

    elif args.hubo_npy_dir is not None:
        npy_files = sorted(f for f in os.listdir(args.hubo_npy_dir) if f.endswith('.npy'))
        tensor_list = [
            np.load(os.path.join(args.hubo_npy_dir, f))
            for f in npy_files[:args.hubo_number]
        ]
        print(f"HUBO: {len(tensor_list)} tensors from {args.hubo_npy_dir}")

    else:
        rng = np.random.default_rng(seed)
        tensor_list = [
            generate_random_hubo(
                args.n_nodes,
                density=args.density,
                weight_range=args.weight_range,
                seed=int(rng.integers(0, 2**31)),
            )
            for _ in range(args.hubo_number)
        ]
        print(f"HUBO: {args.hubo_number} random HUBOs (n={args.n_nodes})")

    run_active_learning(args, gpt_model, rounding_digits, tensor_list, _snap,
                        pool_type=pool_type)


if __name__ == "__main__":
    main()

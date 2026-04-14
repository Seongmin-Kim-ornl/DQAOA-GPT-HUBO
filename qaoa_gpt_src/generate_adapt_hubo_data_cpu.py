import os
import json
import socket
import time
import numpy as np
import pandas as pd
from datetime import datetime

import cudaq
from qaoa_gpt_src.adapt_qaoa import adapt_qaoa_run
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.hubo_hamiltonian import (
    hubo_to_spin_ham,
    hubo_to_projected_graph,
    hubo_to_interaction_lists,
    brute_force_hubo,
    simulated_annealing_hubo,
    generate_random_hubo,
)
import importlib

cudaq.set_target("qpp-cpu")

def _ensure_dirs(output_dir):
    for sub in ['hubo', 'res', 'traces']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

def _projected_elist_json(T, rounding_digits=2):
    G = hubo_to_projected_graph(T)
    elist = []
    for u, v, data in G.edges(data=True):
        w = round(data['weight'], rounding_digits)
        elist.append((u + 1, v + 1, w))
    return json.dumps(elist)

def generate_data_hubo(output_dir='adapt_results_hubo',
                        hubo_tensors=None,
                        hubo_input_npy_dir=None,
                        from_formulated=False,
                        hubo_size=None,
                        n_nodes=4,
                        hubo_number=50,
                        include_linear=True,
                        include_quadratic=True,
                        include_cubic=True,
                        density=0.5,
                        weight_range=1.0,
                        rounding_digits=2,
                        use_brute_force=True,
                        use_simulated_annealing=True,
                        op_pool='all_pool',
                        init_gamma=None,
                        trials_per_hubo=1,
                        optimizer='BFGS',
                        approx_ratio=0.97,
                        max_iter=10,
                        norm_threshold=1e-3,
                        energy_threshold=1e-9,
                        multi_gamma=False,
                        seed_hubo=None,
                        seed_adapt=None,
                        verbose=True,
                        checkpoint_every=None):
    if init_gamma is None:
        init_gamma = [0.01]

    _ensure_dirs(output_dir)

    pid = os.getpid()
    hostname = socket.gethostname()
    ts_string = datetime.now().strftime("%y-%m-%d__%H_%M")

    if hubo_tensors is not None:
        tensor_list = list(hubo_tensors)
        hubo_number = len(tensor_list)

    elif from_formulated:
        if hubo_size is None:
            raise ValueError("hubo_size must be provided when from_formulated=True")
        subhubo_gen = importlib.import_module('subHUBO_gen')
        if verbose:
            print(f"Loading formulated HUBO (size={hubo_size}) from text files...")
        T_full = subhubo_gen.formulate_HUBO(hubo_size)
        if verbose:
            print(f"Full HUBO loaded. Drawing {hubo_number} sub-HUBOs of size {n_nodes}...")
        rng = np.random.default_rng(seed_hubo)
        tensor_list = []
        for _ in range(hubo_number):
            sub_T, _ = subhubo_gen.make_sub_HUBO(T_full, n_nodes)
            tensor_list.append(sub_T)

    elif hubo_input_npy_dir is not None:
        npy_files = sorted([
            f for f in os.listdir(hubo_input_npy_dir) if f.endswith('.npy')
        ])
        tensor_list = [
            np.load(os.path.join(hubo_input_npy_dir, f)) for f in npy_files
        ]
        hubo_number = len(tensor_list)

    else:
        rng = np.random.default_rng(seed_hubo)
        tensor_list = [
            generate_random_hubo(
                n_nodes,
                include_linear=include_linear,
                include_quadratic=include_quadratic,
                include_cubic=include_cubic,
                density=density,
                weight_range=weight_range,
                seed=int(rng.integers(0, 2**31)),
            )
            for _ in range(hubo_number)
        ]

    if verbose:
        print(f"Processing {hubo_number} HUBO tensors...")

    hubo_rows = []
    result_rows = []

    hubo_csv_path = os.path.join(output_dir, 'hubo',
                                 f'pid{pid}_{ts_string}_hubo.csv')
    results_csv_path = os.path.join(output_dir, 'res',
                                    f'pid{pid}_{ts_string}_results.csv')

    for hubo_num, T in enumerate(tensor_list):

        cur_n_nodes = T.shape[0]
        cur_hubo_name = f"HUBO_{hubo_num + 1}"

        if verbose:
            print(f"\nProcessing {cur_hubo_name}  (n={cur_n_nodes})...")

        linear_list, quad_list, cubic_list = hubo_to_interaction_lists(
            T, rounding_digits=rounding_digits)

        linear_json = json.dumps(linear_list)
        quad_json = json.dumps(quad_list)
        cubic_json = json.dumps(cubic_list)
        projected_edgelist_json = _projected_elist_json(T, rounding_digits)

        n_interactions = len(linear_list) + len(quad_list) + len(cubic_list)

        hubo_rows.append({
            'graph_num': hubo_num + 1,
            'hubo_name': cur_hubo_name,
            'n_nodes': cur_n_nodes,
            'linear_json': linear_json,
            'quad_json': quad_json,
            'cubic_json': cubic_json,
            'projected_edgelist_json': projected_edgelist_json,
            'n_interactions': n_interactions,
        })

        spin_ham = hubo_to_spin_ham(T)

        if verbose:
            print(f"  Hamiltonian terms: {spin_ham.term_count}")

        best_ising_energy = -1.0
        classical_bitstring = '0' * cur_n_nodes

        if use_brute_force and cur_n_nodes <= 20:
            best_ising_energy, best_hubo_energy, classical_bitstring = \
                brute_force_hubo(T)
            if verbose:
                print(f"  Brute-force HUBO minimum: {best_hubo_energy:.4f}"
                      f"  (Ising: {best_ising_energy:.4f})")
        elif use_simulated_annealing:
            best_ising_energy, best_hubo_energy, classical_bitstring = \
                simulated_annealing_hubo(T, seed=seed_adapt)
            if verbose:
                print(f"  SA HUBO minimum: {best_hubo_energy:.4f}"
                      f"  (Ising: {best_ising_energy:.4f})")

        true_energy = best_ising_energy
        if abs(true_energy) < 1e-8:
            true_energy = -1e-8

        for gamma_0 in init_gamma:
            for trial_num in range(trials_per_hubo):

                if verbose:
                    print(f"  ADAPT-QAOA trial {trial_num + 1}, gamma_0={gamma_0}")

                start_time = time.time()

                adapt_result = adapt_qaoa_run(
                    spin_ham,
                    cur_n_nodes,
                    pool=op_pool,
                    gamma_0=gamma_0,
                    norm_threshold=norm_threshold,
                    energy_threshold=energy_threshold,
                    approx_ratio=approx_ratio,
                    true_energy=true_energy,
                    optimizer=optimizer,
                    max_iter=max_iter,
                    seed_adapt=seed_adapt,
                    verbose=verbose,
                )

                elapsed_time = time.time() - start_time

                adapt_result = list(adapt_result)
                adapt_result[2] = [int(i) + 1 for i in adapt_result[2]]
                adapt_result = tuple(adapt_result)

                if verbose:
                    print(f"  Completed in {elapsed_time:.2f}s  |  "
                          f"approx_ratio={adapt_result[5]:.4f}  |  "
                          f"bitstring={adapt_result[6]}")

                result_rows.append({
                    'method': 'ADAPT-QAOA-HUBO',
                    'graph_name': cur_hubo_name,
                    'graph_num': hubo_num + 1,
                    'trial_num': trial_num + 1,
                    'n_nodes': cur_n_nodes,
                    'init_gamma': gamma_0,
                    'optimizer': optimizer,
                    'pool_type': op_pool,
                    'edge_weight_scaling_coef': 1.0,
                    'edge_weight_norm_coef': 1.0,
                    'energy_list': adapt_result[0],
                    'true_energy': true_energy,
                    'mixer_pool_pauli_word': adapt_result[1],
                    'mixer_pool_index': adapt_result[2],
                    'gamma_coef': adapt_result[3],
                    'beta_coef': adapt_result[4],
                    'approx_ratio': adapt_result[5],
                    'cut_adapt': adapt_result[6],
                    'cut_classical': classical_bitstring,
                    'num_layers': adapt_result[7],
                    'optimizer_success_flag': adapt_result[8],
                    'elapsed_time': elapsed_time,
                })

                if adapt_result[5] >= approx_ratio:
                    if verbose:
                        print(f"  Approx ratio {adapt_result[5]:.4f} reached.")
                    break

            if adapt_result[5] >= approx_ratio and not multi_gamma:
                break

        if checkpoint_every and (hubo_num + 1) % checkpoint_every == 0:
            pd.DataFrame(hubo_rows).to_csv(hubo_csv_path, index=False)
            pd.DataFrame(result_rows).to_csv(results_csv_path, index=False)
            if verbose:
                print(f"  [checkpoint] Saved {hubo_num + 1}/{hubo_number} HUBOs "
                      f"to {hubo_csv_path}")

    hubo_df = pd.DataFrame(hubo_rows)
    results_df = pd.DataFrame(result_rows)

    hubo_df.to_csv(hubo_csv_path, index=False)
    results_df.to_csv(results_csv_path, index=False)

    if verbose:
        print("\nData generation completed.")
        print(f"  HUBO metadata: {len(hubo_df)} rows")
        print(f"  ADAPT results: {len(results_df)} rows")

    return

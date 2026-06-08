"""
IonQ-compatible version of sample_ansatz_bitstring_only_hubo.

The standard util_hubo.py uses @cudaq.kernel with dynamic Python loops
(for p in range(num_layer), for i in range(len(ham_coeffs)), etc.).
The IonQ backend JIT-compiler requires statically-unrollable loops and
aborts with "must be counted loop" when it encounters dynamic bounds.

Fix: use cudaq.make_kernel() (builder API) where loops run in Python and
gates are added one by one.  The resulting circuit has no loops at all —
just a flat sequence of gates — which the IonQ compiler handles fine.
"""

import os
import sys

import cudaq
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.hubo_hamiltonian import hubo_to_spin_ham
from qaoa_gpt_src.adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double


def sample_ansatz_bitstring_only_hubo_ionq(T, q_circuit, n_nodes,
                                            pool_type='all_pool',
                                            num_shots=1024, verbose=False):
    """
    IonQ-compatible circuit sampler using the CUDA-Q kernel builder API.
    Drop-in replacement for util_hubo.sample_ansatz_bitstring_only_hubo.
    """
    spin_ham   = hubo_to_spin_ham(T)
    ham_coeffs = term_coefficients(spin_ham)   # list[complex]
    ham_words  = term_words(spin_ham, n_nodes) # list[str], e.g. "ZZIIIZ"

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
    gamma_coef = []
    beta_coef  = []
    for j in range(0, len(q_circuit), 4):
        op_indices.append(q_circuit[j + 1] - 1)
        gamma_coef.append(q_circuit[j + 2])
        beta_coef.append(q_circuit[j + 3])

    # mixer_pool[layer] = list of Pauli word strings (not cudaq.pauli_word)
    mixer_pool = []
    for idx in op_indices:
        if idx >= len(pools):
            raise ValueError(
                f"Operator index {idx} out of range for pool size {len(pools)}")
        op = pools[idx]
        layer_ops = [term.get_pauli_word(n_nodes) for term in op]  # plain strings
        mixer_pool.append(layer_ops)

    num_layer = len(mixer_pool)

    if verbose:
        print(f"  num_layer={num_layer}, beta={beta_coef}, gamma={gamma_coef}")

    # --- Builder API: build a flat gate sequence, no dynamic loops in kernel ---
    kernel  = cudaq.make_kernel()
    qubits  = kernel.qalloc(n_nodes)

    # Initial Hadamard on all qubits
    kernel.h(qubits)

    # Cost + mixer layers — loops run in Python, gates appended one by one
    for p in range(num_layer):
        # Cost unitary: e^{-i gamma H}
        for i in range(len(ham_coeffs)):
            angle = float(gamma_coef[p] * ham_coeffs[i].real)
            kernel.exp_pauli(angle, qubits, ham_words[i])
        # Mixer unitary: e^{-i beta P}
        for pw_str in mixer_pool[p]:
            kernel.exp_pauli(float(beta_coef[p]), qubits, pw_str)

    result = cudaq.sample(kernel, shots_count=int(num_shots))
    return result.most_probable()

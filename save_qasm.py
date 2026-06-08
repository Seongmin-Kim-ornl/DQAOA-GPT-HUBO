"""
save_qasm.py — Save a GPT-predicted ADAPT-QAOA circuit as OpenQASM 2.0.

Sign conventions
----------------
cudaq : exp_pauli(θ, qubits, P_big) = exp(+i·θ·P_big)   big-endian (leftmost = qubit 0)
Qiskit: PauliEvolutionGate(op, time=t) = exp(−i·t·op)   little-endian (rightmost = qubit 0)

Conversion:
  P_little = P_big[::-1]   (reverse to switch endianness)
  Cost  layer: cudaq exp_pauli(+γ·c, q, P) → Qiskit time = −(γ·c)
  Mixer layer: cudaq exp_pauli(+β,   q, P) → Qiskit time = −β

Usage:
    python save_qasm.py \\
        --ckpt nanoGPT/out-hubo_save_dir/ckpt_33000.pt \\
        --data_dir nanoGPT/data/hubo_save_dir \\
        --n_nodes 8 --num_gpt_samples 10 --output circuit.qasm

    # From formulated HUBO:
    python save_qasm.py --ckpt ... --n_nodes 8 \\
        --from_formulated --hubo_size 100 --output circuit.qasm

    # Save HUBO tensor alongside for run_qasm_aer.py:
    python save_qasm.py --ckpt ... --n_nodes 8 --output circuit.qasm --save_hubo hubo.npy
"""

import os
import sys
import argparse
import contextlib
import io
import pickle
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    evaluate_hubo_energy,
    brute_force_hubo,
    hubo_to_spin_ham,
)
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.adapt_qaoa_pool import all_pool
from utils.solve_HUBO_GPT import _get_hubo_gpt_model
from qaoa_gpt_src.util_hubo import solve_hubo_circuits_with_bitstrings, circ_sanity_check


# ---------------------------------------------------------------------------
# Pool / circuit helpers
# ---------------------------------------------------------------------------

def _pair_from_idx(flat_idx, n):
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if k == flat_idx:
                return i, j
            k += 1
    return 0, 1


def _pool_label(idx, n_nodes):
    n = n_nodes
    if idx < n:
        return f"X_{idx}"
    idx -= n
    if idx == 0:
        return "ΣX"
    idx -= 1
    kind_map = ["XX", "YY", "YZ", "ZY"]
    pair_idx = idx // 4
    kind_idx = idx % 4
    i, j = _pair_from_idx(pair_idx, n)
    return f"{kind_map[kind_idx]}_{i}{j}"


def _decode_circuit(q_circuit, n_nodes):
    layers = []
    for j in range(0, len(q_circuit), 4):
        op_idx = q_circuit[j + 1] - 1
        gamma  = float(q_circuit[j + 2])
        beta   = float(q_circuit[j + 3])
        label  = _pool_label(op_idx, n_nodes)
        layers.append((label, beta, gamma))
    return layers


# ---------------------------------------------------------------------------
# Qiskit circuit builder
# ---------------------------------------------------------------------------

def build_qiskit_circuit(T_rounded, best_circ, n_nodes, pool_type='all_pool'):
    """
    Build a Qiskit QuantumCircuit implementing the ADAPT-QAOA circuit.

    cudaq kernel structure (per layer p):
        exp_pauli(+gamma[p]*c_i, qubits, P_i)   ← cost unitary
        exp_pauli(+beta[p],      qubits, P_mix)  ← mixer

    Qiskit PauliEvolutionGate: exp(-i*t*P_little)
      → P_little  = P_big[::-1]
      → cost time = -(gamma[p] * c_i)
      → mix  time = -beta[p]
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    spin_ham  = hubo_to_spin_ham(T_rounded)
    raw_words = term_words(spin_ham, n_nodes)    # big-endian strings (leftmost = qubit 0)
    raw_coefs = term_coefficients(spin_ham)

    pools = all_pool(n_nodes)

    op_indices, gamma_coef, beta_coef = [], [], []
    for j in range(0, len(best_circ), 4):
        op_indices.append(best_circ[j + 1] - 1)
        gamma_coef.append(float(best_circ[j + 2]))
        beta_coef.append(float(best_circ[j + 3]))

    num_layer = len(op_indices)

    qc = QuantumCircuit(n_nodes)
    qc.h(range(n_nodes))

    for p in range(num_layer):
        # Cost unitary: exp(+i * gamma[p] * c * P) → time = -(gamma * c)
        for coef, word in zip(raw_coefs, raw_words):
            w_big = word if isinstance(word, str) else ''.join(word)
            if set(w_big) == {'I'}:
                continue
            theta = gamma_coef[p] * float(np.real(coef))
            if abs(theta) < 1e-12:
                continue
            w_little = w_big[::-1]
            gate = PauliEvolutionGate(SparsePauliOp(w_little), time=-theta)
            qc.append(gate, range(n_nodes))

        # Mixer: exp(+i * beta[p] * P_mix) → time = -beta[p]
        if abs(beta_coef[p]) < 1e-12:
            continue
        if op_indices[p] < 0 or op_indices[p] >= len(pools):
            continue
        for term in pools[op_indices[p]]:
            w_big = term.get_pauli_word(n_nodes)   # big-endian: leftmost = qubit 0
            if set(w_big) == {'I'}:
                continue
            w_little = w_big[::-1]
            gate = PauliEvolutionGate(SparsePauliOp(w_little), time=-beta_coef[p])
            qc.append(gate, range(n_nodes))

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Save GPT-predicted ADAPT-QAOA circuit as OpenQASM 2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",            type=str, required=True)
    p.add_argument("--data_dir",        type=str, default=None)
    p.add_argument("--config_file",     type=str, default=None)
    p.add_argument("--n_nodes",         type=int, default=6)
    p.add_argument("--seed",            type=int, default=None)
    p.add_argument("--density",         type=float, default=0.5)
    p.add_argument("--weight_range",    type=float, default=1.0)
    p.add_argument("--from_formulated", action="store_true")
    p.add_argument("--hubo_size",       type=int, default=None)
    p.add_argument("--formulated_dir",  type=str, default=None)
    p.add_argument("--hubo_npy",        type=str, default=None)
    p.add_argument("--device",          type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--num_gpt_samples", type=int, default=10)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--pool_type",       type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])
    p.add_argument("--num_shots",       type=int, default=1024)
    p.add_argument("--no_brute_force",  action="store_true")
    p.add_argument("--output",          type=str, default="circuit.qasm")
    p.add_argument("--save_hubo",       type=str, default=None,
                   help="Save the (rounded) HUBO tensor as .npy for use with run_qasm_aer.py")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = parse_args()
    ckpt        = args.ckpt
    data_dir    = args.data_dir    or os.path.dirname(os.path.abspath(ckpt))
    config_file = args.config_file or os.path.join(data_dir, 'train_adapt_gpt_config.py')

    for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                        (config_file, "config_file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # ---- HUBO ---------------------------------------------------------------
    import random
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Using seed: {seed}")

    if args.from_formulated:
        if args.hubo_size is None:
            print("ERROR: --hubo_size required with --from_formulated")
            sys.exit(1)
        from subHUBO_gen import formulate_HUBO, make_sub_HUBO
        T_full = formulate_HUBO(args.hubo_size, data_dir=args.formulated_dir)
        random.seed(seed)
        T, _ = make_sub_HUBO(T_full, args.n_nodes)
        print(f"HUBO: sub-HUBO (n={args.n_nodes}) from formulated (size={args.hubo_size})")
    elif args.hubo_npy:
        from subHUBO_gen import make_sub_HUBO
        T_full = np.load(args.hubo_npy)
        random.seed(seed)
        T, _ = make_sub_HUBO(T_full, args.n_nodes)
        print(f"HUBO: sub-HUBO (n={args.n_nodes}) from {args.hubo_npy}")
    else:
        rng = np.random.default_rng(seed)
        T = generate_random_hubo(
            args.n_nodes, density=args.density,
            weight_range=args.weight_range,
            seed=int(rng.integers(0, 2**31)),
        )
        print(f"HUBO: random n_nodes={args.n_nodes}")

    # ---- Reference energy ---------------------------------------------------
    energy_ref = None
    if not args.no_brute_force and args.n_nodes <= 20:
        _, energy_ref, _ = brute_force_hubo(T)
        print(f"Brute-force minimum: {energy_ref:.6f}")

    # ---- GPT inference ------------------------------------------------------
    cfg_text = open(config_file).read()
    _rd = re.search(r"token_seq_round\s*=\s*['\"].*_d(\d+)['\"]", cfg_text)
    rounding_digits = int(_rd.group(1)) if _rd else 2
    T_rounded = np.round(T.copy(), rounding_digits)

    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    vocab_floats = np.array(sorted(
        [k for k in stoi if isinstance(k, float) and not isinstance(k, bool)]))
    if len(vocab_floats) > 0:
        vf_min, vf_max = vocab_floats[0], vocab_floats[-1]
        def _snap(val):
            if val == 0.0:
                return 0.0
            clipped = float(np.clip(val, vf_min, vf_max))
            return float(vocab_floats[np.argmin(np.abs(vocab_floats - clipped))])
        T_rounded = np.vectorize(_snap)(T_rounded)

    print(f"Running GPT inference  (num_gpt_samples={args.num_gpt_samples})...")

    @contextlib.contextmanager
    def _silence():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield

    from tqdm import tqdm
    tqdm.disable = True

    with _silence():
        gpt_model = _get_hubo_gpt_model(ckpt, data_dir, config_file, args.device)
        circ_df = gpt_model.generate_circ_from_sub_hubos(
            {'sub_hubo_0': T_rounded},
            n_samples_per_batch=50,
            num_samples=args.num_gpt_samples,
            max_new_tokens=200,
            temperature=args.temperature,
            top_k=200,
            rounding_digits=rounding_digits,
        )

    # ---- Pick best circuit --------------------------------------------------
    all_circs = []
    for _, row in circ_df.iterrows():
        for c in row.get('q_circuits', []):
            if circ_sanity_check(c):
                all_circs.append(c)
        ac = row.get('adapt_circuit', [])
        if ac and circ_sanity_check(ac):
            all_circs.append(ac)

    if not all_circs:
        print("ERROR: No valid circuits produced by GPT.")
        sys.exit(1)

    results_df = solve_hubo_circuits_with_bitstrings(
        qaoa_gpt_circ_df=circ_df,
        T_dict={'sub_hubo_0': T_rounded},
        n_nodes=args.n_nodes,
        pool_type=args.pool_type,
        num_shots=args.num_shots,
        verbose=False,
    )

    best_energy_gpt = float('inf')
    best_bs   = '0' * args.n_nodes
    best_circ = all_circs[0]

    for _, row in results_df.iterrows():
        adapt_gpt_bs = list(row.get('adapt_gpt_bitstrings', []))
        adapt_bs     = row.get('ADAPT_bitstring', None)
        candidates   = adapt_gpt_bs + ([adapt_bs] if adapt_bs else [])

        circs_row = [c for c in row.get('q_circuits', []) if circ_sanity_check(c)]
        ac_row    = row.get('adapt_circuit', [])
        if ac_row and circ_sanity_check(ac_row):
            circs_row.append(ac_row)

        for k, bs_cand in enumerate(candidates):
            if len(bs_cand) != args.n_nodes:
                continue
            x_cand = np.array([int(b) for b in bs_cand], dtype=np.uint8)
            e_cand  = evaluate_hubo_energy(T, x_cand)
            if e_cand < best_energy_gpt:
                best_energy_gpt = e_cand
                best_bs         = bs_cand
                if k < len(circs_row):
                    best_circ = circs_row[k]

    layers = _decode_circuit(best_circ, args.n_nodes)

    print(f"\nBest circuit:")
    print(f"  GPT energy   : {best_energy_gpt:.6f}  bitstring={best_bs}")
    if energy_ref is not None:
        ar = best_energy_gpt / energy_ref if energy_ref < -1e-8 else float('nan')
        print(f"  Ref energy   : {energy_ref:.6f}")
        print(f"  Approx ratio : {ar:.4f}")
    print(f"  ADAPT layers : {len(layers)}")
    for li, (lbl, beta, gamma) in enumerate(layers):
        print(f"    L{li + 1:2d}  op={lbl:12s}  β={beta:.4f}  γ={gamma:.4f}")

    # ---- Build Qiskit circuit and export QASM --------------------------------
    print("\nBuilding Qiskit circuit and transpiling...")
    from qiskit import transpile
    from qiskit import qasm2

    qk_circ = build_qiskit_circuit(T_rounded, best_circ, args.n_nodes,
                                    pool_type=args.pool_type)
    basis = ['h', 'cx', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'id', 'u']
    qk_transpiled = transpile(qk_circ, basis_gates=basis, optimization_level=0)

    ops   = qk_transpiled.count_ops()
    n_cx  = ops.get('cx', 0)
    n_1q  = sum(v for k, v in ops.items() if k not in ('cx', 'measure'))
    print(f"Transpiled: {qk_transpiled.num_qubits} qubits  "
          f"1q={n_1q}  cx={n_cx}  depth={qk_transpiled.depth()}")

    qasm_str = qasm2.dumps(qk_transpiled)
    with open(args.output, 'w') as f:
        f.write(qasm_str)
    print(f"QASM saved → {args.output}")

    if args.save_hubo:
        np.save(args.save_hubo, T)
        print(f"HUBO saved → {args.save_hubo}")
        if energy_ref is not None:
            print(f"  (pass --ref_energy {energy_ref:.6f} to run_qasm_aer.py)")


if __name__ == "__main__":
    main()

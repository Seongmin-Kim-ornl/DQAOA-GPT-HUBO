"""
STEP 4-5 (analysis) — Inspect GPT-generated ADAPT-QAOA circuit structure.

Builds the circuit using the CUDA-Q builder API (same as step4-5_gpt_ionq.py)
but does NOT submit to IonQ.  Instead it prints:
  - ADAPT layer structure (operator index, gamma, beta)
  - Hamiltonian term count and Pauli word weights
  - Estimated 1q / 2q gate counts and circuit depth
  - Circuit diagram (cudaq.draw)

Uses qpp-cpu as backend — no GPU or IonQ credentials needed.

Usage:
    python step4-5_circuit_info.py \\
        --ckpt nanoGPT/out-hubo_6nodes/ckpt_6nodes.pt \\
        --data_dir nanoGPT/data/hubo_6nodes \\
        --n_nodes 6 --n_circuits 3 \\
        --from_formulated --hubo_size 100 --num_gpt_samples 1 \\
        --draw
"""

import os
import sys
import argparse
import contextlib
import io
import re
import pickle

os.environ["CUDA_QUANTUM_OPTIMIZATION_LEVEL"] = "0"
os.environ["CUDAQ_DISABLE_LOOP_UNROLL"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cudaq
cudaq.set_target("qpp-cpu")   # local CPU sim — no IonQ / GPU needed

from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.hubo_hamiltonian import (
    hubo_to_spin_ham,
    generate_random_hubo,
)
from qaoa_gpt_src.adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double
from qaoa_gpt_src.model_interface_hubo import HUBO_GPT
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Gate-count estimation
# ---------------------------------------------------------------------------

def pauli_weight(pw_str: str) -> int:
    """Number of non-identity Paulis in a Pauli word string (e.g. 'ZZIIIZ' → 3)."""
    return sum(1 for c in pw_str if c != 'I')


def gates_for_exp_pauli(pw_str: str):
    """
    Estimate 1q and 2q gate count for a single exp_pauli(angle, qubits, pw_str).

    Standard decomposition:
      - CNOT ladder:  2*(k-1) CNOTs   (k = Pauli weight)
      - Basis change: 2 gates per X qubit (H in + H out)
                      4 gates per Y qubit (Sdg+H in + H+S out)
                      0 gates per Z qubit
      - 1 Rz gate

    Returns (n_1q, n_2q).
    """
    k    = pauli_weight(pw_str)
    if k == 0:
        return 0, 0

    n_x  = pw_str.count('X')
    n_y  = pw_str.count('Y')
    n_2q = 2 * (k - 1)
    n_1q = n_x * 2 + n_y * 4 + 1   # basis-change gates + 1 Rz
    return n_1q, n_2q


def analyze_circuit(T, q_circuit, n_nodes, pool_type='all_pool'):
    """
    Build the circuit and return:
      - gate counts (1q, 2q)
      - depth estimate
      - the cudaq builder kernel (for drawing)
      - human-readable layer info
    """
    spin_ham   = hubo_to_spin_ham(T)
    ham_coeffs = term_coefficients(spin_ham)
    ham_words_ = term_words(spin_ham, n_nodes)   # list[str]

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
    beta_coef  = []
    gamma_coef = []
    for j in range(0, len(q_circuit), 4):
        op_indices.append(q_circuit[j + 1] - 1)
        beta_coef.append(q_circuit[j + 2])
        gamma_coef.append(q_circuit[j + 3])

    mixer_pool = []
    for idx in op_indices:
        if idx >= len(pools):
            raise ValueError(f"Op index {idx} out of range (pool size {len(pools)})")
        op = pools[idx]
        mixer_pool.append([term.get_pauli_word(n_nodes) for term in op])

    num_layer = len(mixer_pool)

    # --- Gate counting ---
    total_1q = n_nodes      # initial H layer
    total_2q = 0
    layer_info = []

    for p in range(num_layer):
        cost_1q = cost_2q = 0
        for i, pw in enumerate(ham_words_):
            a, b = gates_for_exp_pauli(pw)
            cost_1q += a
            cost_2q += b

        mix_1q = mix_2q = 0
        for pw in mixer_pool[p]:
            a, b = gates_for_exp_pauli(pw)
            mix_1q += a
            mix_2q += b

        total_1q += cost_1q + mix_1q
        total_2q += cost_2q + mix_2q

        layer_info.append({
            'layer':      p,
            'op_idx':     op_indices[p],
            'gamma':      gamma_coef[p],
            'beta':       beta_coef[p],
            'mixer_ops':  len(mixer_pool[p]),
            'cost_1q':    cost_1q,
            'cost_2q':    cost_2q,
            'mix_1q':     mix_1q,
            'mix_2q':     mix_2q,
        })

    # Rough depth estimate: sum of max(cost_2q, cost_1q) + mix layers
    # (assumes cost and mixer are sequential)
    depth_est = n_nodes   # H layer
    for info in layer_info:
        # In CNOT ladders gates are sequential → depth ≈ gate count
        depth_est += info['cost_2q'] + info['mix_2q']

    # --- Build kernel for drawing ---
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n_nodes)
    kernel.h(qubits)
    for p in range(num_layer):
        for i, pw in enumerate(ham_words_):
            angle = float(gamma_coef[p] * ham_coeffs[i].real)
            kernel.exp_pauli(angle, qubits, pw)
        for pw in mixer_pool[p]:
            kernel.exp_pauli(float(beta_coef[p]), qubits, pw)

    ham_info = {
        'n_terms':      len(ham_words_),
        'n_z':          sum(1 for w in ham_words_ if pauli_weight(w) == 1),
        'n_zz':         sum(1 for w in ham_words_ if pauli_weight(w) == 2),
        'n_zzz':        sum(1 for w in ham_words_ if pauli_weight(w) == 3),
    }

    return {
        'num_layer':   num_layer,
        'total_1q':    total_1q,
        'total_2q':    total_2q,
        'depth_est':   depth_est,
        'layer_info':  layer_info,
        'ham_info':    ham_info,
        'kernel':      kernel,
    }


# ---------------------------------------------------------------------------
# GPT model helpers (silent)
# ---------------------------------------------------------------------------

_model_cache = {}

def _get_model(ckpt, data_dir, config_file, device):
    key = (ckpt, data_dir, config_file, device)
    if key not in _model_cache:
        @contextlib.contextmanager
        def _silence():
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                yield
        with _silence():
            _model_cache[key] = HUBO_GPT(
                model_ckpt=ckpt,
                data_dir=data_dir,
                config_file=config_file,
                device=device,
            )
    return _model_cache[key]


def generate_circuits(T, ckpt, data_dir, config_file, device,
                      num_gpt_samples, temperature, rounding_digits):
    """Return list of (q_circuit, label) from GPT for one sub-HUBO."""
    @contextlib.contextmanager
    def _silence():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield

    tqdm.disable = True
    model = _get_model(ckpt, data_dir, config_file, device)
    T_r = np.round(T, rounding_digits)

    with _silence():
        df = model.generate_circ_from_sub_hubos(
            {'sub_hubo_0': T_r},
            n_samples_per_batch=50,
            num_samples=num_gpt_samples,
            max_new_tokens=150,
            temperature=temperature,
            top_k=200,
            rounding_digits=rounding_digits,
        )

    circuits = []
    for _, row in df.iterrows():
        for i, circ in enumerate(row.get('q_circuits', [])):
            circuits.append((circ, f'gpt_sample_{i}'))
        adapt = row.get('adapt_circuit', [])
        if adapt:
            circuits.append((adapt, 'adapt_circuit'))

    return circuits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect GPT-generated ADAPT-QAOA circuit structure (no execution)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",        type=str, required=True)
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--n_nodes",     type=int, default=6)
    p.add_argument("--n_circuits",  type=int, default=3,
                   help="Number of sub-HUBOs to inspect")
    p.add_argument("--density",     type=float, default=0.5)
    p.add_argument("--weight_range",type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=2024)
    p.add_argument("--from_formulated", action="store_true")
    p.add_argument("--hubo_size",   type=int,   default=None)
    p.add_argument("--hubo_npy",    type=str,   default=None)
    p.add_argument("--pool_type",   type=str,   default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])
    p.add_argument("--num_gpt_samples", type=int, default=1,
                   help="GPT samples per sub-HUBO (1 is enough for inspection)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device",      type=str,   default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--draw",        action="store_true",
                   help="Print circuit diagram (cudaq.draw) — can be large")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt        = args.ckpt
    data_dir    = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
    config_file = args.config_file or os.path.join(data_dir, 'train_adapt_gpt_config.py')

    for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                        (config_file, "config_file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    with open(config_file) as f:
        cfg = f.read()
    m = re.search(r'^\s*n_nodes\s*=\s*(\d+)', cfg, re.MULTILINE)
    rd_m = re.search(r"token_seq_round\s*=\s*['\"].*_d(\d+)['\"]", cfg)
    rounding_digits = int(rd_m.group(1)) if rd_m else 2

    # Load or generate sub-HUBOs
    if args.from_formulated:
        if args.hubo_size is None:
            print("ERROR: --hubo_size required with --from_formulated")
            sys.exit(1)
        from subHUBO_gen import formulate_HUBO, make_sub_HUBO
        T_full = formulate_HUBO(args.hubo_size)
        rng = np.random.default_rng(args.seed)
        tensors = [make_sub_HUBO(T_full, args.n_nodes)[0]
                   for _ in range(args.n_circuits)]
    elif args.hubo_npy:
        from subHUBO_gen import make_sub_HUBO
        T_full = np.load(args.hubo_npy)
        rng = np.random.default_rng(args.seed)
        tensors = [make_sub_HUBO(T_full, args.n_nodes)[0]
                   for _ in range(args.n_circuits)]
    else:
        rng = np.random.default_rng(args.seed)
        tensors = [
            generate_random_hubo(args.n_nodes, density=args.density,
                                 weight_range=args.weight_range,
                                 seed=int(rng.integers(0, 2**31)))
            for _ in range(args.n_circuits)
        ]

    print("=" * 70)
    print("ADAPT-QAOA Circuit Analysis  (no execution, qpp-cpu target)")
    print("=" * 70)
    print(f"  n_nodes     : {args.n_nodes}")
    print(f"  pool_type   : {args.pool_type}")
    print(f"  checkpoint  : {ckpt}")
    print("=" * 70)

    all_1q, all_2q, all_depth, all_layers = [], [], [], []

    for prob_idx, T in enumerate(tensors):
        print(f"\n{'─'*70}")
        print(f"  Sub-HUBO #{prob_idx + 1}")
        print(f"{'─'*70}")

        # Hamiltonian info
        spin_ham   = hubo_to_spin_ham(T)
        hw         = term_words(spin_ham, args.n_nodes)
        n_z   = sum(1 for w in hw if pauli_weight(w) == 1)
        n_zz  = sum(1 for w in hw if pauli_weight(w) == 2)
        n_zzz = sum(1 for w in hw if pauli_weight(w) == 3)
        print(f"  Hamiltonian: {len(hw)} terms  "
              f"(Z={n_z}, ZZ={n_zz}, ZZZ={n_zzz})")
        print(f"  Gate cost/cost-layer:  "
              f"1q≈{n_z*1 + n_zz*1 + n_zzz*1}  "
              f"2q≈{n_zz*2 + n_zzz*4}")

        # Generate GPT circuits
        try:
            circuits = generate_circuits(
                T, ckpt, data_dir, config_file, args.device,
                args.num_gpt_samples, args.temperature, rounding_digits)
        except Exception as e:
            print(f"  [ERROR generating circuits: {e}]")
            continue

        if not circuits:
            print("  [No valid circuits generated]")
            continue

        for circ, label in circuits:
            if not circ:
                continue
            try:
                info = analyze_circuit(T, circ, args.n_nodes, args.pool_type)
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")
                continue

            print(f"\n  [{label}]")
            print(f"    ADAPT layers : {info['num_layer']}")
            print(f"    Gate counts  : 1q={info['total_1q']}  2q={info['total_2q']}")
            print(f"    Depth (est.) : {info['depth_est']}")

            print(f"    Layer details:")
            for li in info['layer_info']:
                print(f"      layer {li['layer']+1:2d}  "
                      f"op_idx={li['op_idx']:3d}  "
                      f"γ={li['gamma']:+.4f}  β={li['beta']:+.4f}  "
                      f"mixer_terms={li['mixer_ops']}  "
                      f"cost(1q={li['cost_1q']},2q={li['cost_2q']})  "
                      f"mixer(1q={li['mix_1q']},2q={li['mix_2q']})")

            all_1q.append(info['total_1q'])
            all_2q.append(info['total_2q'])
            all_depth.append(info['depth_est'])
            all_layers.append(info['num_layer'])

            if args.draw:
                print(f"\n  Circuit diagram ({label}):")
                try:
                    diagram = cudaq.draw(info['kernel'])
                    for line in diagram.splitlines():
                        print(f"    {line}")
                except Exception as e:
                    print(f"    [draw failed: {e}]")

    if all_1q:
        print(f"\n{'='*70}")
        print("Summary across all circuits")
        print(f"{'='*70}")
        print(f"  ADAPT layers : mean={np.mean(all_layers):.1f}  "
              f"min={np.min(all_layers)}  max={np.max(all_layers)}")
        print(f"  1q gates     : mean={np.mean(all_1q):.1f}  "
              f"min={np.min(all_1q)}  max={np.max(all_1q)}")
        print(f"  2q gates     : mean={np.mean(all_2q):.1f}  "
              f"min={np.min(all_2q)}  max={np.max(all_2q)}")
        print(f"  Depth (est.) : mean={np.mean(all_depth):.1f}  "
              f"min={np.min(all_depth)}  max={np.max(all_depth)}")
        print()
        print("  Note: gate counts are estimated from Pauli word decomposition.")
        print("  IonQ native gate counts may differ (compiler further decomposes")
        print("  exp_pauli into √XX + rotation gates).")


if __name__ == "__main__":
    main()

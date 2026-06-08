"""
run_qasm_aer.py — Simulate an OpenQASM 2 circuit with qiskit-aer.

Loads a .qasm file produced by save_qasm.py, runs it on AerSimulator,
and reports the measurement distribution.  Optionally evaluates HUBO energies
if a HUBO tensor (.npy) is provided.

Bitstring convention
--------------------
Qiskit returns bitstrings in little-endian order (q0 = rightmost character).
This script reverses them to big-endian (q0 = leftmost) to match the cudaq
convention used in the rest of the pipeline.

Usage:
    # Basic: just run and show top bitstrings
    python run_qasm_aer.py --qasm circuit.qasm --shots 4096

    # With HUBO tensor for energy evaluation
    python run_qasm_aer.py --qasm circuit.qasm --shots 4096 --hubo_npy hubo.npy

    # With known reference energy for AR
    python run_qasm_aer.py --qasm circuit.qasm --shots 4096 \\
        --hubo_npy hubo.npy --ref_energy -16.0308

    # Statevector (exact, no shot noise)
    python run_qasm_aer.py --qasm circuit.qasm --method statevector
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate a QASM circuit with qiskit-aer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qasm",       type=str, required=True,
                   help="Path to the OpenQASM 2 file")
    p.add_argument("--shots",      type=int, default=4096,
                   help="Number of measurement shots")
    p.add_argument("--method",     type=str, default="automatic",
                   choices=["automatic", "statevector", "stabilizer",
                            "extended_stabilizer", "matrix_product_state"],
                   help="AerSimulator simulation method")
    p.add_argument("--top_k",      type=int, default=10,
                   help="Number of top bitstrings to display")
    p.add_argument("--hubo_npy",   type=str, default=None,
                   help="Path to a .npy HUBO tensor for energy evaluation")
    p.add_argument("--ref_energy", type=float, default=None,
                   help="Reference (optimal) energy for approximation ratio. "
                        "Ignored unless --hubo_npy is also provided.")
    p.add_argument("--n_nodes",    type=int, default=None,
                   help="Number of qubits/variables. Inferred from QASM if omitted.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.qasm):
        print(f"ERROR: QASM file not found: {args.qasm}")
        sys.exit(1)

    # ---- Load QASM ----------------------------------------------------------
    from qiskit import qasm2, transpile
    print(f"Loading QASM: {args.qasm}")
    qc = qasm2.load(args.qasm)
    n_nodes = args.n_nodes or qc.num_qubits
    print(f"  Qubits : {qc.num_qubits}")
    print(f"  Depth  : {qc.depth()}")
    print(f"  Ops    : {qc.count_ops()}")

    # ---- Load HUBO tensor (optional) ----------------------------------------
    T = None
    if args.hubo_npy:
        if not os.path.exists(args.hubo_npy):
            print(f"ERROR: HUBO tensor not found: {args.hubo_npy}")
            sys.exit(1)
        T = np.load(args.hubo_npy)
        print(f"  HUBO   : {T.shape}")

        if args.ref_energy is None and T.shape[0] <= 20:
            from qaoa_gpt_src.hubo_hamiltonian import brute_force_hubo
            _, ref_e, _ = brute_force_hubo(T)
            args.ref_energy = ref_e
            print(f"  Brute-force minimum: {ref_e:.6f}")

    # ---- Simulate -----------------------------------------------------------
    from qiskit_aer import AerSimulator

    print(f"\nRunning AerSimulator  method={args.method}  shots={args.shots}...")
    sim = AerSimulator(method=args.method)

    if not any(qc.count_ops().get(op, 0) for op in ['measure']):
        qc_run = qc.copy()
        qc_run.measure_all()
    else:
        qc_run = qc

    job    = sim.run(qc_run, shots=args.shots)
    counts = job.result().get_counts()

    # Reverse Qiskit little-endian bitstrings → big-endian (cudaq convention)
    counts_be = {}
    for bs_le, cnt in counts.items():
        bs_be = bs_le.replace(' ', '')[::-1]
        counts_be[bs_be] = counts_be.get(bs_be, 0) + cnt

    total = sum(counts_be.values())
    print(f"Unique bitstrings measured: {len(counts_be)}")

    # ---- Report -------------------------------------------------------------
    top = sorted(counts_be.items(), key=lambda kv: -kv[1])[:args.top_k]

    if T is not None:
        from qaoa_gpt_src.hubo_hamiltonian import evaluate_hubo_energy
        header = f"{'Rank':>4}  {'Bitstring':<{n_nodes}}  {'Count':>6}  {'Prob':>7}  {'Energy':>12}"
        if args.ref_energy is not None:
            header += f"  {'AR':>7}"
        print(f"\n{header}")
        print("-" * len(header))
        for rank, (bs, cnt) in enumerate(top, 1):
            x    = np.array([int(b) for b in bs], dtype=np.uint8)
            e    = evaluate_hubo_energy(T, x)
            prob = cnt / total
            line = f"{rank:>4}  {bs:<{n_nodes}}  {cnt:>6}  {prob:>7.4f}  {e:>12.6f}"
            if args.ref_energy is not None:
                ar = e / args.ref_energy if args.ref_energy < -1e-8 else float('nan')
                line += f"  {ar:>7.4f}"
            print(line)

        most_prob_bs = top[0][0]
        x_mp  = np.array([int(b) for b in most_prob_bs], dtype=np.uint8)
        e_mp  = evaluate_hubo_energy(T, x_mp)
        print(f"\nMost-probable bitstring : {most_prob_bs}  E={e_mp:.6f}")
        if args.ref_energy is not None:
            ar_mp = e_mp / args.ref_energy if args.ref_energy < -1e-8 else float('nan')
            print(f"Approx ratio            : {ar_mp:.4f}")

        best_bs, best_e = min(
            ((bs, evaluate_hubo_energy(T, np.array([int(b) for b in bs], dtype=np.uint8)))
             for bs in counts_be),
            key=lambda kv: kv[1],
        )
        print(f"Best energy bitstring   : {best_bs}  E={best_e:.6f}")
        if args.ref_energy is not None:
            ar_best = best_e / args.ref_energy if args.ref_energy < -1e-8 else float('nan')
            print(f"Approx ratio (best)     : {ar_best:.4f}")

    else:
        header = f"{'Rank':>4}  {'Bitstring':<{n_nodes}}  {'Count':>6}  {'Prob':>7}"
        print(f"\n{header}")
        print("-" * len(header))
        for rank, (bs, cnt) in enumerate(top, 1):
            prob = cnt / total
            print(f"{rank:>4}  {bs:<{n_nodes}}  {cnt:>6}  {prob:>7.4f}")

        most_prob_bs = top[0][0]
        print(f"\nMost-probable bitstring : {most_prob_bs}")


if __name__ == "__main__":
    main()

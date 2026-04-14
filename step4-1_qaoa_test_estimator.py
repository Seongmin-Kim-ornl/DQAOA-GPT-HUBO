"""
STEP 4-1 — Test standard QAOA on sub-HUBOs (comparison baseline).

Mirrors step4_inference_test.py but uses standard QAOA (CudaQ COBYLA optimizer)
instead of GPT inference.  Use this as a baseline AR to compare against GPT.

Usage:
    python step4-1_qaoa_test.py \\
        --n_nodes 8 --n_test 100 --trim_n 5 \\
        --from_formulated --hubo_size 100 \\
        --num_layers 2 --max_iters 200

    python step4-1_qaoa_test.py \\
        --n_nodes 8 --n_test 50 --trim_n 5 \\
        --npy_dir subhubo_npy/ \\
        --num_layers 3 --cudaq_target qpp-cpu
"""

import os
import sys

# Sniff --cudaq_target before any cudaq-importing module is loaded.
if "CUDAQ_TARGET" not in os.environ:
    for _i, _a in enumerate(sys.argv):
        if _a == "--cudaq_target" and _i + 1 < len(sys.argv):
            os.environ["CUDAQ_TARGET"] = sys.argv[_i + 1]
            break
        if _a.startswith("--cudaq_target="):
            os.environ["CUDAQ_TARGET"] = _a.split("=", 1)[1]
            break

import argparse
import glob
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    evaluate_hubo_energy,
    brute_force_hubo,
    simulated_annealing_hubo,
)
from utils.qaoa_CudaQ_hubo import solve_HUBO_CUDAQ


def parse_args():
    p = argparse.ArgumentParser(
        description="Test standard QAOA on sub-HUBOs (trimmed stats baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Test problems
    p.add_argument("--n_nodes",      type=int,   default=8,
                   help="Sub-HUBO size")
    p.add_argument("--n_test",       type=int,   default=10,
                   help="Number of random sub-HUBOs to test")
    p.add_argument("--density",      type=float, default=0.5,
                   help="Interaction density for random HUBOs")
    p.add_argument("--weight_range", type=float, default=1.0,
                   help="Coefficient range U(-w, w)")
    p.add_argument("--seed",         type=int,   default=2024,
                   help="Random seed for test HUBO generation")
    p.add_argument("--npy_dir",      type=str,   default=None,
                   help="Directory with pre-saved .npy sub-HUBO tensors")

    # From real (formulated) HUBO
    p.add_argument("--from_formulated", action="store_true",
                   help="Draw sub-HUBOs from the real full HUBO via subHUBO_gen.formulate_HUBO")
    p.add_argument("--hubo_size",    type=int,   default=None,
                   help="Full HUBO size (required when --from_formulated is used)")
    p.add_argument("--hubo_npy",     type=str,   default=None,
                   help="Load full HUBO tensor from a .npy file and draw sub-HUBOs from it")

    # QAOA solver settings
    p.add_argument("--num_layers",   type=int,   default=2,
                   help="QAOA circuit depth p")
    p.add_argument("--max_iters",    type=int,   default=200,
                   help="Max COBYLA iterations per sub-HUBO")
    p.add_argument("--num_shots",       type=int,   default=1024,
                   help="CUDAQ sampling shots for final bitstring extraction")
    p.add_argument("--estimator_shots", type=int,   default=None,
                   help="Shots for ⟨H⟩ estimation during COBYLA (None=exact statevector). "
                        "Set to 10000 to match Qiskit Estimator behavior.")
    p.add_argument("--cudaq_target", type=str,   default="nvidia",
                   choices=["nvidia", "qpp-cpu"],
                   help="CUDAQ-Q backend")
    p.add_argument("--reverse_bitorder", action="store_true",
                   help="Reverse CUDAQ bitstring ordering")

    # Reference solver
    p.add_argument("--no_brute_force", action="store_true",
                   help="Skip brute-force reference (auto-skipped for n_nodes > 20)")
    p.add_argument("--no_sa",          action="store_true",
                   help="Skip SA reference")

    # Trimmed statistics
    p.add_argument("--trim_n", type=int, default=5,
                   help="Remove worst and best N cases from trimmed summary stats")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("STEP 4-1: Standard QAOA baseline test (trimmed statistics)")
    print("=" * 60)
    print(f"  n_nodes     : {args.n_nodes}")
    print(f"  num_layers  : {args.num_layers}")
    print(f"  max_iters   : {args.max_iters}  (COBYLA per sub-HUBO)")
    print(f"  cudaq_target: {args.cudaq_target}")
    print(f"  trim_n      : {args.trim_n}")
    print("=" * 60)

    # ---- Load or generate test HUBOs ----------------------------------------
    if args.from_formulated:
        if args.hubo_size is None:
            print("ERROR: --hubo_size is required when using --from_formulated")
            sys.exit(1)
        from subHUBO_gen import formulate_HUBO, make_sub_HUBO
        print(f"Loading formulated HUBO (size={args.hubo_size})...")
        T_full = formulate_HUBO(args.hubo_size)
        rng = np.random.default_rng(args.seed)
        test_tensors = []
        for _ in range(args.n_test):
            sub_T, _ = make_sub_HUBO(T_full, args.n_nodes)
            test_tensors.append(sub_T)
        print(f"Drew {args.n_test} sub-HUBOs (n_nodes={args.n_nodes}) "
              f"from formulated HUBO (seed={args.seed})")

    elif args.hubo_npy:
        from subHUBO_gen import make_sub_HUBO
        print(f"Loading full HUBO tensor from: {args.hubo_npy}")
        T_full = np.load(args.hubo_npy)
        rng = np.random.default_rng(args.seed)
        test_tensors = []
        for _ in range(args.n_test):
            sub_T, _ = make_sub_HUBO(T_full, args.n_nodes)
            test_tensors.append(sub_T)
        print(f"Drew {args.n_test} sub-HUBOs (n_nodes={args.n_nodes}) "
              f"from {args.hubo_npy} (seed={args.seed})")

    elif args.npy_dir:
        npy_files = sorted(glob.glob(os.path.join(args.npy_dir, "*.npy")))[:args.n_test]
        if not npy_files:
            print(f"ERROR: no .npy files found in {args.npy_dir}")
            sys.exit(1)
        test_tensors = [np.load(f) for f in npy_files]
        print(f"Loaded {len(test_tensors)} sub-HUBOs from {args.npy_dir}")

    else:
        rng = np.random.default_rng(args.seed)
        test_tensors = [
            generate_random_hubo(
                args.n_nodes,
                density=args.density,
                weight_range=args.weight_range,
                seed=int(rng.integers(0, 2**31)),
            )
            for _ in range(args.n_test)
        ]
        print(f"Generated {args.n_test} random sub-HUBOs (seed={args.seed})")

    # Add one warm-up problem at the front (discarded from stats)
    test_tensors = [test_tensors[0]] + test_tensors

    # ---- Run QAOA and collect stats -----------------------------------------
    print(f"\nRunning QAOA on {len(test_tensors) - 1} sub-HUBOs "
          f"(+1 warm-up discarded)...\n")

    results = []
    for idx, T in enumerate(test_tensors):

        t_start = time.time()
        x_qaoa = solve_HUBO_CUDAQ(
            T,
            num_layers=args.num_layers,
            max_iters=args.max_iters,
            shots=args.num_shots,
            estimator_shots=args.estimator_shots,
            target=args.cudaq_target,
            reverse_bitorder=args.reverse_bitorder,
        )
        t_elapsed = time.time() - t_start
        e_qaoa = evaluate_hubo_energy(T, x_qaoa)

        # Reference solutions
        e_ref = None
        e_sa  = None

        if not args.no_brute_force and args.n_nodes <= 20:
            _, e_bf, _ = brute_force_hubo(T)
            e_ref = e_bf

        if not args.no_sa:
            _, e_sa_val, _ = simulated_annealing_hubo(T, seed=42)
            e_sa = e_sa_val
            if e_ref is None:
                e_ref = e_sa

        approx_ratio = None
        if e_ref is not None and abs(e_ref) > 1e-8:
            if e_ref < 0:
                approx_ratio = e_qaoa / e_ref
            else:
                approx_ratio = e_ref / e_qaoa if e_qaoa > 0 else float('nan')

        results.append({
            'idx':          idx + 1,
            'e_qaoa':       e_qaoa,
            'e_ref':        e_ref,
            'e_sa':         e_sa,
            'approx_ratio': approx_ratio,
            'bitstring':    ''.join(str(int(b)) for b in x_qaoa),
            'time':         t_elapsed,
        })

        ref_str    = f"ref={e_ref:.4f}"    if e_ref        is not None else "ref=N/A"
        ar_str     = f"ar={approx_ratio:.4f}" if approx_ratio is not None else "ar=N/A"
        warmup_tag = " [warm-up]"          if idx == 0     else ""
        print(f"  [{idx:3d}]  qaoa={e_qaoa:.4f}  {ref_str}  {ar_str}  "
              f"bitstring={results[-1]['bitstring']}  time={t_elapsed:.2f}s{warmup_tag}")

    # ---- Summary (exclude warm-up at index 0) --------------------------------
    results_eval = results[1:]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    e_vals  = [r['e_qaoa'] for r in results_eval]
    e_ref_vals = [r['e_ref'] for r in results_eval if r['e_ref'] is not None]
    ar_vals = [r['approx_ratio'] for r in results_eval
               if r['approx_ratio'] is not None and not np.isnan(r['approx_ratio'])]

    print(f"  QAOA energy — mean: {np.mean(e_vals):.4f}  "
          f"std: {np.std(e_vals):.4f}  "
          f"min: {np.min(e_vals):.4f}")

    if e_ref_vals:
        print(f"  Ref energy  — mean: {np.mean(e_ref_vals):.4f}  "
              f"std: {np.std(e_ref_vals):.4f}  "
              f"min: {np.min(e_ref_vals):.4f}")

    if ar_vals:
        print(f"  Approx ratio (full) — mean: {np.mean(ar_vals):.4f}  "
              f"std: {np.std(ar_vals):.4f}  "
              f"min: {np.min(ar_vals):.4f}  n={len(ar_vals)}")

        if len(ar_vals) > 2 * args.trim_n:
            ar_sorted  = sorted(ar_vals)
            ar_trimmed = ar_sorted[args.trim_n : -args.trim_n]
            print(f"  Approx ratio (±{args.trim_n} trimmed) — "
                  f"mean: {np.mean(ar_trimmed):.4f}  "
                  f"std: {np.std(ar_trimmed):.4f}  "
                  f"min: {np.min(ar_trimmed):.4f}  n={len(ar_trimmed)}")
        else:
            print(f"  Trimmed AR skipped — need > {2 * args.trim_n} valid AR values "
                  f"(have {len(ar_vals)})")

    n_exact = sum(1 for r in results_eval
                  if r['approx_ratio'] is not None and r['approx_ratio'] >= 1.0 - 1e-4)
    print(f"  Exact (ar=1) — {n_exact}/{len(results_eval)}")

    t_vals = [r['time'] for r in results_eval]
    print(f"  Time/problem — mean: {np.mean(t_vals):.2f}s  "
          f"std: {np.std(t_vals):.2f}s  "
          f"min: {np.min(t_vals):.2f}s  max: {np.max(t_vals):.2f}s")
    print(f"  Warm-up time — {results[0]['time']:.2f}s (excluded from stats)")

    print("\nQAOA baseline test complete.")


if __name__ == "__main__":
    main()

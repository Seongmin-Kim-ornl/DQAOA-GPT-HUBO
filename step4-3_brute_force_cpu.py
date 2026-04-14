"""
STEP 4-3 — Brute-force optimal solution on a single CPU (no GPU).

Enumerates all 2^n bitstrings and returns the exact minimum energy.
Pure numpy — no cudaq, no torch, no GPU required.

Usage:
    python step4-3_brute_force_cpu.py \\
        --n_nodes 8 --n_test 100 --trim_n 5 \\
        --from_formulated --hubo_size 100

    python step4-3_brute_force_cpu.py \\
        --n_nodes 8 --n_test 50 --seed 42
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    brute_force_hubo,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Brute-force optimal solution for sub-HUBOs on a single CPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--n_nodes",      type=int,   default=8,
                   help="Sub-HUBO size (keep <= 20 for tractable brute force)")
    p.add_argument("--n_test",       type=int,   default=10,
                   help="Number of sub-HUBOs to solve")
    p.add_argument("--density",      type=float, default=0.5,
                   help="Interaction density for random HUBOs")
    p.add_argument("--weight_range", type=float, default=1.0,
                   help="Coefficient range U(-w, w)")
    p.add_argument("--seed",         type=int,   default=2024,
                   help="Random seed for test HUBO generation")
    p.add_argument("--npy_dir",      type=str,   default=None,
                   help="Directory with pre-saved .npy sub-HUBO tensors")

    p.add_argument("--from_formulated", action="store_true",
                   help="Draw sub-HUBOs from the real full HUBO via subHUBO_gen.formulate_HUBO")
    p.add_argument("--hubo_size",    type=int,   default=None,
                   help="Full HUBO size (required when --from_formulated is used)")
    p.add_argument("--hubo_npy",     type=str,   default=None,
                   help="Load full HUBO tensor from a .npy file and draw sub-HUBOs from it")

    p.add_argument("--trim_n",       type=int,   default=5,
                   help="Remove worst and best N energy values from trimmed summary")

    return p.parse_args()


def main():
    args = parse_args()

    if args.n_nodes > 20:
        print(f"WARNING: n_nodes={args.n_nodes} → 2^{args.n_nodes} = {2**args.n_nodes:,} states. "
              f"This may be very slow.")

    print("=" * 60)
    print("STEP 4-3: Brute-force baseline (single CPU, no GPU)")
    print("=" * 60)
    print(f"  n_nodes     : {args.n_nodes}  (2^{args.n_nodes} = {2**args.n_nodes} states)")
    print(f"  n_test      : {args.n_test}")
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

    # ---- Run brute force ----------------------------------------------------
    print(f"\nRunning brute force on {args.n_test} sub-HUBOs...\n")

    results = []
    for idx, T in enumerate(test_tensors):
        t_start = time.time()
        _, e_opt, bitstr = brute_force_hubo(T)
        t_elapsed = time.time() - t_start

        results.append({'idx': idx + 1, 'e_opt': e_opt,
                        'bitstring': bitstr, 'time': t_elapsed})

        print(f"  [{idx+1:3d}]  e_opt={e_opt:.4f}  "
              f"bitstring={bitstr}  time={t_elapsed:.8f}s")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    e_vals = [r['e_opt'] for r in results]
    print(f"  Optimal energy — mean: {np.mean(e_vals):.4f}  "
          f"std: {np.std(e_vals):.4f}  "
          f"min: {np.min(e_vals):.4f}  "
          f"max: {np.max(e_vals):.4f}")

    if len(e_vals) > 2 * args.trim_n:
        e_sorted  = sorted(e_vals)
        e_trimmed = e_sorted[args.trim_n : -args.trim_n]
        print(f"  Optimal energy (±{args.trim_n} trimmed) — "
              f"mean: {np.mean(e_trimmed):.4f}  "
              f"std: {np.std(e_trimmed):.4f}  n={len(e_trimmed)}")

    n_zero = sum(1 for e in e_vals if abs(e) < 1e-8)
    print(f"  Trivial (e=0) — {n_zero}/{len(results)}  "
          f"(all-zeros bitstring is optimal)")

    t_vals = [r['time'] for r in results]
    print(f"  Time/problem  — mean: {np.mean(t_vals):.8f}s  "
          f"std: {np.std(t_vals):.8f}s  "
          f"min: {np.min(t_vals):.8f}s  max: {np.max(t_vals):.8f}s")
    print(f"  Total time    — {sum(t_vals):.8f}s")

    print("\nBrute-force CPU baseline complete.")


if __name__ == "__main__":
    main()

"""
STEP 4-2 — Test inference on new unseen sub-HUBOs, executing ONLY the
GPT-generated candidate circuits (not the extra empty adapt_circuit slot
that step4_inference_test.py also samples on every problem).

Identical to step4_inference_test.py in every other respect (same CLI
args, same warm-up/trimmed-stats reporting) — the only difference is the
solver: this uses utils.solve_HUBO_GPT_gpt_only.solve_sub_HUBO_GPT_gpt_only
instead of utils.solve_HUBO_GPT.solve_sub_HUBO_GPT, so with
--num_gpt_samples 1 exactly 1 circuit runs per problem, not 2. See that
module's docstring for why the extra circuit exists in the original and
why it's excluded here rather than just excluded from timing.

Usage:
    python step4-2_inference_test_gpt_only.py \\
        --ckpt nanoGPT/out-hubo_save_dir/ckpt_10000.pt \\
        --data_dir nanoGPT/data/hubo_save_dir \\
        --n_nodes 4 --n_test 10 --trim_n 2
"""

import argparse
import os
import sys
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
from utils.solve_HUBO_GPT_gpt_only import solve_sub_HUBO_GPT_gpt_only


def parse_args():
    p = argparse.ArgumentParser(description="Test HUBO-GPT inference on sub-HUBOs, GPT circuits only")

    # Model paths
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to trained checkpoint (.pt file)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Tokenized data directory with meta.pkl (default: dirname of --ckpt)")
    p.add_argument("--config_file", type=str, default=None,
                   help="Config .py (default: <data_dir>/train_adapt_gpt_config.py)")

    # Test problems
    p.add_argument("--n_nodes", type=int, default=4,
                   help="Sub-HUBO size — must match training (default: 4)")
    p.add_argument("--n_test", type=int, default=10,
                   help="Number of random sub-HUBOs to test (default: 10)")
    p.add_argument("--density", type=float, default=0.5,
                   help="Interaction density for random HUBOs (default: 0.5)")
    p.add_argument("--weight_range", type=float, default=1.0,
                   help="Coefficient range U(-w,w) (default: 1.0)")
    p.add_argument("--seed", type=int, default=2024,
                   help="Random seed for test HUBO generation (default: 2024)")
    p.add_argument("--npy_dir", type=str, default=None,
                   help="Directory with pre-saved .npy sub-HUBO tensors to test instead")

    # From real (formulated) HUBO — same as step1 --from_formulated
    p.add_argument("--from_formulated", action="store_true",
                   help="Draw sub-HUBOs from the real full HUBO via subHUBO_gen.formulate_HUBO. "
                        "Requires --hubo_size.")
    p.add_argument("--hubo_size", type=int, default=None,
                   help="Full HUBO size (required when --from_formulated is used)")
    p.add_argument("--hubo_npy", type=str, default=None,
                   help="Load full HUBO tensor from a .npy file and draw sub-HUBOs from it")

    # Solver settings
    p.add_argument("--device",    type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device for GPT inference (default: cuda)")
    p.add_argument("--pool_type", type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"],
                   help="Operator pool (default: all_pool)")
    p.add_argument("--num_shots", type=int, default=1024,
                   help="CUDAQ sampling shots (default: 1024)")
    p.add_argument("--num_gpt_samples", type=int, default=10,
                   help="GPT candidate circuits per problem; best energy is kept (default: 10)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="GPT sampling temperature; higher = more diverse candidates (default: 0.8)")

    # Reference solver
    p.add_argument("--no_brute_force", action="store_true",
                   help="Skip brute-force reference (auto-skipped for n_nodes > 20)")
    p.add_argument("--no_sa", action="store_true",
                   help="Skip SA reference")

    # Trimmed statistics
    p.add_argument("--trim_n", type=int, default=5,
                   help="Remove worst and best N cases from trimmed summary stats (default: 5)")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    ckpt       = args.ckpt
    data_dir   = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
    config_file = (args.config_file or
                   os.path.join(data_dir, 'train_adapt_gpt_config.py'))

    for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                        (config_file, "config_file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("=" * 60)
    print("STEP 4-2: HUBO-GPT inference test (GPT circuits only, no NULL circuit)")
    print("=" * 60)
    print(f"  checkpoint  : {ckpt}")
    print(f"  data_dir    : {data_dir}")
    print(f"  n_nodes     : {args.n_nodes}")
    print(f"  device      : {args.device}")
    print(f"  trim_n      : {args.trim_n}")
    print(f"  circuits/problem : {args.num_gpt_samples}  (GPT-generated only, no extra empty circuit)")
    print("=" * 60)

    # ---- Load or generate test HUBOs --------------------------------------
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
                args.n_nodes, density=args.density,
                weight_range=args.weight_range,
                seed=int(rng.integers(0, 2**31)),
            )
            for _ in range(args.n_test)
        ]
        print(f"Generated {args.n_test} random sub-HUBOs (seed={args.seed})")

    # Add one warm-up problem at the front (discarded from stats)
    test_tensors = [test_tensors[0]] + test_tensors

    # ---- Run inference and collect stats -----------------------------------
    print(f"\nRunning inference on {len(test_tensors) - 1} sub-HUBOs "
          f"(+1 warm-up discarded)...\n")

    results = []
    for idx, T in enumerate(test_tensors):

        # GPT-based solution (GPT-generated circuits only)
        t_start = time.time()
        x_gpt = solve_sub_HUBO_GPT_gpt_only(
            T,
            model_ckpt=ckpt,
            data_dir=data_dir,
            config_file=config_file,
            device=args.device,
            num_shots=args.num_shots,
            pool_type=args.pool_type,
            num_gpt_samples=args.num_gpt_samples,
            temperature=args.temperature,
        )
        t_elapsed = time.time() - t_start
        e_gpt = evaluate_hubo_energy(T, x_gpt)

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
                approx_ratio = e_gpt / e_ref   # should be close to 1 from below
            else:
                approx_ratio = e_ref / e_gpt if e_gpt > 0 else float('nan')

        results.append({
            'idx': idx + 1,
            'e_gpt': e_gpt,
            'e_ref': e_ref,
            'e_sa':  e_sa,
            'approx_ratio': approx_ratio,
            'bitstring': ''.join(str(int(b)) for b in x_gpt),
            'time': t_elapsed,
        })

        ref_str = f"ref={e_ref:.4f}" if e_ref is not None else "ref=N/A"
        ar_str  = f"ar={approx_ratio:.4f}" if approx_ratio is not None else "ar=N/A"
        warmup_tag = " [warm-up]" if idx == 0 else ""
        print(f"  [{idx:3d}]  gpt={e_gpt:.4f}  {ref_str}  {ar_str}  "
              f"bitstring={results[-1]['bitstring']}  time={t_elapsed:.2f}s{warmup_tag}")

    # ---- Summary (exclude warm-up at index 0) ------------------------------
    results_eval = results[1:]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    e_gpt_vals = [r['e_gpt'] for r in results_eval]
    e_ref_vals = [r['e_ref'] for r in results_eval if r['e_ref'] is not None]
    ar_vals    = [r['approx_ratio'] for r in results_eval
                  if r['approx_ratio'] is not None and not np.isnan(r['approx_ratio'])]

    print(f"  GPT energy  — mean: {np.mean(e_gpt_vals):.4f}  "
          f"std: {np.std(e_gpt_vals):.4f}  "
          f"min: {np.min(e_gpt_vals):.4f}")

    if e_ref_vals:
        print(f"  Ref energy  — mean: {np.mean(e_ref_vals):.4f}  "
              f"std: {np.std(e_ref_vals):.4f}  "
              f"min: {np.min(e_ref_vals):.4f}")

    if ar_vals:
        print(f"  Approx ratio (full) — mean: {np.mean(ar_vals):.4f}  "
              f"std: {np.std(ar_vals):.4f}  "
              f"min: {np.min(ar_vals):.4f}  n={len(ar_vals)}")

        if len(ar_vals) > 2 * args.trim_n:
            ar_sorted = sorted(ar_vals)
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

    print("\n Inference test complete.")

if __name__ == "__main__":
    main()

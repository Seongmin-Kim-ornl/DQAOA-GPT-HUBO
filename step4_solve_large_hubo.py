import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    evaluate_hubo_energy,
)
from DQAOA import DQAOA

def _find_file(directory, size, suffix, ctn):
    candidates = [
        f"{size}_{suffix}_Copy{ctn}.txt",
        f"{size}_{suffix}-Copy{ctn}.txt",
        f"{size}_{suffix}.txt",
    ]
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    tried = [os.path.join(directory, n) for n in candidates]
    raise FileNotFoundError(
        f"Cannot find {suffix} file for size={size}, ctn={ctn}. Tried:\n"
        + "\n".join(f"  {p}" for p in tried)
    )

def load_hubo_from_txt(hubo_size, ctn=0, hubo_dir="."):
    q_path  = _find_file(hubo_dir, hubo_size, "QUBO",    ctn)
    kt_path = _find_file(hubo_dir, hubo_size, "k_third", ctn)
    print(f"  Q matrix  : {q_path}")
    print(f"  k_third   : {kt_path}")

    with open(q_path, "r") as f:
        Q = np.array(np.reshape(np.asmatrix(f.read()),
                                (hubo_size, hubo_size)))

    with open(kt_path, "r") as f:
        k_third = np.array(np.reshape(np.asmatrix(f.read()),
                                      (hubo_size, 8)))

    T = np.zeros((hubo_size, hubo_size, hubo_size))
    for i in range(hubo_size):
        T[i, i, i] = Q[i, i]
    for i in range(hubo_size):
        for j in range(i + 1, hubo_size):
            T[i, i, j] = Q[i, j]
    for i in range(hubo_size):
        for j in range(i + 1, hubo_size):
            for k in range(j + 1, hubo_size):
                T[i, j, k] = float(np.sum(k_third[i, :] * k_third[j, :] * k_third[k, :]))
    return T

def parse_args():
    p = argparse.ArgumentParser(
        description="Solve a large HUBO with DQAOA (GPT or standard QAOA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--solver", type=str, default="gpt",
                   choices=["gpt", "qaoa"],
                   help="Sub-HUBO solver: 'gpt' (QAOA-GPT) or 'qaoa' (standard QAOA)")

    p.add_argument("--ckpt", type=str, default=None,
                   help="[gpt] Path to trained HUBO-GPT checkpoint (.pt file)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="[gpt] Directory with meta.pkl. Defaults to dirname(--ckpt).")
    p.add_argument("--config_file", type=str, default=None,
                   help="[gpt] Config .py. Defaults to <data_dir>/train_adapt_gpt_config.py")
    p.add_argument("--device",    type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--pool_type", type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])

    p.add_argument("--qaoa_num_layers", type=int,   default=2,
                   help="[qaoa] QAOA circuit depth p")
    p.add_argument("--qaoa_max_iters",  type=int,   default=200,
                   help="[qaoa] Max COBYLA iterations per sub-HUBO")
    p.add_argument("--qaoa_target",     type=str,   default="nvidia",
                   choices=["nvidia", "qpp-cpu"],
                   help="[qaoa] CUDA-Q backend")
    p.add_argument("--qaoa_reverse_bitorder", action="store_true",
                   help="[qaoa] Reverse CUDA-Q bitstring ordering")

    hubo_src = p.add_mutually_exclusive_group()
    hubo_src.add_argument("--hubo_npy", type=str, default=None,
                          help="Path to a pre-saved full HUBO tensor (.npy)")
    hubo_src.add_argument("--load_formulated", action="store_true",
                          help="Load via subHUBO_gen.formulate_HUBO() "
                               "(uses {size}_QUBO.txt in cwd)")
    hubo_src.add_argument("--load_txt", action="store_true",
                          help="Load from {size}_QUBO_Copy{ctn}.txt and "
                               "{size}_k_third_Copy{ctn}.txt (see --hubo_dir, --ctn)")

    p.add_argument("--hubo_size",    type=int,   default=100,
                   help="Full HUBO size N")
    p.add_argument("--ctn",          type=int,   default=0,
                   help="[load_txt] Copy number in filename, e.g. 0 → *_Copy0.txt")
    p.add_argument("--hubo_dir",     type=str,   default=".",
                   help="[load_txt] Directory containing the QUBO/k_third text files")
    p.add_argument("--density",      type=float, default=0.5)
    p.add_argument("--weight_range", type=float, default=1.0)
    p.add_argument("--seed_hubo",    type=int,   default=0)

    p.add_argument("--sub_n",      type=int, default=4,
                   help="Sub-HUBO size (must match n_nodes from training for --solver gpt)")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--num_shots",  type=int, default=1024)
    p.add_argument("--seed_dq",    type=int, default=12345)

    p.add_argument("--save_solution", type=str, default=None)
    p.add_argument("--verbose",       action="store_true")

    return p.parse_args()

def main():
    args = parse_args()

    ckpt = data_dir = config_file = None

    if args.solver == "gpt":
        if args.ckpt is None:
            print("ERROR: --ckpt is required for --solver gpt")
            sys.exit(1)
        ckpt        = args.ckpt
        data_dir    = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
        config_file = args.config_file or os.path.join(data_dir, 'train_adapt_gpt_config.py')
        for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                            (config_file, "config_file")]:
            if not os.path.exists(path):
                print(f"ERROR: {label} not found: {path}")
                sys.exit(1)

    if args.hubo_npy:
        print(f"Loading HUBO tensor from: {args.hubo_npy}")
        T_full = np.load(args.hubo_npy)
        hubo_size = T_full.shape[0]
    elif args.load_formulated:
        print(f"Loading formulated HUBO (size={args.hubo_size})...")
        from subHUBO_gen import formulate_HUBO
        T_full = formulate_HUBO(args.hubo_size)
        hubo_size = args.hubo_size
    elif args.load_txt:
        print(f"Loading HUBO from text files (size={args.hubo_size}, ctn={args.ctn})...")
        T_full = load_hubo_from_txt(args.hubo_size, ctn=args.ctn, hubo_dir=args.hubo_dir)
        hubo_size = args.hubo_size
    else:
        print(f"Generating random HUBO (size={args.hubo_size}, seed={args.seed_hubo})...")
        T_full = generate_random_hubo(
            args.hubo_size,
            density=args.density,
            weight_range=args.weight_range,
            seed=args.seed_hubo,
        )
        hubo_size = args.hubo_size

    print("=" * 60)
    print("STEP 4: Solving large HUBO with DQAOA")
    print("=" * 60)
    print(f"  solver      : {args.solver}")
    print(f"  HUBO size   : {hubo_size}")
    print(f"  sub_n       : {args.sub_n}")
    print(f"  iterations  : {args.iterations}")
    if args.solver == "gpt":
        print(f"  checkpoint  : {ckpt}")
        print(f"  device      : {args.device}")
    else:
        print(f"  num_layers  : {args.qaoa_num_layers}")
        print(f"  max_iters   : {args.qaoa_max_iters}  (COBYLA per sub-HUBO)")
        print(f"  target      : {args.qaoa_target}")
    print("=" * 60)


    print(f"\nStarting DQAOA loop...")
    solver = DQAOA(
        T=T_full,
        sub_n=args.sub_n,
        iterations=args.iterations,
        model_ckpt=ckpt,
        data_dir=data_dir,
        config_file=config_file,
        device=args.device,
        pool_type=args.pool_type,
        num_shots=args.num_shots,
        seed=args.seed_dq,
        verbose=args.verbose,
        qaoa_num_layers=args.qaoa_num_layers,
        qaoa_max_iters=args.qaoa_max_iters,
        qaoa_target=args.qaoa_target,
        qaoa_reverse_bitorder=args.qaoa_reverse_bitorder,
        solver=args.solver,
    )

    t0 = time.time()
    x_opt, best_energy = solver.run()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Best HUBO energy  : {best_energy:.6f}")
    print(f"  Total time        : {elapsed:.6f}s")
    print(f"  Solution (first 20 vars): {x_opt[:20]}")

    if args.save_solution:
        np.save(args.save_solution, x_opt)
        print(f"\nSolution saved to: {args.save_solution}")

if __name__ == "__main__":
    main()

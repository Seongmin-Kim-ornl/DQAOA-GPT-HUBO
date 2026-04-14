import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.generate_adapt_hubo_data_cpu import generate_data_hubo

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate HUBO ADAPT-QAOA training data (with --rounding_digits)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--n_nodes",     type=int,   default=4)
    p.add_argument("--hubo_number", type=int,   default=500)

    p.add_argument("--no_linear",    action="store_true")
    p.add_argument("--no_quadratic", action="store_true")
    p.add_argument("--no_cubic",     action="store_true")
    p.add_argument("--density",      type=float, default=0.5)
    p.add_argument("--weight_range", type=float, default=1.0,
                   help="Coefficients sampled from U(-w, w)")

    p.add_argument("--rounding_digits", type=int, default=1,
                   help="Decimal places for coefficient rounding. "
                        "1 → tokens like -5.0…+5.0 (vocab ~2100 even with weight_range=5). "
                        "2 → tokens like -5.00…+5.00 (vocab ~4049, harder to train). "
                        "Recommend: 1 when weight_range > 1.")

    p.add_argument("--no_brute_force", action="store_true")
    p.add_argument("--no_sa",          action="store_true")

    p.add_argument("--op_pool",      type=str,   default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])
    p.add_argument("--optimizer",    type=str,   default="BFGS",
                   choices=["BFGS", "L-BFGS-B", "COBYLA"])
    p.add_argument("--approx_ratio", type=float, default=0.8)
    p.add_argument("--max_iter",     type=int,   default=15)
    p.add_argument("--trials",       type=int,   default=1)
    p.add_argument("--init_gamma",   type=float, nargs="+", default=[0.01])
    p.add_argument("--multi_gamma",  action="store_true")

    p.add_argument("--from_formulated", action="store_true")
    p.add_argument("--hubo_size",       type=int, default=None)

    p.add_argument("--npy_dir", type=str, default=None)

    p.add_argument("--output_dir",       type=str, default="adapt_results_hubo")
    p.add_argument("--checkpoint_every", type=int, default=None,
                   help="Save partial results every N HUBOs (overwrites same file). "
                        "Useful for long runs to avoid losing all progress on crash.")
    p.add_argument("--seed_hubo",  type=int, default=42)
    p.add_argument("--seed_adapt", type=int, default=None)
    p.add_argument("--quiet",      action="store_true")

    return p.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("STEP 1: Generating HUBO ADAPT-QAOA training data")
    print("=" * 60)
    print(f"  n_nodes        : {args.n_nodes}")
    print(f"  hubo_number    : {args.hubo_number}")
    print(f"  weight_range   : {args.weight_range}")
    print(f"  density        : {args.density}")
    print(f"  rounding_digits: {args.rounding_digits}  "
          f"← controls vocab size (1 = smaller vocab, recommended)")
    print(f"  op_pool        : {args.op_pool}")
    print(f"  optimizer      : {args.optimizer}")
    print(f"  approx_ratio   : {args.approx_ratio}")
    print(f"  max_iter       : {args.max_iter}")
    print(f"  output_dir     : {args.output_dir}")
    if args.from_formulated:
        print(f"  mode           : from_formulated (hubo_size={args.hubo_size})")
    elif args.npy_dir:
        print(f"  npy_dir        : {args.npy_dir}")
    else:
        print(f"  mode           : random HUBOs")
    print("=" * 60)

    generate_data_hubo(
        output_dir=args.output_dir,
        hubo_tensors=None,
        hubo_input_npy_dir=args.npy_dir,
        from_formulated=args.from_formulated,
        hubo_size=args.hubo_size,
        n_nodes=args.n_nodes,
        hubo_number=args.hubo_number,
        include_linear=not args.no_linear,
        include_quadratic=not args.no_quadratic,
        include_cubic=not args.no_cubic,
        density=args.density,
        weight_range=args.weight_range,
        rounding_digits=args.rounding_digits,
        use_brute_force=(not args.no_brute_force),
        use_simulated_annealing=(not args.no_sa),
        op_pool=args.op_pool,
        init_gamma=args.init_gamma,
        trials_per_hubo=args.trials,
        optimizer=args.optimizer,
        approx_ratio=args.approx_ratio,
        max_iter=args.max_iter,
        multi_gamma=args.multi_gamma,
        seed_hubo=args.seed_hubo,
        seed_adapt=args.seed_adapt,
        verbose=not args.quiet,
        checkpoint_every=args.checkpoint_every,
    )

    print(f"Results saved to: {args.output_dir}/res/  and  {args.output_dir}/hubo/")

if __name__ == "__main__":
    main()

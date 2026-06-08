import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaoa_gpt_src.prepare_tokens_hubo import generate_tokens_hubo

def parse_args():
    p = argparse.ArgumentParser(description="Tokenize HUBO data for nanoGPT")

    p.add_argument("--results_dir", type=str, default="adapt_results_hubo",
                   help="Directory produced by step1 (contains res/ and hubo/)")
    p.add_argument("--save_dir", type=str, default="nanoGPT/data/hubo_save_dir",
                   help="Output directory for tokenized data (default: nanoGPT/data/hubo_save_dir)")
    p.add_argument("--config_template_dir", type=str, default="qaoa_gpt_src",
                   help="Directory containing train_adapt_gpt_config_template.py "
                        "(default: qaoa_gpt_src)")

    p.add_argument("--n_nodes", type=int, default=4,
                   help="Number of binary variables (must match step1, default: 4)")

    p.add_argument("--rounding_digits",  type=int,   default=2,
                   help="Decimal places for rounding coefficients (default: 2)")
    p.add_argument("--min_block_size",   type=int,   default=128,
                   help="Minimum sliding-window block size (default: 128)")
    p.add_argument("--max_block_size",   type=int,   default=256,
                   help="Maximum sliding-window block size = nanoGPT block_size (default: 256)")
    p.add_argument("--max_abs_param",    type=float, default=10.0,
                   help="Clip gamma/beta beyond this range (default: 10)")
    p.add_argument("--no_coef_mod",      action="store_true",
                   help="Disable beta modulation to [0, pi)")
    p.add_argument("--no_sliding_window",action="store_true",
                   help="Disable sliding window (use full sequences only)")
    p.add_argument("--skip_only_qaoa",   action="store_true",
                   help="Skip circuits that use only the standard QAOA mixer")

    p.add_argument("--val_frac",  type=float, default=0.1,
                   help="Fraction of graphs for validation (default: 0.1)")
    p.add_argument("--test_frac", type=float, default=0.1,
                   help="Fraction of graphs for test (default: 0.1)")
    p.add_argument("--approx_ratio_thr", type=float, default=0.97,
                   help="Min approximation ratio to include a sample (default: 0.97)")

    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel workers for CSV loading and FEATHER (default: 1)")
    p.add_argument("--debug_limit", type=int, default=0,
                   help="Limit CSV files loaded per directory (0 = no limit)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose output")

    p.add_argument("--extra_results_dirs", type=str, default="",
                   help="Additional result directories separated by semicolons")

    return p.parse_args()

def main():
    args = parse_args()

    results_path = args.results_dir
    if args.extra_results_dirs:
        results_path += ";" + args.extra_results_dirs

    print("=" * 60)
    print("STEP 2: Tokenizing HUBO data for nanoGPT")
    print("=" * 60)
    print(f"  results_dir   : {results_path}")
    print(f"  save_dir      : {args.save_dir}")
    print(f"  n_nodes       : {args.n_nodes}")
    print(f"  max_block_size: {args.max_block_size}")
    print(f"  val_frac      : {args.val_frac}")
    print(f"  test_frac     : {args.test_frac}")
    print(f"  approx_thr    : {args.approx_ratio_thr}")
    print("=" * 60)

    generate_tokens_hubo(
        results_fpath_str=results_path,
        save_path_str=args.save_dir,
        config_path_temp=args.config_template_dir,
        n_nodes=args.n_nodes,
        rounding_digits=args.rounding_digits,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_abs_param_val=args.max_abs_param,
        perform_coef_mod_range=not args.no_coef_mod,
        apply_sliding_window=not args.no_sliding_window,
        n_workers=args.n_workers,
        skip_only_qaoa_circ=args.skip_only_qaoa,
        approx_ratio_thr=args.approx_ratio_thr,
        debug_limit=args.debug_limit,
        verbose=not args.quiet,
    )

    print(f"Tokenized data saved to: {args.save_dir}/")

if __name__ == "__main__":
    main()

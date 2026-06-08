import argparse
import os
import sys
import subprocess
import shutil

def parse_args():
    p = argparse.ArgumentParser(
        description="Train nanoGPT on HUBO tokenized data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_dir", type=str, default="nanoGPT/data/hubo_save_dir",
                   help="Tokenized data directory (output of step2); may be an absolute path")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to save checkpoints; defaults to nanoGPT/out-<data_stem>. "
                        "Use an absolute path (e.g. /lustre/.../out-hubo_4nodes) to redirect "
                        "to scratch storage.")
    p.add_argument("--nanogpt_dir", type=str, default="nanoGPT",
                   help="Path to the nanoGPT directory")
    p.add_argument("--train_script", type=str, default="train_pad_gemb.py",
                   help="Training script inside nanogpt_dir")

    p.add_argument("--max_iters",   type=int,   default=None, help="Training iterations")
    p.add_argument("--batch_size",  type=int,   default=None, help="Batch size")
    p.add_argument("--n_layer",     type=int,   default=None, help="Transformer layers")
    p.add_argument("--n_head",      type=int,   default=None, help="Attention heads")
    p.add_argument("--n_embd",      type=int,   default=None, help="Embedding dimension")
    p.add_argument("--dropout",     type=float, default=None, help="Dropout rate")
    p.add_argument("--learning_rate",   type=float, default=None, help="Learning rate")
    p.add_argument("--lr_decay_iters", type=int,   default=None, help="Iters over which LR decays to min_lr (default: same as max_iters)")
    p.add_argument("--device",      type=str,   default=None,
                   choices=["cuda", "cpu", "mps"],
                   help="Device (default: uses config value)")
    p.add_argument("--compile",     type=str,   default=None,
                   choices=["true", "false"],
                   help="torch.compile the model (true/false)")
    p.add_argument("--eval_interval", type=int, default=None,
                   help="Steps between validation loss evaluations")
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="Rename best checkpoint to this name after training "
                        "(e.g. ckpt_4nodes.pt). Also copies meta.pkl and "
                        "train_adapt_gpt_config.py next to it so the output "
                        "dir is self-contained for inference.")

    return p.parse_args()

def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    nanogpt_dir  = os.path.join(project_root, args.nanogpt_dir)
    train_script = os.path.join(nanogpt_dir, args.train_script)

    # Resolve data_dir to absolute so it works regardless of cwd inside nanogpt/
    abs_data_dir = os.path.abspath(args.data_dir)
    config_file  = os.path.join(abs_data_dir, 'train_adapt_gpt_config.py')

    # Resolve out_dir: explicit --out_dir wins; otherwise default to nanoGPT/out-<stem>
    data_stem = os.path.basename(abs_data_dir.rstrip("/"))
    if args.out_dir:
        abs_out_dir = os.path.abspath(args.out_dir)
    else:
        abs_out_dir = os.path.join(nanogpt_dir, f"out-{data_stem}")

    if not os.path.isfile(train_script):
        print(f"ERROR: training script not found: {train_script}")
        sys.exit(1)

    if not os.path.isfile(config_file):
        print(f"ERROR: config file not found: {config_file}")
        print("       Did you run step2_tokenize.py first?")
        sys.exit(1)

    cmd = [sys.executable, train_script, config_file]

    overrides = {
        "data_dir":      f"'{abs_data_dir}'",
        "out_dir":       f"'{abs_out_dir}'",
        "max_iters":     args.max_iters,
        "batch_size":    args.batch_size,
        "n_layer":       args.n_layer,
        "n_head":        args.n_head,
        "n_embd":        args.n_embd,
        "dropout":       args.dropout,
        "learning_rate":   args.learning_rate,
        "lr_decay_iters":  args.lr_decay_iters,
        "device":        f"'{args.device}'" if args.device else None,
        "compile":       args.compile,
        "eval_interval": args.eval_interval,
    }
    for key, val in overrides.items():
        if val is not None:
            cmd.append(f"--{key}={val}")

    print("=" * 60)
    print("STEP 3: Training nanoGPT on HUBO data")
    print("=" * 60)
    print(f"  Script     : {train_script}")
    print(f"  Config     : {config_file}")
    print(f"  Data dir   : {abs_data_dir}")
    print(f"  Out dir    : {abs_out_dir}")
    if any(v is not None for v in overrides.values()):
        print("  Overrides  :")
        for k, v in overrides.items():
            if v is not None:
                print(f"    {k} = {v}")
    print("=" * 60)
    print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=nanogpt_dir)

    if result.returncode == 0:
        print("\nStep 3 complete.")

        for fname in ['meta.pkl', 'train_adapt_gpt_config.py']:
            src = os.path.join(abs_data_dir, fname)
            dst = os.path.join(abs_out_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Copied {fname} → {abs_out_dir}/")

        if args.ckpt_name:
            ckpt_name = args.ckpt_name
            if not ckpt_name.endswith('.pt'):
                ckpt_name += '.pt'
            best_ckpt  = os.path.join(abs_out_dir, 'ckpt.pt')
            named_ckpt = os.path.join(abs_out_dir, ckpt_name)
            if os.path.exists(best_ckpt):
                shutil.copy2(best_ckpt, named_ckpt)
                print(f"  Best checkpoint → {named_ckpt}")
            else:
                print(f"  WARNING: ckpt.pt not found in {abs_out_dir}; "
                      f"skipping rename to {ckpt_name}")

        print(f"Checkpoint saved to: {abs_out_dir}/ckpt_*.pt")
    else:
        print(f"\nTraining exited with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()

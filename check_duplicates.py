"""
Check for duplicate HUBOs in the adapt_results_vqe_* data directory.

Two rows are duplicates if they encode the same sub-HUBO (identical
linear + quadratic + cubic interaction weights). This happens when two
data-generation runs used the same --seed_hubo value.

Usage:
    python check_duplicates.py --data_dir adapt_results_vqe_10nodes
    python check_duplicates.py --data_dir /proj/stf246/scratch/gvk/.../adapt_results_vqe_10nodes --remove
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def hubo_hash(row):
    """Hash a HUBO row by its interaction content (weights + indices)."""
    key = row['linear_json'] + '|' + row['quad_json'] + '|' + row['cubic_json']
    return hashlib.md5(key.encode()).hexdigest()


def load_hubo_csvs(hubo_dir, quiet=False):
    files = sorted(Path(hubo_dir).glob('*.csv'))
    if not files:
        print(f"No CSV files found in {hubo_dir}")
        sys.exit(1)
    dfs = []
    for f in tqdm(files, desc='Loading hubo CSVs', disable=quiet):
        try:
            df = pd.read_csv(f)
            df['_source_file'] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_dir', type=str, required=True,
                   help='adapt_results_vqe_* directory (contains hubo/ and res/ subdirs)')
    p.add_argument('--remove', action='store_true',
                   help='Remove duplicate CSV files that are fully redundant '
                        '(all HUBOs in the file are duplicates of another file). '
                        'Safe: keeps the first-seen file, removes later duplicates.')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    hubo_dir = Path(args.data_dir) / 'hubo'
    if not hubo_dir.exists():
        print(f"ERROR: {hubo_dir} not found")
        sys.exit(1)

    print(f"Scanning: {hubo_dir}")
    df = load_hubo_csvs(hubo_dir, quiet=args.quiet)
    print(f"Loaded {len(df):,} rows from {df['_source_file'].nunique()} files\n")

    print("Computing HUBO hashes...")
    df['_hash'] = df.apply(hubo_hash, axis=1)

    # --- duplicate statistics ---
    total        = len(df)
    n_unique     = df['_hash'].nunique()
    n_duplicates = total - n_unique
    dup_frac     = n_duplicates / total * 100 if total > 0 else 0

    print(f"\n{'='*55}")
    print(f"  Total HUBO rows   : {total:,}")
    print(f"  Unique HUBOs      : {n_unique:,}")
    print(f"  Duplicate rows    : {n_duplicates:,}  ({dup_frac:.1f}%)")
    print(f"{'='*55}\n")

    if n_duplicates == 0:
        print("No duplicates found.")
        return

    # Show which files share the most duplicates
    dup_hashes = df[df.duplicated('_hash', keep='first')]['_hash']
    dup_df     = df[df['_hash'].isin(dup_hashes)]

    print("Files with highest duplicate counts:")
    file_dup_counts = (
        dup_df.groupby('_source_file')
              .size()
              .sort_values(ascending=False)
              .head(20)
    )
    for fname, count in file_dup_counts.items():
        print(f"  {fname:50s}  {count:5d} dup rows")

    # Identify files where ALL rows are duplicates of rows in earlier files
    # (these are safe to remove entirely)
    if args.remove:
        print("\nIdentifying fully-redundant files (all rows duplicated elsewhere)...")

        # Assign a canonical file to each hash (first file that introduced it)
        df_sorted     = df.sort_values('_source_file')   # alphabetical = stable order
        first_seen    = df_sorted.drop_duplicates('_hash', keep='first')
        hash_owner    = first_seen.set_index('_hash')['_source_file'].to_dict()

        df['_owner']  = df['_hash'].map(hash_owner)
        df['_is_dup'] = df['_source_file'] != df['_owner']

        # A file is fully redundant if every one of its rows is a duplicate
        file_stats = df.groupby('_source_file')['_is_dup'].agg(['sum', 'count'])
        file_stats.columns = ['dup_rows', 'total_rows']
        redundant_files = file_stats[file_stats['dup_rows'] == file_stats['total_rows']]

        if redundant_files.empty:
            print("  No fully-redundant files found "
                  "(all files contain at least some unique HUBOs).")
            print("  The duplicates are spread across files — "
                  "deduplication happens automatically during tokenization.")
        else:
            print(f"  Found {len(redundant_files)} fully-redundant file(s):")
            for fname in redundant_files.index:
                fpath = hubo_dir / fname
                print(f"    Removing: {fname}")
                fpath.unlink()
            print(f"\n  Removed {len(redundant_files)} files.")
            print(f"  NOTE: also remove the matching res/ files manually if needed.")
    else:
        print("\nTo remove fully-redundant files, re-run with --remove")
        print("NOTE: If duplicates are spread across files (not concentrated in")
        print("      one file), just re-tokenize — the tokenizer will deduplicate")
        print("      via graph_id when building the training set.")


if __name__ == '__main__':
    main()

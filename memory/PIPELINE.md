# DQAOA-GPT-HUBO Pipeline

Extends the original DQAOA-GPT-QUBO codebase to solve **HUBO** (Higher-Order
Unconstrained Binary Optimization) problems with up to 3rd-order (cubic)
interactions.  The full pipeline mirrors the QUBO version but adds ZZZ
Hamiltonian terms, extended token vocabulary, and MPI-parallel solving.

---

## Environment

```bash
source /autofs/nccsopen-svm1_home/gvk/defiant/CudaQ/bin/activate   # conda/venv
# or wherever your CudaQ environment lives
```

Working directory: `QAOA-GPT_HUBO2/`

---

## Pipeline overview

```
Step 1  generate_data    →  ADAPT-QAOA circuits for sub-HUBOs  →  adapt_results_hubo/
Step 2  tokenize         →  FEATHER embeddings + token arrays   →  nanoGPT/data/hubo_save_dir/
Step 3  train            →  train nanoGPT                       →  nanoGPT/out-hubo_save_dir/
Step 4  inference_test   →  test model on unseen sub-HUBOs
Step 5  solve (single)   →  dqHUBO sequential solver
Step 6  solve (parallel) →  dqHUBO parallel_subs (single node, no MPI)
Step 7  solve (MPI)      →  DQAOA_HUBO manager-worker, 1 GPU per rank
```

---

## Step 1 — Generate ADAPT-QAOA training data

```bash
# Mode A: from your real HUBO (recommended)
python step1_generate_data.py \
    --from_formulated --hubo_size 100 \
    --n_nodes 4 --hubo_number 500

# Mode B: from pre-saved .npy sub-HUBOs
python step1_generate_data.py --npy_dir subhubo_npy/ --n_nodes 4

# Mode C: random HUBOs (quick test)
python step1_generate_data.py --n_nodes 4 --hubo_number 200
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--from_formulated` | off | loads `{hubo_size}_QUBO.txt` + `{hubo_size}_k_third.txt` via `subHUBO_gen.py` |
| `--hubo_size` | — | required with `--from_formulated` |
| `--n_nodes` | 4 | sub-HUBO size; must match training |
| `--hubo_number` | 500 | number of sub-HUBOs to generate |
| `--approx_ratio` | 0.97 | ADAPT early-stop threshold |
| `--max_iter` | 10 | max ADAPT iterations per sub-HUBO |
| `--output_dir` | `adapt_results_hubo` | saves to `res/` and `hubo/` subdirs |

Output: `adapt_results_hubo/res/*.csv` and `adapt_results_hubo/hubo/*.csv`

**Diagnose approx_ratio filtering:**
```bash
python -c "
import pandas as pd, glob
for f in glob.glob('adapt_results_hubo/res/*.csv'):
    df = pd.read_csv(f)
    print(f, len(df), 'rows, mean ar:', df['approx_ratio'].mean().round(3),
          'pct>=0.97:', round((df['approx_ratio']>=0.97).mean()*100,1), '%')
"
```

---

## Step 2 — Tokenize

```bash
python step2_tokenize.py \
    --results_dir adapt_results_hubo \
    --save_dir nanoGPT/data/hubo_save_dir \
    --n_nodes 4

# If approx_ratio filter drops too many rows, lower the threshold:
python step2_tokenize.py --approx_ratio_thr 0.5 ...
```

Output: `nanoGPT/data/hubo_save_dir/{train,val,test}.npy`, `meta.pkl`,
`feather_emb_d500.npy`, `train_adapt_gpt_config.py`

**Common error — merge produces 0 rows:**
Caused by prefix mismatch between `res/` and `hubo/` CSV filenames.
Fixed in `prepare_tokens_hubo.py`: strips `_results` (8 chars) and `_hubo`
(5 chars) to get matching `pid_timestamp` prefix.

---

## Step 3 — Train nanoGPT

```bash
python step3_train.py --max_iters 20000

# Override hyperparameters
python step3_train.py --max_iters 20000 --batch_size 32 --n_layer 6
```

Checkpoints saved to `nanoGPT/out-hubo_save_dir/`:

| File | When saved |
|---|---|
| `ckpt_{iter}.pt` | when val loss improves (`always_save_checkpoint=False`) |
| `ckpt_overfit.pt` | same as last best checkpoint |
| `ckpt_{final_iter}.pt` | always at end of training (added fix) |

**Common issues:**

- `0it` in tqdm → `train_data.shape[0] < batch_size`.
  Fix: lower `--approx_ratio_thr` in step 2, or generate more data in step 1.
- `ValueError: high <= 0` in `get_batch` → `val_data` is empty.
  Fixed in `train_pad_gemb.py`: falls back to train data when val is empty.
- No `ckpt_20000.pt` → `always_save_checkpoint=False` only saves on improvement.
  Fixed: final checkpoint always saved after training loop.

**Which checkpoint to use for inference:**
Use the one with the lowest iter number that exists (e.g. `ckpt_250.pt`) —
this is the best val-loss checkpoint.  The final `ckpt_20000.pt` may be
more overfit on a small dataset.

---

## Step 4 — Inference test

```bash
python step4_inference_test.py \
    --ckpt nanoGPT/out-hubo_save_dir/ckpt_250.pt \
    --data_dir nanoGPT/data/hubo_save_dir \
    --n_nodes 4 --n_test 10
```

Reports GPT energy vs brute-force / SA reference and approximation ratio.

---

## Step 5 — Solve large HUBO (sequential)

```bash
python step5_solve_large_hubo.py \
    --ckpt nanoGPT/out-hubo_save_dir/ckpt_250.pt \
    --data_dir nanoGPT/data/hubo_save_dir \
    --hubo_size 100 --sub_n 4 --iterations 200
```

Uses `dqHUBO` class: sequential decompose → solve → aggregate loop.

---

## Step 6 — Solve large HUBO (parallel, no MPI)

```bash
# parallel_subs: n_workers sub-HUBOs per cycle, single process, model cached
python step6_solve_large_hubo.py \
    --ckpt nanoGPT/out-hubo_save_dir/ckpt_250.pt \
    --data_dir nanoGPT/data/hubo_save_dir \
    --hubo_size 100 --sub_n 4 --iterations 200 \
    --n_workers 8 --verbose
```

Two modes (`--mode`):

| Mode | Description | Best for |
|---|---|---|
| `parallel_subs` (default) | sequential loop, `n_workers` sub-HUBOs/cycle, 1 process | single GPU |
| `parallel_restarts` | `n_workers` independent chains via `multiprocessing` | multi-GPU or CPU |

> Note: `parallel_subs` is NOT truly parallel — it's a loop.
> True parallelism → use Step 7 (MPI).

---

## Step 7 — Solve large HUBO (MPI, multi-GPU)

```bash
# 1 manager (rank 0) + 8 workers (ranks 1-8, one per GPU)
mpirun -np 9 python step7_mpi_solve_hubo.py \
    --ckpt nanoGPT/out-hubo_save_dir/ckpt_250.pt \
    --data_dir nanoGPT/data/hubo_save_dir \
    --hubo_size 100 --sub_n 4 --iterations 200 \
    --report_every 10

# With SLURM
srun --ntasks=9 --gpus-per-task=1 python step7_mpi_solve_hubo.py ...
```

MPI pattern (mirrors `DQAOA.py`):
- **Rank 0** (manager): decomposes sub-HUBOs, sends to workers, aggregates results
- **Rank 1..N-1** (workers): each on `cuda:{rank-1}`, solves sub-HUBO, sends back

Each cycle: `N-1` sub-HUBOs solved truly in parallel.

**First cycle is slow** (~30-90s) — each worker loads the GPT model cold.
From cycle 2 onward the model is cached and much faster.

Progress output (always on, no flag needed):
```
  [rank 1] starting on cuda:0
  [rank 0] cycle 1/200: sending sub-HUBOs to 8 workers...
  [rank 1 cuda:0] cycle 1: solving...
  [rank 1 cuda:0] cycle 1: done (45.2s)  bitstring=0110
  [rank 0] cycle 1/200  energy=-13.12  elapsed=47.3s
```

---

## Key source files

### New files created for HUBO

| File | Purpose |
|---|---|
| `qaoa_gpt_src/hubo_hamiltonian.py` | HUBO→Ising conversion, ZZZ spin Hamiltonian, brute-force/SA solvers, random HUBO generator |
| `qaoa_gpt_src/generate_adapt_hubo_data.py` | ADAPT-QAOA data generation for HUBO (3 input modes) |
| `qaoa_gpt_src/prepare_tokens_hubo.py` | Tokenization: 1/2/3-tuple vocab, FEATHER on projected graph |
| `qaoa_gpt_src/util_hubo.py` | Inference utilities: tokenize, FEATHER embed, circuit execution |
| `qaoa_gpt_src/model_interface_hubo.py` | `HUBO_GPT` class, wraps GPT inference for HUBO |
| `utils/solve_HUBO_GPT.py` | `solve_sub_HUBO_GPT()` — main solver entry point with model cache |
| `dqHUBO.py` | Sequential dqHUBO solver class |

### Modified files

| File | What changed |
|---|---|
| `nanoGPT/train_pad_gemb.py` | Loop uses `max_iters` (not epochs×batches); handles empty val; saves final ckpt |
| `step1_generate_data.py` | Added `--from_formulated` / `--hubo_size` flags |

### Original QUBO files (unchanged, for reference)

| File | Purpose |
|---|---|
| `DQAOA.py` | MPI manager-worker DQAOA for QUBO — template for step7 |
| `DQAOA_run.py` | CLI runner for `DQAOA.py` |
| `dqQAOA.py` | Sequential dqQAOA for QUBO |
| `subHUBO_gen.py` | `formulate_HUBO(size)` and `make_sub_HUBO(T, sub_n)` |
| `qaoa_gpt_src/adapt_qaoa.py` | ADAPT-QAOA engine (cudaq); unchanged — works for HUBO too |

---

## Key design notes

### HUBO → Ising conversion
`x_i = (1 - z_i) / 2`

Cubic term `T[i,j,k] · x_i x_j x_k` expands to:
- Linear: `-c/8 · (z_i + z_j + z_k)`
- Quadratic: `+c/8 · (z_i z_j + z_i z_k + z_j z_k)`
- Cubic: `-c/8 · z_i z_j z_k`

`true_energy` passed to ADAPT must be the **Ising energy** at the classical
optimum (what `cudaq.observe` returns), not the HUBO energy.

### Projected graph for FEATHER
Cubic term `(i,j,k,w)` → triangle edges `(i,j)`, `(i,k)`, `(j,k)` each with
weight `|w|/3`.  This is the graph used for FEATHER embedding.

### Token sequence format
```
[bos]
[(i,) weight] ...         ← linear terms
[(i,j) weight] ...        ← quadratic terms
[(i,j,k) weight] ...      ← cubic terms
[end_of_graph]
[new_layer_p op_idx beta gamma] × n_layers
[eos]
```

### Vocabulary size
`vocab_size = 2049` for 4-node HUBO with `all_pool`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `0it` in training tqdm | `train_data.shape[0] < batch_size` | Lower `--approx_ratio_thr` in step2 or generate more data |
| `ValueError: high <= 0` | `val_data` is empty | Fixed in `train_pad_gemb.py` |
| No `ckpt_20000.pt` | `always_save_checkpoint=False` | Fixed: final ckpt always saved; use `ckpt_250.pt` (best val) |
| Step 2 merge → 0 rows | prefix mismatch in CSV filenames | Fixed: strips `_results` (8) and `_hubo` (5) chars |
| MPI step7 appears stuck | Workers loading model cold on cycle 1 | Normal — wait ~60s; progress prints added |
| `dataset_n_batches=1` | Training set < 64 sequences | More data or lower approx_ratio_thr |

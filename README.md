# DQAOA-GPT-HUBO

A framework for solving large Higher-Order Unconstrained Binary Optimization (HUBO) problems using a decomposition-based QAOA approach (**DQAOA**) with a GPT-based circuit predictor.

Instead of running QAOA directly on a large HUBO, DQAOA repeatedly decomposes the full problem into small sub-HUBOs of size `sub_n`, solves each with a sub-solver (either a trained GPT model or standard QAOA), and aggregates the results back into a global solution.

The GPT sub-solver is trained to predict near-optimal ADAPT-QAOA circuits for small HUBOs, avoiding the expensive classical optimization loop at inference time.

---

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** `cuda-quantum` installation depends on your CUDA version — see the [NVIDIA CUDA-Q docs](https://nvidia.github.io/cuda-quantum/latest) for the correct wheel.

---

## Pipeline Overview

```
Step 1: Generate HUBO training data (ADAPT-QAOA)
  ↓
Step 2: Tokenize data for nanoGPT
  ↓
Step 3: Train nanoGPT
  ↓
Step 4: Test inference on unseen sub-HUBOs
  ↓
Step 5: Solve a large HUBO with DQAOA
```

---

## Step 1 — Generate Training Data

Generates small random (or formulated) HUBOs and solves them with ADAPT-QAOA to produce training samples.

```bash
python step1_generate_data.py \
    --n_nodes 4 \
    --hubo_number 500 \
    --rounding_digits 1 \
    --output_dir adapt_results_hubo
```

To generate data from a formulated large HUBO (e.g. size 100):

```bash
python step1_generate_data.py \
    --from_formulated --hubo_size 100 \
    --n_nodes 4 --hubo_number 30 \
    --rounding_digits 1 \
    --output_dir adapt_results_hubo
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--n_nodes` | 4 | Number of binary variables per sub-HUBO |
| `--hubo_number` | 500 | Number of HUBOs to generate |
| `--rounding_digits` | 1 | Coefficient precision (1 = smaller vocab, recommended) |
| `--density` | 0.5 | Coefficient density (random HUBOs only, not used with `--from_formulated`) |
| `--weight_range` | 1.0 | Coefficients sampled from U(-w, w) (random HUBOs only, not used with `--from_formulated`) |
| `--op_pool` | all_pool | ADAPT operator pool |
| `--approx_ratio` | 0.97 | Minimum approximation ratio to accept a solution |
| `--output_dir` | adapt_results_hubo | Output directory |

---

## Step 2 — Tokenize Data

Converts the ADAPT-QAOA results into token sequences for nanoGPT training.

```bash
python step2_tokenize.py \
    --results_dir adapt_results_hubo \
    --save_dir nanoGPT/data/hubo_4nodes \
    --n_nodes 4 \
    --rounding_digits 1 \
    --approx_ratio_thr 0.5
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--results_dir` | adapt_results_hubo | Output directory from Step 1 |
| `--save_dir` | nanoGPT/data/hubo_save_dir | Output directory for tokenized data |
| `--n_nodes` | 4 | Must match Step 1 |
| `--rounding_digits` | 1 | Must match Step 1 |
| `--approx_ratio_thr` | 0.5 | Minimum approximation ratio filter |

---

## Step 3 — Train nanoGPT

Trains the GPT model on the tokenized HUBO data.

```bash
python step3_train.py \
    --data_dir nanoGPT/data/hubo_4nodes \
    --max_iters 5000
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | nanoGPT/data/hubo_save_dir | Tokenized data directory from Step 2 |
| `--max_iters` | (from config) | Training iterations |
| `--batch_size` | (from config) | Batch size |
| `--n_layer` / `--n_head` / `--n_embd` | (from config) | Model architecture |
| `--device` | (from config) | `cuda`, `cpu`, or `mps` |
| `--ckpt_name` | None | Optional name for the saved checkpoint |

The best checkpoint is saved to `nanoGPT/out-<data_dir_name>/ckpt.pt`.

---

## Step 4 — Test Inference

Tests the trained GPT model on unseen sub-HUBOs and reports approximation ratio (AR) statistics compared to brute-force or simulated annealing references.

```bash
python step4_inference_test.py \
    --ckpt nanoGPT/out-hubo_4nodes/ckpt.pt \
    --data_dir nanoGPT/data/hubo_4nodes \
    --n_nodes 4 --n_test 100 --trim_n 5 \
    --from_formulated --hubo_size 100 --num_gpt_samples 10
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | — | Path to trained checkpoint (.pt file) |
| `--data_dir` | dirname(ckpt) | Directory with `meta.pkl` |
| `--n_nodes` | 4 | Sub-HUBO size — must match training |
| `--n_test` | 10 | Number of sub-HUBOs to test |
| `--num_gpt_samples` | 10 | GPT candidate circuits per problem; best energy is kept |
| `--temperature` | 0.8 | GPT sampling temperature |
| `--trim_n` | 5 | Remove worst and best N cases from trimmed summary |
| `--from_formulated` | — | Draw sub-HUBOs from a formulated large HUBO (requires `--hubo_size`) |
| `--hubo_size` | — | Full HUBO size (used with `--from_formulated`) |
| `--no_brute_force` | — | Skip brute-force reference |
| `--no_sa` | — | Skip simulated annealing reference |

---

## Step 5 — Solve Large HUBO with DQAOA

Solves a large HUBO by iteratively decomposing it into sub-HUBOs and solving each with the trained GPT model (or standard QAOA). Two modes are available: single-process and MPI-parallel.

### Single-process (`step5_solve_large_hubo.py`)

**Using GPT sub-solver:**

```bash
python step5_solve_large_hubo.py --solver gpt \
    --load_txt --hubo_size 100 --ctn 0 --hubo_dir ./HUBOs \
    --ckpt nanoGPT/out-hubo_4nodes/ckpt.pt \
    --sub_n 4 --iterations 200
```

**Using standard QAOA sub-solver:**

```bash
python step5_solve_large_hubo.py --solver qaoa \
    --load_txt --hubo_size 100 --ctn 0 --hubo_dir ./HUBOs \
    --sub_n 4 --iterations 200
```

### MPI-parallel (`step6_mpi_solve_large_hubo.py`)

Distributes sub-HUBO solves across MPI ranks: rank 0 manages decomposition and aggregation, ranks 1..N each run a solver on their own GPU. Each DQAOA cycle, all workers solve in parallel, giving `(MPI_size - 1)` solutions per cycle instead of 1.

```bash
mpirun -np 9 python step6_mpi_solve_large_hubo.py --solver gpt \
    --ckpt nanoGPT/out-hubo_4nodes/ckpt.pt \
    --load_txt --hubo_size 100 --ctn 0 --hubo_dir ./HUBOs \
    --sub_n 4 --iterations 200 --num_gpt_samples 10
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--solver` | gpt | Sub-solver: `gpt` or `qaoa` |
| `--ckpt` | — | [gpt] Path to trained checkpoint |
| `--hubo_size` | 100 | Full HUBO size N |
| `--sub_n` | 4 | Sub-HUBO size (must match training) |
| `--iterations` | 200 | DQAOA iterations |
| `--num_gpt_samples` | 10 | [gpt] GPT candidate circuits per sub-HUBO |
| `--load_txt` | — | Load HUBO from text files in `--hubo_dir` |
| `--load_formulated` | — | Load via `subHUBO_gen.formulate_HUBO()` |
| `--no_factorized` | — | Use dense O(N³) energy eval instead of factorized O(N²) |
| `--save_solution` | None | Path to save the solution vector (`.npy`) |

---

## Project Structure

```
├── step1_generate_data.py       # Step 1: data generation
├── step2_tokenize.py            # Step 2: tokenization
├── step3_train.py               # Step 3: GPT training
├── step4_inference_test.py      # Step 4: inference test (AR statistics)
├── step5_solve_large_hubo.py    # Step 5: DQAOA solver (single-process)
├── step6_mpi_solve_large_hubo.py# Step 5: DQAOA solver (MPI-parallel)
├── DQAOA.py                     # DQAOA class
├── subHUBO_gen.py               # Sub-HUBO decomposition
├── qaoa_gpt_src/                # Core library (ADAPT-QAOA, tokenizer, model interface)
├── nanoGPT/                     # nanoGPT training code
├── utils/                       # Utility solvers
├── HUBOs/                       # Example HUBO problem instances
└── requirements.txt
```

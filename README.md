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

Step 6 additionally requires MPI (`mpi4py`) since it runs as a manager/worker job across multiple ranks.

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
Step 5: Self-bootstrap more training data with the trained GPT (optional,
        feeds back into Step 2 → Step 3 for another training round)
  ↓
Step 6: Solve a large HUBO with DQAOA (MPI, multi-GPU)
```

---

## Step 1 — Generate Training Data

Generates small random (or formulated) HUBOs and solves them with ADAPT-QAOA to produce training samples.

```bash
python step1_generate_data.py \
    --n_nodes 6 \
    --hubo_number 5000 \
    --rounding_digits 1 \
    --output_dir adapt_results_hubo
```

To generate data from a formulated large HUBO (e.g. size 100):

```bash
python step1_generate_data.py \
    --from_formulated --hubo_size 100 \
    --n_nodes 6 --hubo_number 5000 \
    --rounding_digits 1 \
    --output_dir adapt_results_hubo
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--n_nodes` | 6 | Number of binary variables per sub-HUBO |
| `--hubo_number` | 5000 | Number of HUBOs to generate |
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
    --save_dir nanoGPT/data/hubo_6nodes \
    --n_nodes 6 \
    --rounding_digits 1 \
    --approx_ratio_thr 0.5
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--results_dir` | adapt_results_hubo | Output directory from Step 1 |
| `--save_dir` | nanoGPT/data/hubo_save_dir | Output directory for tokenized data |
| `--n_nodes` | 6 | Must match Step 1 |
| `--rounding_digits` | 1 | Must match Step 1 |
| `--approx_ratio_thr` | 0.5 | Minimum approximation ratio filter |

---

## Step 3 — Train nanoGPT

Trains the GPT model on the tokenized HUBO data.

```bash
python step3_train.py \
    --data_dir nanoGPT/data/hubo_6nodes \
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

The best checkpoint is saved to `nanoGPT/out-<data_dir_name>/ckpt.pt` (or `--out_dir` if given).

---

## Step 4 — Test Inference

Tests the trained GPT model on unseen sub-HUBOs and reports approximation ratio (AR) statistics compared to brute-force or simulated annealing references.

```bash
python step4_inference_test.py \
    --ckpt nanoGPT/out-hubo_6nodes/ckpt.pt \
    --data_dir nanoGPT/data/hubo_6nodes \
    --n_nodes 6 --n_test 100 --trim_n 5 \
    --from_formulated --hubo_size 100 --num_gpt_samples 10
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | — | Path to trained checkpoint (.pt file) |
| `--data_dir` | dirname(ckpt) | Directory with `meta.pkl` |
| `--n_nodes` | 6 | Sub-HUBO size — must match training |
| `--n_test` | 10 | Number of sub-HUBOs to test |
| `--num_gpt_samples` | 10 | GPT candidate circuits per problem; best energy is kept |
| `--temperature` | 0.8 | GPT sampling temperature |
| `--trim_n` | 5 | Remove worst and best N cases from trimmed summary |
| `--from_formulated` | — | Draw sub-HUBOs from a formulated large HUBO (requires `--hubo_size`) |
| `--hubo_size` | — | Full HUBO size (used with `--from_formulated`) |
| `--no_brute_force` | — | Skip brute-force reference |
| `--no_sa` | — | Skip simulated annealing reference |

---

## Step 5 — Self-Bootstrap Training Data with GPT

Once a GPT model has been trained (Steps 1–3), this step uses the trained model itself to generate *new* training data — GPT proposes a circuit for each sampled sub-HUBO directly, skipping the expensive classical ADAPT-QAOA optimization loop from Step 1. Only circuits meeting `--ar_threshold` are kept. Output is written in the same CSV format as Step 1, so it can be fed straight back into Step 2 for another round of tokenization + training (active-learning / self-bootstrap loop), or used to cheaply extend the dataset to sizes where Step 1's classical solve is too slow.

```bash
python step5_GPTdataset.py \
    --ckpt nanoGPT/out-hubo_6nodes/ckpt.pt \
    --output_dir adapt_results_hubo_bootstrap \
    --n_nodes 6 --hubo_number 500 \
    --ar_threshold 0.80 --num_gpt_samples 5
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | — | Path to trained checkpoint (.pt file) |
| `--data_dir` | dirname(ckpt) | Directory with `meta.pkl` |
| `--n_nodes` | 6 | Sub-HUBO size — must match training |
| `--hubo_number` | 500 | Number of sub-HUBOs to sample and try |
| `--ar_threshold` | 0.80 | Minimum AR (vs. brute-force/SA reference) to keep a GPT-generated circuit |
| `--num_gpt_samples` | 5 | GPT candidate circuits per sub-HUBO; best-AR one is kept |
| `--temperature` | 0.8 | GPT sampling temperature |
| `--from_formulated` | — | Draw sub-HUBOs from a formulated large HUBO (requires `--hubo_size`) |
| `--hubo_npy_dir` | — | Alternative: load sub-HUBOs from a directory of `.npy` tensors |
| `--output_dir` | adapt_results_hubo | Output directory (Step 2-compatible CSVs) |
| `--checkpoint_every` | 20 | Write partial results to disk every N accepted circuits |
| `--no_brute_force` / `--no_sa` | — | Skip the corresponding reference solver used to compute AR |

---

## Step 6 — Solve Large HUBO with DQAOA (MPI)

Solves a large HUBO by iteratively decomposing it into sub-HUBOs and solving each with the trained GPT model (or standard QAOA). Runs as an MPI job: rank 0 is the manager (decomposes, dispatches, aggregates with a greedy coordinate update); ranks 1..N are workers, each pinned to one GPU, each solving one sub-HUBO per cycle in parallel.

**Using GPT sub-solver:**

```bash
mpirun -np 9 python step6_DQAOA_GPT.py --solver gpt \
    --load_txt --hubo_size 100 --ctn 0 --hubo_dir ./HUBOs \
    --ckpt nanoGPT/out-hubo_6nodes/ckpt.pt \
    --sub_n 6 --iterations 200 --dec_algo ifd
```

**Using standard QAOA sub-solver:**

```bash
mpirun -np 5 python step6_DQAOA_GPT.py --solver qaoa \
    --load_txt --hubo_size 100 --ctn 0 --hubo_dir ./HUBOs \
    --sub_n 6 --iterations 200 --dec_algo bfs
```

`-np N` gives 1 manager + `N-1` workers; each cycle solves `N-1` sub-HUBOs in parallel. On SLURM, launch with `srun -N <nodes> -n <total ranks> python step6_DQAOA_GPT.py ...` instead of `mpirun`.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--solver` | gpt | Sub-solver: `gpt` or `qaoa` |
| `--ckpt` | — | [gpt] Path to trained checkpoint |
| `--dec_algo` | random | Sub-HUBO decomposition: `random`, `ifd`, `bfs`, or `pfs` |
| `--hubo_size` | 100 | Full HUBO size N |
| `--sub_n` | 10 | Sub-HUBO size (must match training) |
| `--iterations` | 200 | DQAOA cycles (all workers solve in parallel each cycle) |
| `--num_gpt_samples` | 1 | [gpt] GPT candidate circuits per sub-HUBO |
| `--load_txt` | — | Load HUBO from `{size}_QUBO*.txt` / `{size}_k_third*.txt` in `--hubo_dir` (see `--ctn`) |
| `--load_formulated` | — | Load via `subHUBO_gen.formulate_HUBO()` (reads `{size}_QUBO.txt` / `{size}_k_third.txt` from cwd) |
| `--hubo_npy` | — | Load a pre-saved dense HUBO tensor (`.npy`) |
| (none of the above) | — | Generate a random HUBO instead (`--density`, `--weight_range`, `--seed_hubo`) |
| `--no_factorized` | — | Use dense O(N³) energy eval instead of factorized O(N²) |
| `--threads_per_worker` | 2 | OMP/MKL/torch threads per worker rank |
| `--save_solution` | None | Path to save the solution vector (`.npy`) |

---

## Project Structure

```
├── step1_generate_data.py       # Step 1: data generation (ADAPT-QAOA)
├── step2_tokenize.py            # Step 2: tokenization
├── step3_train.py               # Step 3: GPT training
├── step4_inference_test.py      # Step 4: inference test (AR statistics)
├── step5_GPTdataset.py          # Step 5: self-bootstrap data generation via GPT
├── step6_DQAOA_GPT.py           # Step 6: MPI DQAOA solver
├── DQAOA.py                     # Single-process (non-MPI) DQAOA class
├── subHUBO_gen.py                # Sub-HUBO decomposition + formulated-HUBO loading
├── save_qasm.py / run_qasm_aer.py  # Export a GPT circuit to OpenQASM and replay it on qiskit-aer
├── draw_circuit.py              # Visualize a GPT-generated circuit
├── check_duplicates.py          # Dedupe generated sub-HUBOs by hash
├── qaoa_gpt_src/                # Core library (ADAPT-QAOA, tokenizer, model interface)
├── nanoGPT/                     # nanoGPT training code
├── utils/                       # Solver utilities (GPT sub-solver, standard-QAOA CudaQ solver)
├── HUBOs/                       # Example large HUBO instances (QUBO + k_third text files)
├── 01_data/                     # Example Step 1 output (ADAPT-QAOA training data)
├── 03_trained_GPT_model/        # Example trained checkpoints (6/8/10-node)
└── requirements.txt
```

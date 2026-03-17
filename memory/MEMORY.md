# QAOA-GPT-HUBO Project Memory

## Project Overview
Extends DQAOA-GPT-QUBO (MaxCut via 2-body Hamiltonian) to HUBO problems
(up to 3rd-order interactions with ZZZ terms in the Hamiltonian).

## New Files Created (HUBO Extension)
- `qaoa_gpt_src/hubo_hamiltonian.py` — HUBO → cudaq SpinOperator, projected graph, classical solvers
- `qaoa_gpt_src/generate_adapt_hubo_data.py` — training data generation
- `qaoa_gpt_src/prepare_tokens_hubo.py` — tokenization (extends vocab with 1-tuples, 3-tuples)
- `qaoa_gpt_src/util_hubo.py` — inference utilities, HUBO circuit execution
- `qaoa_gpt_src/model_interface_hubo.py` — HUBO_GPT class (analogous to QAOA_GPT)
- `utils/solve_HUBO_GPT.py` — main solver entry: `solve_sub_HUBO_GPT(sub_HUBO)`
- `dqHUBO.py` — DQAOA loop: `dqHUBO` class and `solve_large_HUBO()`

## HUBO Tensor Convention
T[n,n,n] in canonical upper-triangular form:
- T[i,i,i]  = coeff of x_i          (linear)
- T[i,i,j]  = coeff of x_i*x_j      (quadratic, i<j)
- T[i,j,k]  = coeff of x_i*x_j*x_k  (cubic, i<j<k)

## Substitution: x_i = (1 - z_i) / 2
Cubic term maps to: linear z, ZZ pairs, ZZZ all with coeff/8.

## Token Sequence Format (HUBO)
[bos] + [(i,) w]* + [(i,j) w]* + [(i,j,k) w]* + [end_of_graph] + [circuit] + [eos]

## FEATHER Input
Projected graph: cubic (i,j,k) → triangle edges (i,j),(i,k),(j,k) with w/3 each.

## Pipeline Steps
1. generate_data_hubo() → res/ and hubo/ CSVs
2. generate_tokens_hubo() → train/val/test .npy + meta.pkl (extended vocab)
3. nanoGPT training (same train_pad_gemb.py, uses HUBO meta.pkl)
4. solve_sub_HUBO_GPT(T) → binary vector
5. dqHUBO.run() / solve_large_HUBO(T) → solution for large HUBO

## Key Data Paths (adjust after training)
- Model checkpoint: nanoGPT/out-hubo_save_dir/ckpt_hubo.pt
- Data dir: nanoGPT/data/hubo_save_dir/
- Config: nanoGPT/data/hubo_save_dir/train_adapt_gpt_config.py

## Existing QUBO Pipeline (unchanged)
- `utils/solve_Qubo_GPT.py` → `solve_Q_GPT(Q_matrix)`
- `dqQAOA.py` → `dqQAOA` class
- Training: `qaoa_gpt_src/generate_adapt_qaoa_data.py` + `prepare_tokens.py`

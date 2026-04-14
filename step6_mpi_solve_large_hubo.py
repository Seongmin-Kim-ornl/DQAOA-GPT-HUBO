"""
STEP 6 — MPI-parallel DQAOA-GPT-HUBO.

Mirrors step5_solve_large_hubo.py but distributes sub-HUBO solves across MPI
ranks so all workers solve in parallel each cycle.

  Rank 0   : manager — decompose sub-HUBOs, send to workers, aggregate results
  Rank 1..N: workers — each assigned one GPU, receive sub-HUBO, solve, send back

Each DQAOA cycle, rank 0 sends one sub-HUBO to every worker simultaneously,
collecting (MPI_size - 1) solutions per cycle instead of 1.

Usage:
    mpirun -np 9 python step6_mpi_solve_large_hubo.py \\
        --solver gpt \\
        --ckpt nanoGPT/out-hubo_save_dir/ckpt_5000.pt \\
        --hubo_size 100 --sub_n 4 --iterations 200 \\
        --num_gpt_samples 10

    mpirun -np 5 python step6_mpi_solve_large_hubo.py \\
        --solver qaoa \\
        --load_txt --hubo_size 100 --ctn 0 --hubo_dir HUBOs/ \\
        --sub_n 4 --iterations 100
"""

# ---- MPI must be the very first import --------------------------------------
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---- set thread limits BEFORE importing numpy/torch/sklearn -----------------
# numpy/OpenBLAS/MKL/OpenMP all read these env vars at import time.
# Without this, each MPI worker spawns O(n_cores) threads → severe oversubscription.
import os
import sys

_force_cpu = any(a == "--device=cpu" for a in sys.argv) or (
    "--device" in sys.argv and
    sys.argv[sys.argv.index("--device") + 1] == "cpu"
)

# Sniff --cudaq_target from sys.argv before cudaq is imported
if "CUDAQ_TARGET" not in os.environ:
    for _i, _a in enumerate(sys.argv):
        if _a == "--cudaq_target" and _i + 1 < len(sys.argv):
            os.environ["CUDAQ_TARGET"] = sys.argv[_i + 1]
            break
        if _a.startswith("--cudaq_target="):
            os.environ["CUDAQ_TARGET"] = _a.split("=", 1)[1]
            break

_n_workers = max(1, size - 1)

# Use sched_getaffinity to get CPUs actually allocated to this process by SLURM.
# os.cpu_count() returns total physical CPUs (128), but SLURM may bind each
# MPI task to far fewer cores via --cpus-per-task. Without this, we spawn
# 64 threads on 1 allocated CPU → massive contention.
try:
    _n_cores = len(os.sched_getaffinity(0))
except AttributeError:
    _n_cores = os.cpu_count() or 1

_threads_per_worker = max(1, _n_cores)   # each process uses all its own CPUs

# Set thread limits for numpy/OpenBLAS/MKL/OpenMP before they are imported
if rank != 0:
    _t = str(_threads_per_worker)
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_env, _t)

# ---- now it is safe to import numpy / torch / sklearn -----------------------
import time
import random
import argparse
import numpy as np

import torch
torch.set_num_threads(_threads_per_worker)

# ---- assign GPU per worker rank ---------------------------------------------
n_gpus = torch.cuda.device_count()
if rank != 0 and n_gpus > 0 and not _force_cpu:
    gpu_id = (rank - 1) % n_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    print(f"[rank {rank}] assigned gpu {gpu_id}", flush=True)
else:
    device = 'cpu'
    if rank == 1:
        print(f"[cpu mode] rank 1 has {_n_cores} allocated CPUs "
              f"→ {_threads_per_worker} threads", flush=True)

# ---- project imports (cudaq now sees the right GPU) -------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subHUBO_gen import make_sub_HUBO
from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    evaluate_hubo_energy,
    evaluate_hubo_energy_factorized,
)
from utils.solve_HUBO_GPT import solve_sub_HUBO_GPT
from utils.qaoa_CudaQ_hubo import solve_HUBO_CUDAQ

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ============================================================================ #
# DQAOA_MPI                                                                    #
# ============================================================================ #

class DQAOA_MPI:
    """
    MPI manager-worker DQAOA for HUBO.

    Each cycle rank 0 sends one fresh sub-HUBO to each worker.  Workers solve
    in parallel and return their bitstring.  Rank 0 greedily aggregates all
    results before the next cycle.
    """

    def __init__(self,
                 T,
                 sub_n,
                 iterations,
                 # GPT solver
                 model_ckpt=None,
                 data_dir=None,
                 config_file=None,
                 pool_type='all_pool',
                 num_shots=1024,
                 num_gpt_samples=10,
                 temperature=0.8,
                 # standard QAOA solver
                 qaoa_num_layers=2,
                 qaoa_max_iters=200,
                 qaoa_target='nvidia',
                 qaoa_reverse_bitorder=False,
                 # shared
                 solver='gpt',
                 seed=12345,
                 verbose=False,
                 # factorized energy (optional)
                 Q=None,
                 k_third=None):

        self.T               = T
        self.N               = T.shape[0]
        self.sub_n           = sub_n
        self.iterations      = iterations
        self.model_ckpt      = model_ckpt
        self.data_dir        = data_dir
        self.config_file     = config_file
        self.pool_type       = pool_type
        self.num_shots       = num_shots
        self.num_gpt_samples = num_gpt_samples
        self.temperature     = temperature
        self.qaoa_num_layers = qaoa_num_layers
        self.qaoa_max_iters  = qaoa_max_iters
        self.qaoa_target     = qaoa_target
        self.qaoa_reverse_bitorder = qaoa_reverse_bitorder
        self.solver          = solver
        self.seed            = seed
        self.verbose         = verbose
        self.Q               = Q
        self.k_third         = k_third
        self._use_factorized = (Q is not None) and (k_third is not None)

    def _eval_energy(self, x):
        if self._use_factorized:
            return evaluate_hubo_energy_factorized(self.Q, self.k_third, x)
        return evaluate_hubo_energy(self.T, x)

    def initialize(self):
        rng = np.random.default_rng(self.seed)
        self.x_optimal = rng.integers(0, 2, size=self.N).astype(float)

    def decompose(self):
        """Rank 0: randomly extract one sub-HUBO."""
        random.seed(int(time.time() * 1e6) % (2**31))
        sub_T, choicelist = make_sub_HUBO(self.T, self.sub_n)
        return sub_T, choicelist

    def solve(self, sub_T):
        """Workers: solve sub-HUBO with the configured solver."""
        if self.solver == 'gpt':
            return solve_sub_HUBO_GPT(
                sub_T,
                model_ckpt=self.model_ckpt,
                data_dir=self.data_dir,
                config_file=self.config_file,
                device=device,
                num_shots=self.num_shots,
                pool_type=self.pool_type,
                num_gpt_samples=self.num_gpt_samples,
                temperature=self.temperature,
                verbose=False,
            )
        else:
            return solve_HUBO_CUDAQ(
                sub_T,
                num_layers=self.qaoa_num_layers,
                max_iters=self.qaoa_max_iters,
                shots=self.num_shots,
                target=self.qaoa_target,
                reverse_bitorder=self.qaoa_reverse_bitorder,
            )

    def aggregate(self, x_sub, choicelist):
        """Rank 0: greedy coordinate update — same as DQAOA.aggregate()."""
        current_energy = self._eval_energy(self.x_optimal)
        for local_idx, global_idx in enumerate(choicelist):
            x_candidate = self.x_optimal.copy()
            x_candidate[global_idx] = x_sub[local_idx]
            candidate_energy = self._eval_energy(x_candidate)
            if candidate_energy < current_energy:
                self.x_optimal = x_candidate
                current_energy = candidate_energy

    def run(self):
        """
        MPI manager-worker loop.

        Rank 0  : sends one sub-HUBO to each worker per cycle, collects &
                  aggregates all (MPI_size - 1) results, prints progress.
        Rank 1+ : receives sub-HUBO, solves, sends bitstring + choicelist back.
        """
        MPI_size = comm.Get_size()
        n_workers = MPI_size - 1

        self.initialize()

        if rank == 0:
            init_e = self._eval_energy(self.x_optimal)
            print(f"[rank 0] initialized  energy={init_e:.4f}", flush=True)
            print(f"[rank 0] {n_workers} workers × {self.iterations} cycles "
                  f"= {n_workers * self.iterations} total sub-HUBO solves",
                  flush=True)
            tic = time.time()

        for cycle in range(self.iterations):

            # ---- rank 0: decompose and dispatch to all workers --------------
            if rank == 0:
                for worker in range(1, MPI_size):
                    sub_T, choicelist = self.decompose()
                    comm.send(sub_T,      dest=worker, tag=10)
                    comm.send(choicelist, dest=worker, tag=11)

            # ---- workers: receive, solve, send back -------------------------
            if rank != 0:
                sub_T      = comm.recv(source=0, tag=10)
                choicelist = comm.recv(source=0, tag=11)
                t0 = MPI.Wtime()
                if self.verbose:
                    print(f"  [rank {rank}] cycle {cycle+1}: solving...",
                          flush=True)
                x_sub = self.solve(sub_T)
                if self.verbose:
                    print(f"  [rank {rank}] cycle {cycle+1}: done "
                          f"({MPI.Wtime()-t0:.1f}s)  "
                          f"bits={''.join(str(int(b)) for b in x_sub)}",
                          flush=True)
                comm.send(x_sub,      dest=0, tag=20)
                comm.send(choicelist, dest=0, tag=21)

            # ---- rank 0: collect and aggregate all worker results -----------
            if rank == 0:
                for worker in range(1, MPI_size):
                    x_sub      = comm.recv(source=worker, tag=20)
                    choicelist = comm.recv(source=worker, tag=21)
                    self.aggregate(x_sub, choicelist)

                cur_e   = self._eval_energy(self.x_optimal)
                elapsed = time.time() - tic
                print(f"[rank 0] cycle {cycle+1}/{self.iterations}  "
                      f"energy={cur_e:.4f}  elapsed={elapsed:.1f}s",
                      flush=True)

        if rank == 0:
            best_energy = self._eval_energy(self.x_optimal)
            print(f"[rank 0] done  energy={best_energy:.4f}  "
                  f"total={time.time()-tic:.1f}s", flush=True)
            return self.x_optimal.astype(int), best_energy

        return None, None


# ============================================================================ #
# Helpers — same as step5_solve_large_hubo.py                                  #
# ============================================================================ #

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
    if rank == 0:
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
    return T, Q, k_third


# ============================================================================ #
# CLI                                                                           #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="MPI-parallel DQAOA-GPT-HUBO (step 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--solver", type=str, default="gpt",
                   choices=["gpt", "qaoa"],
                   help="Sub-HUBO solver: 'gpt' (QAOA-GPT) or 'qaoa' (standard QAOA)")

    # GPT solver
    p.add_argument("--ckpt",        type=str, default=None,
                   help="[gpt] Path to trained HUBO-GPT checkpoint (.pt file)")
    p.add_argument("--data_dir",    type=str, default=None,
                   help="[gpt] Directory with meta.pkl. Defaults to dirname(--ckpt).")
    p.add_argument("--config_file", type=str, default=None,
                   help="[gpt] Config .py. Defaults to <data_dir>/train_adapt_gpt_config.py")
    p.add_argument("--device",       type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="[gpt] Device override (normally auto-assigned per rank)")
    p.add_argument("--cudaq_target", type=str, default="auto",
                   choices=["auto", "nvidia", "qpp-cpu"],
                   help="CUDA-Q target for circuit simulation. "
                        "'auto' uses nvidia if GPU available, else qpp-cpu (default: auto)")
    p.add_argument("--pool_type",   type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])
    p.add_argument("--num_gpt_samples", type=int,   default=10,
                   help="[gpt] GPT candidate circuits per sub-HUBO; best energy kept (default: 10)")
    p.add_argument("--temperature",     type=float, default=0.8,
                   help="[gpt] GPT sampling temperature (default: 0.8)")

    # Standard QAOA solver
    p.add_argument("--qaoa_num_layers",      type=int,  default=2,
                   help="[qaoa] Circuit depth p")
    p.add_argument("--qaoa_max_iters",       type=int,  default=200,
                   help="[qaoa] Max COBYLA iterations per sub-HUBO")
    p.add_argument("--qaoa_target",          type=str,  default="nvidia",
                   choices=["nvidia", "qpp-cpu"],
                   help="[qaoa] CUDA-Q backend")
    p.add_argument("--qaoa_reverse_bitorder", action="store_true",
                   help="[qaoa] Reverse CUDA-Q bitstring ordering")

    # HUBO source (mutually exclusive)
    hubo_src = p.add_mutually_exclusive_group()
    hubo_src.add_argument("--hubo_npy",       type=str, default=None,
                          help="Path to a pre-saved full HUBO tensor (.npy)")
    hubo_src.add_argument("--load_formulated", action="store_true",
                          help="Load via subHUBO_gen.formulate_HUBO()")
    hubo_src.add_argument("--load_txt",        action="store_true",
                          help="Load from {size}_QUBO*.txt and {size}_k_third*.txt")

    p.add_argument("--hubo_size",    type=int,   default=100)
    p.add_argument("--ctn",          type=int,   default=0,
                   help="[load_txt] Copy number in filename")
    p.add_argument("--hubo_dir",     type=str,   default=".",
                   help="[load_txt] Directory containing QUBO/k_third text files")
    p.add_argument("--density",      type=float, default=0.5)
    p.add_argument("--weight_range", type=float, default=1.0)
    p.add_argument("--seed_hubo",    type=int,   default=0)

    p.add_argument("--no_factorized", action="store_true",
                   help="Force dense O(N^3) energy evaluation instead of factorized O(N^2)")

    p.add_argument("--sub_n",       type=int, default=4,
                   help="Sub-HUBO size (must match n_nodes from training for --solver gpt)")
    p.add_argument("--iterations",  type=int, default=200,
                   help="DQAOA cycles (each cycle: all workers solve in parallel)")
    p.add_argument("--num_shots",   type=int, default=1024)
    p.add_argument("--seed_dq",     type=int, default=12345)

    p.add_argument("--save_solution", type=str, default=None)
    p.add_argument("--verbose",       action="store_true")

    return p.parse_args()


# ============================================================================ #
# Main                                                                          #
# ============================================================================ #

def main():
    args = parse_args()

    # Resolve unified cudaq target (same logic as step5)
    cudaq_target = args.cudaq_target
    if cudaq_target == "auto":
        cudaq_target = os.environ.get("CUDAQ_TARGET", "nvidia")
    qaoa_target = args.qaoa_target if args.qaoa_target != "nvidia" else cudaq_target

    if size < 2:
        if rank == 0:
            print("ERROR: need at least 2 ranks (1 manager + 1 worker).\n"
                  "  Run: mpirun -np <N+1> python step6_mpi_solve_large_hubo.py ...")
        sys.exit(1)

    # Resolve GPT paths
    ckpt = data_dir = config_file = None
    if args.solver == 'gpt':
        if args.ckpt is None:
            if rank == 0:
                print("ERROR: --ckpt is required for --solver gpt")
            sys.exit(1)
        ckpt        = args.ckpt
        data_dir    = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
        config_file = args.config_file or os.path.join(data_dir,
                                                        'train_adapt_gpt_config.py')
        for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                            (config_file, "config_file")]:
            if not os.path.exists(path):
                if rank == 0:
                    print(f"ERROR: {label} not found: {path}")
                sys.exit(1)

    # All ranks load the same HUBO tensor
    Q_full = k_third_full = None

    if args.hubo_npy:
        if rank == 0:
            print(f"Loading HUBO tensor from: {args.hubo_npy}", flush=True)
        T_full    = np.load(args.hubo_npy)
        hubo_size = T_full.shape[0]

    elif args.load_formulated:
        if rank == 0:
            print(f"Loading formulated HUBO (size={args.hubo_size})...", flush=True)
        from subHUBO_gen import formulate_HUBO, load_QUBO, load_k_third
        T_full = formulate_HUBO(args.hubo_size)
        if not args.no_factorized:
            Q_full       = load_QUBO(args.hubo_size)
            k_third_full = load_k_third(args.hubo_size)
        hubo_size = args.hubo_size

    elif args.load_txt:
        if rank == 0:
            print(f"Loading HUBO from text files (size={args.hubo_size}, "
                  f"ctn={args.ctn})...", flush=True)
        T_full, Q_full, k_third_full = load_hubo_from_txt(
            args.hubo_size, ctn=args.ctn, hubo_dir=args.hubo_dir)
        if args.no_factorized:
            Q_full = k_third_full = None
        hubo_size = args.hubo_size

    else:
        if rank == 0:
            print(f"Generating random HUBO (size={args.hubo_size}, "
                  f"seed={args.seed_hubo})...", flush=True)
        T_full = generate_random_hubo(
            args.hubo_size,
            density=args.density,
            weight_range=args.weight_range,
            seed=args.seed_hubo,
        )
        hubo_size = args.hubo_size

    comm.Barrier()

    if rank == 0:
        print("=" * 60, flush=True)
        print("STEP 6: MPI DQAOA-GPT-HUBO", flush=True)
        print("=" * 60, flush=True)
        print(f"  ranks        : {size}  (1 manager + {size-1} workers)", flush=True)
        print(f"  GPUs total   : {n_gpus}", flush=True)
        if n_gpus == 0:
            _nc = os.cpu_count() or 1
            print(f"  CPU cores    : {_nc}  "
                  f"(~{max(1, _nc//(size-1))} torch threads/worker)", flush=True)
        print(f"  solver       : {args.solver}", flush=True)
        print(f"  HUBO size    : {hubo_size}", flush=True)
        print(f"  sub_n        : {args.sub_n}", flush=True)
        print(f"  iterations   : {args.iterations}  "
              f"({(size-1)*args.iterations} total sub-HUBO solves)", flush=True)
        if args.solver == 'gpt':
            print(f"  checkpoint   : {ckpt}", flush=True)
            print(f"  device       : {device}  (GPT model)", flush=True)
            print(f"  cudaq_target : {cudaq_target}  (circuit sim)", flush=True)
            print(f"  num_gpt_samples: {args.num_gpt_samples}", flush=True)
            print(f"  temperature  : {args.temperature}", flush=True)
        else:
            print(f"  qaoa_layers  : {args.qaoa_num_layers}", flush=True)
            print(f"  qaoa_iters   : {args.qaoa_max_iters}", flush=True)
            print(f"  qaoa_target  : {qaoa_target}", flush=True)
        print(f"  energy eval  : "
              f"{'factorized O(N^2)' if Q_full is not None else 'dense O(N^3)'}",
              flush=True)
        print("=" * 60, flush=True)

    try:
        solver = DQAOA_MPI(
            T=T_full,
            sub_n=args.sub_n,
            iterations=args.iterations,
            model_ckpt=ckpt,
            data_dir=data_dir,
            config_file=config_file,
            pool_type=args.pool_type,
            num_shots=args.num_shots,
            num_gpt_samples=args.num_gpt_samples,
            temperature=args.temperature,
            qaoa_num_layers=args.qaoa_num_layers,
            qaoa_max_iters=args.qaoa_max_iters,
            qaoa_target=qaoa_target,
            qaoa_reverse_bitorder=args.qaoa_reverse_bitorder,
            solver=args.solver,
            seed=args.seed_dq,
            verbose=args.verbose,
            Q=Q_full,
            k_third=k_third_full,
        )
        x_opt, best_energy = solver.run()

        if rank == 0:
            print("\n" + "=" * 60, flush=True)
            print("Results", flush=True)
            print("=" * 60, flush=True)
            print(f"  Best HUBO energy   : {best_energy:.6f}", flush=True)
            print(f"  Non-zero variables : {int(x_opt.sum())} / {hubo_size}",
                  flush=True)
            print(f"  Solution (first 20): {x_opt[:20]}", flush=True)
            if args.save_solution:
                np.save(args.save_solution, x_opt)
                print(f"  Saved to           : {args.save_solution}", flush=True)
            print("\nStep 6 complete.", flush=True)

    except KeyError as e:
        if rank == 0:
            print(f"[KEYERROR] {repr(e)}", flush=True)
        comm.Abort(99)

    except Exception as e:
        print(f"[rank {rank}] [EXCEPTION] {type(e).__name__}: {repr(e)}",
              flush=True)
        comm.Abort(98)


if __name__ == "__main__":
    main()

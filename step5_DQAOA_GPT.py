"""
STEP 6 — MPI-parallel DQAOA-GPT-HUBO with structured decomposition.

Rank 0 : manager — decompose sub-HUBOs (random/IFD/BFS/PFS), dispatch to
         workers, aggregate results with greedy coordinate update.
Rank 1+: workers — each assigned one GPU, receive sub-HUBO, solve with
         QAOA-GPT (or standard QAOA), return best bitstring.

Each DQAOA cycle all workers solve one sub-HUBO in parallel, then rank 0
aggregates all (MPI_size-1) results before the next cycle.

Decomposition algorithms
------------------------
  random : uniformly sample sub_n variables (default)
  ifd    : Iterative Flip Decomposition — select top frac*N variables by
           |ΔE| as candidate pool, randomly sample sub_n per worker
  bfs    : BFS on the HUBO interaction graph starting from the highest-|ΔE|
           node; worker w uses the w-th highest-|ΔE| seed
  pfs    : Priority-First Search — greedily add the variable most strongly
           connected to the already-selected set

Usage
-----
    mpirun -np 9 python step6_DQAOA_GPT.py \\
        --solver gpt \\
        --ckpt nanoGPT/out-hubo_save_dir/ckpt_10000.pt \\
        --hubo_size 100 --sub_n 10 --iterations 200 \\
        --num_gpt_samples 10 --dec_algo ifd

    mpirun -np 5 python step6_DQAOA_GPT.py \\
        --solver qaoa \\
        --load_txt --hubo_size 100 --ctn 0 --hubo_dir HUBOs/ \\
        --sub_n 10 --iterations 100 --dec_algo bfs
"""

# ---- MPI must be the very first import ----------------------------------------
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---- thread limits BEFORE importing numpy / torch -----------------------------
import os
import sys

_force_cpu = any(a == "--device=cpu" for a in sys.argv) or (
    "--device" in sys.argv and
    sys.argv[sys.argv.index("--device") + 1] == "cpu"
)

if "CUDAQ_TARGET" not in os.environ:
    for _i, _a in enumerate(sys.argv):
        if _a == "--cudaq_target" and _i + 1 < len(sys.argv):
            os.environ["CUDAQ_TARGET"] = sys.argv[_i + 1]
            break
        if _a.startswith("--cudaq_target="):
            os.environ["CUDAQ_TARGET"] = _a.split("=", 1)[1]
            break

try:
    _n_cores = len(os.sched_getaffinity(0))
except AttributeError:
    _n_cores = os.cpu_count() or 1

# Sniff --threads_per_worker before argparse is available
_threads_per_worker = 2   # default
for _i, _a in enumerate(sys.argv):
    if _a == "--threads_per_worker" and _i + 1 < len(sys.argv):
        _threads_per_worker = max(1, int(sys.argv[_i + 1]))
        break
    if _a.startswith("--threads_per_worker="):
        _threads_per_worker = max(1, int(_a.split("=", 1)[1]))
        break

if rank != 0:
    _t = str(_threads_per_worker)
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_env] = _t   # override, not setdefault

# ---- assign GPU BEFORE importing torch (avoids CUDA init ordering issue) -----
# CUDA_VISIBLE_DEVICES must be set before torch (or any CUDA library) initializes
# its context.  We detect physical GPU count via SLURM_GPUS_ON_NODE (reliable
# inside a SLURM job) or nvidia-smi (fallback for interactive use).
def _count_physical_gpus():
    val = os.environ.get('SLURM_GPUS_ON_NODE', '').strip()
    if val.isdigit() and int(val) > 0:
        return int(val)
    try:
        import subprocess as _sp
        r = _sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return len([l for l in r.stdout.strip().split('\n') if l.strip()])
    except Exception:
        pass
    return 0

_n_gpus_physical = _count_physical_gpus()   # visible to this node; 0 = CPU-only
if rank != 0 and not _force_cpu:
    if _n_gpus_physical > 0:
        # Use SLURM_LOCALID (0-indexed within each node, always unique on that node)
        # rather than (rank-1)%n_gpus.  The global-rank formula can alias when SLURM
        # puts an unequal number of tasks on each node: e.g. with ntasks=17 on 2 nodes,
        # SLURM may put 8 tasks on node1 and 9 on node2, making ranks 8 and 16 both
        # compute gpu_id=7 and collide (CUDA exclusive-process mode → "CUDA busy").
        # SLURM_LOCALID is not set when running outside SLURM; fall back to rank-1.
        _local_id = int(os.environ.get('SLURM_LOCALID', str(rank - 1)))
        _gpu_id = _local_id % _n_gpus_physical
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
        # CudaQ's _set_cudaq_target() (triggered at project import) will now
        # see exactly 1 GPU and pick the nvidia target automatically.
    else:
        # No GPU on this node — force CudaQ to CPU before project imports fire.
        os.environ['CUDAQ_TARGET'] = 'qpp-cpu'

# ---- safe to import numpy / torch now (CUDA_VISIBLE_DEVICES already committed) --
import time
import random
import argparse
import numpy as np

import torch
torch.set_num_threads(_threads_per_worker)

# torch.cuda.device_count() now reflects CUDA_VISIBLE_DEVICES set above:
# 1 for GPU workers, 0 for CPU-only workers.
n_gpus = torch.cuda.device_count()
if rank != 0 and n_gpus > 0 and not _force_cpu:
    device = 'cuda'
    if rank == 1:
        print(f"[gpu mode] rank 1: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
              flush=True)
else:
    device = 'cpu'
    if rank == 1:
        print(f"[cpu mode] rank 1: {_n_cores} CPUs avail → {_threads_per_worker} threads/worker",
              flush=True)

# ---- project imports ---------------------------------------------------------
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
# Decomposition utilities — HUBO-native IFD / BFS / PFS                        #
# ============================================================================ #

def _hubo_flip_deltas(T, x):
    """
    Approximate |ΔE_m| for each variable m using the linear field:
      h[m] = sum over all tensor positions where m appears, contracted with x.
    ΔE_m ≈ |(1 - 2*x[m]) * h[m]|.  O(N^3) via einsum.
    """
    x = np.asarray(x, dtype=float)
    h = (np.einsum('mjk,j,k->m', T, x, x) +
         np.einsum('imk,i,k->m', T, x, x) +
         np.einsum('ijm,i,j->m', T, x, x))
    deltas = np.abs((1.0 - 2.0 * x) * h)
    if deltas.sum() == 0.0:
        deltas = np.ones(T.shape[0])   # flat fallback → uniform random
    return deltas


def _hubo_interaction_weight(T):
    """
    Symmetric N×N weight matrix: W[a,b] = sum_k(|T[a,b,k]|+|T[a,k,b]|+|T[k,a,b]|).
    Captures total interaction strength between every pair of variables.
    """
    absT = np.abs(T)
    W = (np.sum(absT, axis=2) +   # sum_k |T[a,b,k]|
         np.sum(absT, axis=1) +   # sum_j |T[a,j,b]|  (axis-1 removes middle)
         np.sum(absT, axis=0))    # sum_i |T[i,a,b]|  (axis-0 removes first)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def _decomp_batch_random(T, sub_n, n_workers, base_seed):
    """n_workers independent random sub-HUBOs."""
    N = T.shape[0]
    rng = random.Random(base_seed)
    results = []
    for _ in range(n_workers):
        chosen = sorted(rng.sample(range(N), sub_n))
        idx = np.ix_(chosen, chosen, chosen)
        results.append((T[idx].copy(), chosen))
    return results


def _decomp_batch_ifd(T, sub_n, n_workers, x_opt, frac, base_seed):
    """
    IFD: top frac*N variables by |ΔE| form a candidate pool;
    each worker randomly draws sub_n from that pool.
    """
    N = T.shape[0]
    deltas  = _hubo_flip_deltas(T, x_opt)
    n_cand  = max(sub_n, min(N, int(np.ceil(frac * N))))
    top_idx = np.argsort(deltas)[::-1][:n_cand].tolist()

    rng = random.Random(base_seed)
    results = []
    for _ in range(n_workers):
        chosen = sorted(rng.sample(top_idx, sub_n))
        idx = np.ix_(chosen, chosen, chosen)
        results.append((T[idx].copy(), chosen))
    return results


def _decomp_batch_bfs(T, sub_n, n_workers, x_opt, base_seed, cycle=0):
    """
    BFS on HUBO interaction graph.  Worker w starts from the (w+cycle)-th
    highest-|ΔE| node; neighbors expanded in order of decreasing weight.
    Rotating by cycle ensures diversity across cycles even with 1 worker.
    """
    N = T.shape[0]
    deltas      = _hubo_flip_deltas(T, x_opt)
    W           = _hubo_interaction_weight(T)
    sorted_seeds = np.argsort(deltas)[::-1]

    # Adjacency lists sorted by decreasing interaction weight
    neighbor_lists = []
    for i in range(N):
        nbr = np.where(W[i] > 0)[0]
        if len(nbr):
            nbr = nbr[np.argsort(W[i, nbr])[::-1]]
        neighbor_lists.append(nbr.tolist())

    rng = random.Random(base_seed)
    results = []
    for w in range(n_workers):
        seed_node = int(sorted_seeds[(w + cycle * n_workers) % N])
        visited   = []
        queue     = [seed_node]
        seen      = {seed_node}

        while queue and len(visited) < sub_n:
            node = queue.pop(0)
            visited.append(node)
            for nb in neighbor_lists[node]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

        if len(visited) < sub_n:
            remaining = [i for i in range(N) if i not in set(visited)]
            visited  += rng.sample(remaining, sub_n - len(visited))

        chosen = sorted(visited[:sub_n])
        idx = np.ix_(chosen, chosen, chosen)
        results.append((T[idx].copy(), chosen))
    return results


def _decomp_batch_pfs(T, sub_n, n_workers, x_opt, base_seed, cycle=0):
    """
    PFS: greedily add the variable maximally connected to the current
    selected set (by cumulative interaction weight).  Worker w starts
    from the (w+cycle)-th highest-|ΔE| seed.
    Rotating by cycle ensures diversity across cycles even with 1 worker.
    """
    N = T.shape[0]
    deltas       = _hubo_flip_deltas(T, x_opt)
    W            = _hubo_interaction_weight(T)
    sorted_seeds = np.argsort(deltas)[::-1]

    rng = random.Random(base_seed)
    results = []
    for w in range(n_workers):
        seed_node    = int(sorted_seeds[(w + cycle * n_workers) % N])
        selected     = [seed_node]
        selected_set = {seed_node}
        priority     = W[seed_node].copy().astype(float)
        priority[seed_node] = -np.inf

        while len(selected) < sub_n:
            best = int(np.argmax(priority))
            if priority[best] <= 0.0:
                remaining = [i for i in range(N) if i not in selected_set]
                if not remaining:
                    break
                best = rng.choice(remaining)
            selected.append(best)
            selected_set.add(best)
            priority    += W[best]
            for s in selected_set:
                priority[s] = -np.inf

        chosen = sorted(selected[:sub_n])
        idx = np.ix_(chosen, chosen, chosen)
        results.append((T[idx].copy(), chosen))
    return results


# ============================================================================ #
# DQAOA_MPI                                                                     #
# ============================================================================ #

class DQAOA_MPI:
    """
    MPI manager-worker DQAOA-GPT-HUBO.

    Rank 0  : decompose → dispatch → aggregate (greedy coord update).
    Rank 1+ : receive sub-HUBO → solve with QAOA-GPT or standard QAOA → return.
    """

    def __init__(self,
                 T,
                 sub_n,
                 iterations,
                 # decomposition
                 dec_algo='random',
                 dec_frac=0.6,
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
        self.dec_algo        = dec_algo
        self.dec_frac        = dec_frac
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
        self.x_optimal       = None

    def _eval_energy(self, x):
        if self._use_factorized:
            return evaluate_hubo_energy_factorized(self.Q, self.k_third, x)
        return evaluate_hubo_energy(self.T, x)

    def initialize(self):
        rng = np.random.default_rng(self.seed)
        self.x_optimal = rng.integers(0, 2, size=self.N).astype(float)

    def decompose_batch(self, n_workers, cycle):
        """Rank 0: generate n_workers sub-HUBOs for this cycle."""
        base_seed = int(time.time() * 1e6) % (2**31) + cycle * 7919
        if self.dec_algo == 'random':
            return _decomp_batch_random(self.T, self.sub_n, n_workers, base_seed)
        elif self.dec_algo == 'ifd':
            return _decomp_batch_ifd(self.T, self.sub_n, n_workers,
                                     self.x_optimal, self.dec_frac, base_seed)
        elif self.dec_algo == 'bfs':
            return _decomp_batch_bfs(self.T, self.sub_n, n_workers,
                                     self.x_optimal, base_seed, cycle)
        elif self.dec_algo == 'pfs':
            return _decomp_batch_pfs(self.T, self.sub_n, n_workers,
                                     self.x_optimal, base_seed, cycle)
        else:
            raise ValueError(f"Unknown dec_algo: {self.dec_algo}")

    def solve(self, sub_T):
        """Workers: solve received sub-HUBO."""
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

    def run(self):
        MPI_size = comm.Get_size()
        n_workers = MPI_size - 1

        self.initialize()

        if rank == 0:
            init_e = self._eval_energy(self.x_optimal)
            print(f"[rank 0] initialized  energy={init_e:.4f}  "
                  f"dec_algo={self.dec_algo}", flush=True)
            print(f"[rank 0] {n_workers} workers × {self.iterations} cycles "
                  f"= {n_workers * self.iterations} sub-HUBO solves", flush=True)
            tic = time.time()

        for cycle in range(self.iterations):

            # ---- rank 0: decompose and dispatch (non-blocking) ----------------
            # isend posts all sends simultaneously so all workers start together,
            # rather than each waiting for the previous worker to be served first.
            if rank == 0:
                batch = self.decompose_batch(n_workers, cycle)
                reqs = []
                for worker, (sub_T, choicelist) in enumerate(batch, start=1):
                    reqs.append(comm.isend(sub_T,      dest=worker, tag=10))
                    reqs.append(comm.isend(choicelist, dest=worker, tag=11))
                MPI.Request.Waitall(reqs)

            # ---- workers: receive, solve, send back ---------------------------
            if rank != 0:
                sub_T      = comm.recv(source=0, tag=10)
                choicelist = comm.recv(source=0, tag=11)
                t0 = MPI.Wtime()
                if self.verbose:
                    print(f"  [rank {rank}] cycle {cycle+1}: solving...", flush=True)
                x_sub = self.solve(sub_T)
                if self.verbose:
                    print(f"  [rank {rank}] cycle {cycle+1}: done "
                          f"({MPI.Wtime()-t0:.1f}s)  "
                          f"bits={''.join(str(int(b)) for b in x_sub)}", flush=True)
                comm.send(x_sub,      dest=0, tag=20)
                comm.send(choicelist, dest=0, tag=21)

            # ---- rank 0: collect from any worker, then aggregate in one pass --
            # ANY_SOURCE: receive from whichever worker finishes first instead of
            # blocking on a fixed order (avoids waiting for one slow worker while
            # all others are already done).
            # Single energy eval at the start of aggregation (instead of one per
            # worker call), since current_energy is tracked through all updates.
            if rank == 0:
                worker_results = []
                for _ in range(n_workers):
                    status = MPI.Status()
                    x_sub      = comm.recv(source=MPI.ANY_SOURCE, tag=20, status=status)
                    src        = status.Get_source()
                    choicelist = comm.recv(source=src, tag=21)
                    worker_results.append((x_sub, choicelist))

                current_energy = self._eval_energy(self.x_optimal)
                for x_sub, choicelist in worker_results:
                    for local_idx, global_idx in enumerate(choicelist):
                        x_candidate = self.x_optimal.copy()
                        x_candidate[global_idx] = x_sub[local_idx]
                        candidate_energy = self._eval_energy(x_candidate)
                        if candidate_energy < current_energy:
                            self.x_optimal = x_candidate
                            current_energy = candidate_energy

                elapsed = time.time() - tic
                print(f"[rank 0] cycle {cycle+1}/{self.iterations}  "
                      f"energy={current_energy:.4f}  elapsed={elapsed:.1f}s", flush=True)

        if rank == 0:
            total_time  = time.time() - tic
            best_energy = self._eval_energy(self.x_optimal)
            print(f"[rank 0] done  energy={best_energy:.4f}  "
                  f"total={total_time:.1f}s", flush=True)
            return self.x_optimal.astype(int), best_energy, total_time

        return None, None, None


# ============================================================================ #
# Helper — load HUBO from text files                                            #
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
    raise FileNotFoundError(
        f"Cannot find {suffix} for size={size}, ctn={ctn}. Tried:\n"
        + "\n".join(f"  {os.path.join(directory, n)}" for n in candidates)
    )


def load_hubo_from_txt(hubo_size, ctn=0, hubo_dir="."):
    q_path  = _find_file(hubo_dir, hubo_size, "QUBO",    ctn)
    kt_path = _find_file(hubo_dir, hubo_size, "k_third", ctn)
    if rank == 0:
        print(f"  Q matrix  : {q_path}")
        print(f"  k_third   : {kt_path}")

    with open(q_path, "r") as f:
        Q = np.array(np.reshape(np.asmatrix(f.read()), (hubo_size, hubo_size)))
    with open(kt_path, "r") as f:
        k_third = np.array(np.reshape(np.asmatrix(f.read()), (hubo_size, 8)))

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
# CLI                                                                            #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="MPI-parallel DQAOA-GPT-HUBO (step 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--solver", type=str, default="gpt",
                   choices=["gpt", "qaoa"],
                   help="Sub-HUBO solver: 'gpt' (QAOA-GPT hybrid) or 'qaoa' (standard QAOA)")

    # ---- GPT solver -----------------------------------------------------------
    p.add_argument("--ckpt",        type=str, default=None,
                   help="[gpt] Path to trained checkpoint (.pt)")
    p.add_argument("--data_dir",    type=str, default=None,
                   help="[gpt] Directory with meta.pkl (default: dirname(--ckpt))")
    p.add_argument("--config_file", type=str, default=None,
                   help="[gpt] Config .py (default: data_dir/train_adapt_gpt_config.py)")
    p.add_argument("--device",      type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="[gpt] Torch device for GPT inference")
    p.add_argument("--cudaq_target", type=str, default="auto",
                   choices=["auto", "nvidia", "qpp-cpu"],
                   help="CUDA-Q simulation target ('auto' → nvidia if GPU present)")
    p.add_argument("--pool_type",   type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"],
                   help="[gpt] ADAPT operator pool — must match training")
    p.add_argument("--num_gpt_samples", type=int, default=10,
                   help="[gpt] GPT circuits per sub-HUBO; best-energy solution kept")
    p.add_argument("--temperature",     type=float, default=0.8,
                   help="[gpt] GPT sampling temperature")

    # ---- standard QAOA solver -------------------------------------------------
    p.add_argument("--qaoa_num_layers",       type=int,  default=2)
    p.add_argument("--qaoa_max_iters",        type=int,  default=200)
    p.add_argument("--qaoa_target",           type=str,  default="nvidia",
                   choices=["nvidia", "qpp-cpu"])
    p.add_argument("--qaoa_reverse_bitorder", action="store_true")

    # ---- decomposition --------------------------------------------------------
    p.add_argument("--dec_algo", type=str, default="random",
                   choices=["random", "ifd", "bfs", "pfs"],
                   help="Sub-HUBO decomposition strategy")
    p.add_argument("--dec_frac", type=float, default=0.6,
                   help="[ifd] Fraction of N variables in the candidate pool")

    # ---- HUBO source ----------------------------------------------------------
    hubo_src = p.add_mutually_exclusive_group()
    hubo_src.add_argument("--hubo_npy",        type=str,  default=None,
                          help="Pre-saved HUBO tensor (.npy)")
    hubo_src.add_argument("--load_formulated", action="store_true",
                          help="Load via subHUBO_gen.formulate_HUBO()")
    hubo_src.add_argument("--load_txt",        action="store_true",
                          help="Load from {size}_QUBO*.txt / {size}_k_third*.txt")

    p.add_argument("--hubo_size",    type=int,   default=100)
    p.add_argument("--ctn",          type=int,   default=0,
                   help="[load_txt] Copy number in filename")
    p.add_argument("--hubo_dir",     type=str,   default=".",
                   help="[load_txt] Directory containing QUBO/k_third text files")
    p.add_argument("--density",      type=float, default=0.5)
    p.add_argument("--weight_range", type=float, default=1.0)
    p.add_argument("--seed_hubo",    type=int,   default=0)

    p.add_argument("--no_factorized", action="store_true",
                   help="Force dense O(N^3) energy eval instead of factorized O(N^2)")

    # ---- DQAOA control --------------------------------------------------------
    p.add_argument("--sub_n",       type=int, default=10,
                   help="Sub-HUBO size — should match n_nodes from GPT training")
    p.add_argument("--iterations",  type=int, default=200,
                   help="DQAOA cycles (all workers solve in parallel each cycle)")
    p.add_argument("--num_shots",   type=int, default=1024)
    p.add_argument("--seed_dq",     type=int, default=12345,
                   help="Seed for initial binary solution")

    p.add_argument("--threads_per_worker", type=int, default=2,
                   help="OMP/MKL/torch threads per worker rank (default: 2)")
    p.add_argument("--save_solution", type=str, default=None,
                   help="Save best solution vector to .npy")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# ============================================================================ #
# Main                                                                           #
# ============================================================================ #

def main():
    args = parse_args()

    cudaq_target = args.cudaq_target
    if cudaq_target == "auto":
        cudaq_target = os.environ.get(
            "CUDAQ_TARGET",
            "nvidia" if torch.cuda.is_available() else "qpp-cpu"
        )
    qaoa_target = args.qaoa_target if args.qaoa_target != "nvidia" else cudaq_target

    if size < 2:
        if rank == 0:
            print("ERROR: need at least 2 ranks (1 manager + 1 worker).\n"
                  "  Run: mpirun -np <N+1> python step6_DQAOA_GPT.py ...")
        sys.exit(1)

    # Resolve GPT checkpoint paths
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
            print(f"Loading HUBO from text (size={args.hubo_size}, "
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
        print(f"  GPUs/node    : {n_gpus}  "
              f"(~{min(size-1, n_gpus)} workers/node)", flush=True)
        if n_gpus == 0:
            _nc = os.cpu_count() or 1
            print(f"  CPU cores    : {_nc}  "
                  f"({_threads_per_worker} threads/worker)", flush=True)
        print(f"  solver       : {args.solver}", flush=True)
        print(f"  dec_algo     : {args.dec_algo}", flush=True)
        if args.dec_algo == 'ifd':
            print(f"  dec_frac     : {args.dec_frac}", flush=True)
        print(f"  HUBO size    : {hubo_size}", flush=True)
        print(f"  sub_n        : {args.sub_n}", flush=True)
        print(f"  iterations   : {args.iterations}  "
              f"({(size-1)*args.iterations} total sub-HUBO solves)", flush=True)
        if args.solver == 'gpt':
            print(f"  checkpoint   : {ckpt}", flush=True)
            print(f"  pool_type    : {args.pool_type}", flush=True)
            _worker_device = 'cuda' if _n_gpus_physical > 0 and not _force_cpu else 'cpu'
            print(f"  device       : {_worker_device}  (GPT model, workers)", flush=True)
            print(f"  cudaq_target : {cudaq_target}  (circuit sim)", flush=True)
            print(f"  gpt_samples  : {args.num_gpt_samples}", flush=True)
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
        solver_obj = DQAOA_MPI(
            T=T_full,
            sub_n=args.sub_n,
            iterations=args.iterations,
            dec_algo=args.dec_algo,
            dec_frac=args.dec_frac,
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
        x_opt, best_energy, total_time = solver_obj.run()

        if rank == 0:
            print("\n" + "=" * 60, flush=True)
            print("Results", flush=True)
            print("=" * 60, flush=True)
            print(f"  Best HUBO energy   : {best_energy:.6f}", flush=True)
            print(f"  Elapsed time       : {total_time:.1f}s", flush=True)
            print(f"  Non-zero variables : {int(x_opt.sum())} / {hubo_size}", flush=True)
            print(f"  Solution (first 20): {x_opt[:20]}", flush=True)
            if args.save_solution:
                save_dir = os.path.dirname(os.path.abspath(args.save_solution))
                os.makedirs(save_dir, exist_ok=True)
                save_path = args.save_solution
                if save_path.endswith('.npy'):
                    save_path = save_path[:-4]
                np.savez(save_path,
                         bitstring=x_opt,
                         energy=np.array([best_energy]),
                         time=np.array([total_time]))
                print(f"  Saved to           : {save_path}.npz", flush=True)
            print("\nStep 6 complete.", flush=True)

    except KeyError as e:
        if rank == 0:
            print(f"[KEYERROR] {repr(e)}", flush=True)
        comm.Abort(99)

    except Exception as e:
        print(f"[rank {rank}] [EXCEPTION] {type(e).__name__}: {repr(e)}", flush=True)
        comm.Abort(98)


if __name__ == "__main__":
    main()

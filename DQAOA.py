import numpy as np
import random
import time
import warnings

warnings.simplefilter("ignore")

from subHUBO_gen import make_sub_HUBO
from qaoa_gpt_src.hubo_hamiltonian import evaluate_hubo_energy
from utils.solve_HUBO_GPT import solve_sub_HUBO_GPT
from utils.qaoa_CudaQ_hubo import solve_HUBO_CUDAQ

class DQAOA:

    def __init__(self,
                 T,
                 sub_n,
                 iterations,
                 model_ckpt=None,
                 data_dir=None,
                 config_file=None,
                 device='cuda',
                 pool_type='all_pool',
                 num_shots=1024,
                 seed=12345,
                 verbose=False,
                 qaoa_num_layers=2,
                 qaoa_max_iters=200,
                 qaoa_target='nvidia',
                 qaoa_reverse_bitorder=False,
                 solver='gpt'):

        assert T.ndim == 3 and T.shape[0] == T.shape[1] == T.shape[2], \
            "T must be a 3D cubic tensor of shape (N, N, N)."
        assert sub_n <= T.shape[0], "sub_n cannot exceed the HUBO size."

        self.T = T
        self.N = T.shape[0]
        self.sub_n = sub_n
        self.iterations = iterations
        self.model_ckpt = model_ckpt
        self.data_dir = data_dir
        self.config_file = config_file
        self.device = device
        self.pool_type = pool_type
        self.num_shots = num_shots
        self.seed = seed
        self.verbose = verbose
        self.qaoa_num_layers = qaoa_num_layers
        self.qaoa_max_iters = qaoa_max_iters
        self.qaoa_target = qaoa_target
        self.qaoa_reverse_bitorder = qaoa_reverse_bitorder
        self.solver = solver

        self.x_optimal = None

    def initialize(self):
        rng = np.random.default_rng(self.seed)
        self.x_optimal = rng.integers(0, 2, size=self.N).astype(float)
        if self.verbose:
            init_e = evaluate_hubo_energy(self.T, self.x_optimal)
            print(f"[dqHUBO] Initialized: energy = {init_e:.4f}")

    def decompose(self):
        sub_T, choicelist = make_sub_HUBO(self.T, self.sub_n)
        return sub_T, choicelist

    def solve(self, sub_T):
        if self.solver == 'gpt':
            return solve_sub_HUBO_GPT(
                sub_T,
                model_ckpt=self.model_ckpt,
                data_dir=self.data_dir,
                config_file=self.config_file,
                device=self.device,
                num_shots=self.num_shots,
                pool_type=self.pool_type,
                verbose=self.verbose,
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
        current_energy = evaluate_hubo_energy(self.T, self.x_optimal)

        for local_idx, global_idx in enumerate(choicelist):
            x_candidate = self.x_optimal.copy()
            x_candidate[global_idx] = x_sub[local_idx]

            candidate_energy = evaluate_hubo_energy(self.T, x_candidate)
            if candidate_energy < current_energy:
                self.x_optimal = x_candidate
                current_energy = candidate_energy

    def run(self):
        self.initialize()
        tic = time.time()

        for cycle in range(self.iterations):
            sub_T, choicelist = self.decompose()
            x_sub = self.solve(sub_T)
            self.aggregate(x_sub, choicelist)

            if self.verbose and (cycle + 1) % 10 == 0:
                cur_energy = evaluate_hubo_energy(self.T, self.x_optimal)
                elapsed = time.time() - tic
                print(f"[dqHUBO] iter {cycle + 1}/{self.iterations}  "
                      f"energy={cur_energy:.4f}  elapsed={elapsed:.1f}s")

        best_energy = evaluate_hubo_energy(self.T, self.x_optimal)
        if self.verbose:
            print(f"[dqHUBO] Done. Best energy = {best_energy:.4f}  "
                  f"total time = {time.time() - tic:.1f}s")

        return self.x_optimal.astype(int), best_energy

def solve_large_HUBO(T,
                      sub_n,
                      iterations=100,
                      model_ckpt=None,
                      data_dir=None,
                      config_file=None,
                      device='cuda',
                      pool_type='all_pool',
                      num_shots=1024,
                      seed=12345,
                      verbose=True,
                      qaoa_num_layers=2,
                      qaoa_max_iters=200,
                      qaoa_target='nvidia',
                      qaoa_reverse_bitorder=False,
                      solver='gpt'):
    solver = dqHUBO(
        T=T,
        sub_n=sub_n,
        iterations=iterations,
        model_ckpt=model_ckpt,
        data_dir=data_dir,
        config_file=config_file,
        device=device,
        pool_type=pool_type,
        num_shots=num_shots,
        seed=seed,
        verbose=verbose,
        qaoa_num_layers=qaoa_num_layers,
        qaoa_max_iters=qaoa_max_iters,
        qaoa_target=qaoa_target,
        qaoa_reverse_bitorder=qaoa_reverse_bitorder,
        solver=solver,
    )
    return solver.run()

if __name__ == "__main__":
    from subHUBO_gen import formulate_HUBO

    HUBO_size = 100
    sub_HUBO_size = 4
    num_iterations = 50

    print(f"Loading HUBO of size {HUBO_size}...")
    T_full = formulate_HUBO(HUBO_size)

    print(f"Running dqHUBO with sub_n={sub_HUBO_size}, iterations={num_iterations}")
    x_opt, energy = solve_large_HUBO(
        T=T_full,
        sub_n=sub_HUBO_size,
        iterations=num_iterations,
        verbose=True,
    )

    print(f"\nFinal solution energy: {energy:.4f}")
    print(f"Solution (first 20 vars): {x_opt[:20]}")

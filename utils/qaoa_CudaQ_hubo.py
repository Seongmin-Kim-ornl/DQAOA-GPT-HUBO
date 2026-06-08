import time
from typing import List, Tuple, Optional

import numpy as np
import cudaq
from cudaq import spin
from scipy.optimize import minimize

class CudaQQAOAHUBO:
    def __init__(
        self,
        problem_size: int,
        ctn: int,
        num_layers: int = 2,
        target: str = "nvidia",
        reverse_bitorder: bool = False,
    ):
        self.problem_size = int(problem_size)
        self.ctn = int(ctn)
        self.num_layers = int(num_layers)
        self.reverse_bitorder = bool(reverse_bitorder)

        cudaq.set_target(target)

        self.z_terms: Optional[List[Tuple[Tuple[int, ...], float]]] = None
        self.shift: float = 0.0
        self.H: Optional[cudaq.SpinOperator] = None

        self.idx_a_l: Optional[List[int]] = None
        self.idx_b_l: Optional[List[int]] = None
        self.idx_c_l: Optional[List[int]] = None
        self.term_order_l: Optional[List[int]] = None
        self.term_coeff_l: Optional[List[float]] = None

        self.kernel = None

    @staticmethod
    def hubo_to_z_terms(hubo: np.ndarray) -> Tuple[List[Tuple[Tuple[int, ...], float]], float]:
        w = np.asarray(hubo, dtype=float)
        n = w.shape[0]
        terms: List[Tuple[Tuple[int, ...], float]] = []
        shift = 0.0

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c = w[i, j, k]
                    if c == 0:
                        continue

                    if (i != j and i != k and j != k):
                        terms.append(((i, j, k), -0.125 * c))
                        terms.append(((i, j), 0.125 * c))
                        terms.append(((i, k), 0.125 * c))
                        terms.append(((j, k), 0.125 * c))
                        terms.append(((i,), -0.125 * c))
                        terms.append(((j,), -0.125 * c))
                        terms.append(((k,), -0.125 * c))
                        shift += 0.125 * c

                    if (i == j and i != k):
                        terms.append(((i, k), 0.25 * c))
                        terms.append(((i,), -0.25 * c))
                        terms.append(((k,), -0.25 * c))
                        shift += 0.25 * c

                    if (i == j and i == k):
                        terms.append(((i,), -0.5 * c))
                        shift += 0.5 * c

        return terms, shift

    @staticmethod
    def build_cudaq_hamiltonian(
        z_terms: List[Tuple[Tuple[int, ...], float]],
        shift: float
    ) -> cudaq.SpinOperator:
        H = 0.0
        for idxs, coeff in z_terms:
            if coeff == 0:
                continue
            op = 1.0
            for q in idxs:
                op = op * spin.z(int(q))
            H = H + float(coeff) * op
        return H + float(shift)

    @staticmethod
    def parse_cudaq_bitstring(bitstr: str, reverse: bool = False) -> np.ndarray:
        bits = np.array([int(b) for b in bitstr], dtype=int)
        return bits[::-1] if reverse else bits

    def build_kernel(self):
        N = self.problem_size
        p = self.num_layers

        @cudaq.kernel
        def qaoa_kernel(
            thetas: List[float],
            idx_a: List[int],
            idx_b: List[int],
            idx_c: List[int],
            term_order: List[int],
            term_coeff: List[float],
        ):
            qubits = cudaq.qvector(N)
            h(qubits)

            for layer in range(p):
                gamma = thetas[layer]
                beta = thetas[layer + p]

                for t in range(len(term_coeff)):
                    coeff = term_coeff[t]
                    angle = 2.0 * coeff * gamma

                    if term_order[t] == 1:
                        qa = idx_a[t]
                        rz(angle, qubits[qa])

                    elif term_order[t] == 2:
                        qa = idx_a[t]
                        qb = idx_b[t]
                        x.ctrl(qubits[qa], qubits[qb])
                        rz(angle, qubits[qb])
                        x.ctrl(qubits[qa], qubits[qb])

                    else:
                        qa = idx_a[t]
                        qb = idx_b[t]
                        qc = idx_c[t]
                        x.ctrl(qubits[qa], qubits[qc])
                        x.ctrl(qubits[qb], qubits[qc])
                        rz(angle, qubits[qc])
                        x.ctrl(qubits[qb], qubits[qc])
                        x.ctrl(qubits[qa], qubits[qc])

                for q in range(N):
                    rx(2.0 * beta, qubits[q])

        self.kernel = qaoa_kernel

    def objective(self, params: np.ndarray) -> float:
        assert self.kernel is not None, "Kernel not built. Call build_kernel()."
        assert self.H is not None, "Hamiltonian not built. Call build_problem()."
        return cudaq.observe(
            self.kernel,
            self.H,
            list(params),
            self.idx_a_l, self.idx_b_l, self.idx_c_l,
            self.term_order_l, self.term_coeff_l
        ).expectation()

    def optimize(
        self,
        max_iters: int = 200,
        method: str = "COBYLA",
        xtol: float = 1e-4,
        ftol: float = 1e-4,
        init_params: Optional[np.ndarray] = None,
    ):
        p = self.num_layers
        if init_params is None:
            init_params = np.array([0.01] * p + [0.01] * p, dtype=float)
            #init_params = np.array([np.pi / 4.0] * p + [np.pi / 8.0] * p, dtype=float)

        result = minimize(
            lambda x: self.objective(np.asarray(x, dtype=float)),
            x0=init_params,
            method=method,
            tol = 1e-4,
            options={"maxiter": int(max_iters)},
        )
        return result

    def sample_most_probable(self, opt_params: np.ndarray, shots: int = 10_000) -> Tuple[str, np.ndarray]:
        assert self.kernel is not None, "Kernel not built. Call build_kernel()."
        sr = cudaq.sample(
            self.kernel,
            list(opt_params),
            self.idx_a_l, self.idx_b_l, self.idx_c_l,
            self.term_order_l, self.term_coeff_l,
            shots_count=int(shots)
        )
        bitstr = sr.most_probable()
        x = self.parse_cudaq_bitstring(bitstr, reverse=self.reverse_bitorder)
        return bitstr, x

def solve_HUBO_CUDAQ(
    sub_HUBO: np.ndarray,
    num_layers: int = 2,
    max_iters: int = 200,
    shots: int = 1024,
    target: str = "nvidia",
    reverse_bitorder: bool = False,
) -> np.ndarray:
    n = sub_HUBO.shape[0]

    solver = CudaQQAOAHUBO(
        problem_size=n,
        ctn=0,
        num_layers=num_layers,
        target=target,
        reverse_bitorder=reverse_bitorder,
    )

    solver.z_terms, solver.shift = CudaQQAOAHUBO.hubo_to_z_terms(sub_HUBO)
    solver.H = CudaQQAOAHUBO.build_cudaq_hamiltonian(solver.z_terms, solver.shift)

    idx_a, idx_b, idx_c, term_order, term_coeff = [], [], [], [], []
    for idxs, coeff in solver.z_terms:
        if coeff == 0:
            continue
        if len(idxs) == 1:
            idx_a.append(idxs[0]); idx_b.append(-1); idx_c.append(-1)
            term_order.append(1)
        elif len(idxs) == 2:
            idx_a.append(idxs[0]); idx_b.append(idxs[1]); idx_c.append(-1)
            term_order.append(2)
        else:
            idx_a.append(idxs[0]); idx_b.append(idxs[1]); idx_c.append(idxs[2])
            term_order.append(3)
        term_coeff.append(coeff)

    solver.idx_a_l      = np.asarray(idx_a,      dtype=np.int32).tolist()
    solver.idx_b_l      = np.asarray(idx_b,      dtype=np.int32).tolist()
    solver.idx_c_l      = np.asarray(idx_c,      dtype=np.int32).tolist()
    solver.term_order_l = np.asarray(term_order, dtype=np.int32).tolist()
    solver.term_coeff_l = np.asarray(term_coeff, dtype=np.float64).tolist()

    solver.build_kernel()

    result    = solver.optimize(max_iters=max_iters)
    opt_params = result.x
    bitstr, x = solver.sample_most_probable(opt_params, shots=shots)

    return x

import numpy as np
from cudaq import spin
import networkx as nx

def hubo_to_spin_ham(T):
    n = T.shape[0]

    linear = np.zeros(n)
    quadratic = {}
    cubic_dict = {}

    for i in range(n):
        c = float(T[i, i, i])
        if c != 0.0:
            linear[i] -= c / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            c = float(T[i, i, j])
            if c != 0.0:
                linear[i] -= c / 4.0
                linear[j] -= c / 4.0
                key = (i, j)
                quadratic[key] = quadratic.get(key, 0.0) + c / 4.0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = float(T[i, j, k])
                if c != 0.0:
                    linear[i] -= c / 8.0
                    linear[j] -= c / 8.0
                    linear[k] -= c / 8.0
                    for a, b in [(i, j), (i, k), (j, k)]:
                        quadratic[(a, b)] = quadratic.get((a, b), 0.0) + c / 8.0
                    cubic_dict[(i, j, k)] = cubic_dict.get((i, j, k), 0.0) - c / 8.0

    ham = 0.0
    for i in range(n):
        if abs(linear[i]) > 1e-12:
            ham += linear[i] * spin.z(i)
    for (i, j), c in quadratic.items():
        if abs(c) > 1e-12:
            ham += c * spin.z(i) * spin.z(j)
    for (i, j, k), c in cubic_dict.items():
        if abs(c) > 1e-12:
            ham += c * spin.z(i) * spin.z(j) * spin.z(k)

    return ham

def hubo_to_projected_graph(T):
    n = T.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    edge_weights = {}

    for i in range(n):
        for j in range(i + 1, n):
            c = float(T[i, i, j])
            if c != 0.0:
                edge_weights[(i, j)] = edge_weights.get((i, j), 0.0) + abs(c)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = float(T[i, j, k])
                if c != 0.0:
                    w = abs(c) / 3.0
                    for a, b in [(i, j), (i, k), (j, k)]:
                        edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + w

    for (i, j), w in edge_weights.items():
        G.add_edge(i, j, weight=w)

    return G

def hubo_to_interaction_lists(T, rounding_digits=2):
    n = T.shape[0]
    linear_list, quad_list, cubic_list = [], [], []

    for i in range(n):
        c = round(float(T[i, i, i]), rounding_digits)
        if c != 0.0:
            linear_list.append((i + 1, c))

    for i in range(n):
        for j in range(i + 1, n):
            c = round(float(T[i, i, j]), rounding_digits)
            if c != 0.0:
                quad_list.append((i + 1, j + 1, c))

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = round(float(T[i, j, k]), rounding_digits)
                if c != 0.0:
                    cubic_list.append((i + 1, j + 1, k + 1, c))

    return linear_list, quad_list, cubic_list

def evaluate_hubo_energy(T, x):
    n = T.shape[0]
    x = np.asarray(x, dtype=float)
    energy = 0.0
    for i in range(n):
        energy += float(T[i, i, i]) * x[i]
    for i in range(n):
        for j in range(i + 1, n):
            energy += float(T[i, i, j]) * x[i] * x[j]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                energy += float(T[i, j, k]) * x[i] * x[j] * x[k]
    return energy

def evaluate_ising_energy_from_hubo(T, x):
    n = T.shape[0]
    z = 1.0 - 2.0 * np.asarray(x, dtype=float)
    energy = 0.0

    for i in range(n):
        c = float(T[i, i, i])
        if c != 0.0:
            energy += (-c / 2.0) * z[i]

    for i in range(n):
        for j in range(i + 1, n):
            c = float(T[i, i, j])
            if c != 0.0:
                energy += (-c / 4.0) * z[i]
                energy += (-c / 4.0) * z[j]
                energy += (c / 4.0) * z[i] * z[j]

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = float(T[i, j, k])
                if c != 0.0:
                    energy += (-c / 8.0) * z[i]
                    energy += (-c / 8.0) * z[j]
                    energy += (-c / 8.0) * z[k]
                    energy += (c / 8.0) * z[i] * z[j]
                    energy += (c / 8.0) * z[i] * z[k]
                    energy += (c / 8.0) * z[j] * z[k]
                    energy += (-c / 8.0) * z[i] * z[j] * z[k]

    return energy

def brute_force_hubo(T):
    n = T.shape[0]
    best_hubo = float('inf')
    best_x = None

    for bits in range(2 ** n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        e = evaluate_hubo_energy(T, x)
        if e < best_hubo:
            best_hubo = e
            best_x = x.copy()

    best_ising = evaluate_ising_energy_from_hubo(T, best_x)
    best_str = ''.join(str(int(b)) for b in best_x)
    return best_ising, best_hubo, best_str

def simulated_annealing_hubo(T, initial_temp=100.0, cooling_rate=0.95,
                              iterations=2000, seed=None):
    rng = np.random.default_rng(seed)
    n = T.shape[0]

    x = rng.integers(0, 2, size=n).astype(float)
    cur_e = evaluate_hubo_energy(T, x)
    best_x = x.copy()
    best_e = cur_e
    temp = float(initial_temp)

    for _ in range(iterations):
        idx = int(rng.integers(0, n))
        x_new = x.copy()
        x_new[idx] = 1.0 - x_new[idx]
        new_e = evaluate_hubo_energy(T, x_new)
        delta = new_e - cur_e
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            x = x_new
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_x = x.copy()
        temp *= cooling_rate

    best_ising = evaluate_ising_energy_from_hubo(T, best_x)
    best_str = ''.join(str(int(b)) for b in best_x)
    return best_ising, best_e, best_str

def generate_random_hubo(n_nodes,
                          include_linear=True,
                          include_quadratic=True,
                          include_cubic=True,
                          density=0.5,
                          weight_range=1.0,
                          seed=None):
    rng = np.random.default_rng(seed)
    T = np.zeros((n_nodes, n_nodes, n_nodes))

    if include_linear:
        for i in range(n_nodes):
            if rng.random() < density:
                T[i, i, i] = rng.uniform(-weight_range, weight_range)

    if include_quadratic:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < density:
                    T[i, i, j] = rng.uniform(-weight_range, weight_range)

    if include_cubic:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                for k in range(j + 1, n_nodes):
                    if rng.random() < density:
                        T[i, j, k] = rng.uniform(-weight_range, weight_range)

    return T

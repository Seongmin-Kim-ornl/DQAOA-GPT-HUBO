import cudaq
import numpy as np
from scipy.optimize import minimize
import random
from qaoa_gpt_src.adapt_qaoa_pool import all_pool, qaoa_mixer, qaoa_single_x, qaoa_double
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words

def adapt_qaoa_run(hamiltonian,
                   qubits_num,
                   pool='all_pool',
                   gamma_0=0.01,
                   norm_threshold=1e-3,
                   energy_threshold=1e-5,
                   approx_ratio=1.0,
                   true_energy=0.0,
                   optimizer='BFGS',
                   parameter_shift=False,
                   max_iter=10,
                   seed_adapt=None,
                   verbose=False):

    E_prev = 0.0
    energy_list = []
    pool_list_index = []

    ham_coeffs = term_coefficients(hamiltonian)
    ham_words = term_words(hamiltonian, qubits_num)

    if pool == 'all_pool':
        pools = all_pool(qubits_num)
    elif pool == 'qaoa_mixer':
        pools = qaoa_mixer(qubits_num)
    elif pool == 'qaoa_single_x':
        pools = qaoa_single_x(qubits_num)
    elif pool == 'qaoa_double_ops':
        pools = qaoa_double(qubits_num)
    else:
        raise ValueError(
            "Invalid pool name. Choose from 'all_pool', 'qaoa_mixer', 'qaoa_single_x', or 'qaoa_double'."
        )

    if verbose:
        print(f"Number of hamiltoninian terms: {hamiltonian.term_count}")
        print(f"Pool size: {len(pools)}")

        pool_word = []
        for i in range(len(pools)):
            temp = []
            for term in pools[i]:
                temp.append(term.get_pauli_word(qubits_num))
            pool_word.append(temp)

    com_op = []
    for i in range(len(pools)):
        op = pools[i]
        com_op.append(hamiltonian * op - op * hamiltonian)

    @cudaq.kernel
    def initial_state(qubits_num: int):
        qubits = cudaq.qvector(qubits_num)
        h(qubits)

    state = cudaq.get_state(initial_state, qubits_num)

    @cudaq.kernel
    def grad(state: cudaq.State, ham_words: list[cudaq.pauli_word],
             ham_coeffs: list[complex], gamma_0: float):
        q = cudaq.qvector(state)

        for i in range(len(ham_coeffs)):
            exp_pauli(gamma_0 * ham_coeffs[i].real, q, ham_words[i])

    @cudaq.kernel
    def kernel_qaoa(qubits_num: int, ham_words: list[cudaq.pauli_word],
                    ham_coeffs: list[complex],
                    mixer_pool: list[list[cudaq.pauli_word]],
                    gamma: list[float], beta: list[float], num_layer: int):

        qubits = cudaq.qvector(qubits_num)

        h(qubits)

        for p in range(num_layer):
            for i in range(len(ham_coeffs)):
                exp_pauli(gamma[p] * ham_coeffs[i].real, qubits, ham_words[i])

            for word in mixer_pool[p]:
                exp_pauli(beta[p], qubits, word)

    beta = []
    gamma = []

    mixer_pool = []
    mixer_pool_str = []
    layer = []

    istep = 1

    for iter in range(max_iter):

        if verbose:
            print('Step: ', istep)

        gradient_vec = []
        for op in com_op:
            op = op * -1j
            gradient_vec.append(
                cudaq.observe(grad, op, state, ham_words, ham_coeffs,
                              gamma_0).expectation())

        norm = np.linalg.norm(np.array(gradient_vec))
        if verbose:
            print('Norm of the gradient: ', norm)

        if norm <= norm_threshold:
            if verbose:
                print('\n', 'Final Result: ', '\n')
            if verbose:
                print('Norm of the gradient is below the threshold', norm)
            if verbose:
                print('Final mixer_pool: ', mixer_pool_str)
            if verbose:
                print('Number of layers: ', len(layer))
            if verbose:
                print('Number of mixer pool in each layer: ', layer)
            if verbose:
                print('Final Energy: ', E_current)
            if verbose:
                print('Ratio of the energy: ', ratio)

            break

        else:
            temp_pool = []
            temp_index = []
            tot_pool = 0

            max_grad = np.max(np.abs(gradient_vec))

            for i in range(len(pools)):
                if np.abs(gradient_vec[i]) == max_grad:
                    tot_pool += 1
                    temp_pool.append(pools[i])
                    temp_index.append(i)

            if verbose:
                print('Total number of pool with max gradient: ', tot_pool)
            if seed_adapt is not None:
                random.seed(seed_adapt)

            layer.append(1)
            random_mixer = random.choice(temp_pool)

            for i in range(len(temp_index)):
                if temp_pool[i] == random_mixer:
                    pool_list_index.append(temp_index[i])

            pool_added = []
            pool_added_str = []
            for term in random_mixer:
                pool_added.append(
                    cudaq.pauli_word(term.get_pauli_word(qubits_num)))
                pool_added_str.append(term.get_pauli_word(qubits_num))

            mixer_pool.append(pool_added)
            mixer_pool_str.append(pool_added_str)

            if verbose:
                print('Mixer pool at step', istep)
            if verbose:
                print(mixer_pool_str)

            num_layer = len(layer)
            if verbose:
                print('Number of layers: ', num_layer)

            beta_count = layer[num_layer - 1]
            init_beta = [0.0] * beta_count
            beta = beta + init_beta
            gamma = gamma + [gamma_0]
            theta = gamma + beta

            def cost(theta):

                gamma = theta[:num_layer]
                beta = theta[num_layer:]

                energy = cudaq.observe(kernel_qaoa, hamiltonian, qubits_num,
                                       ham_words, ham_coeffs, mixer_pool, gamma,
                                       beta, num_layer).expectation()
                return energy

            if parameter_shift:

                def parameter_shift(theta):

                    parameter_count = len(theta)
                    epsilon = np.pi / 4
                    grad = np.zeros(parameter_count)
                    theta2 = theta.copy()

                    for i in range(parameter_count):
                        theta2[i] = theta[i] + epsilon
                        exp_val_plus = cost(theta2)
                        theta2[i] = theta[i] - epsilon
                        exp_val_minus = cost(theta2)
                        grad[i] = (exp_val_plus - exp_val_minus) / (2 * epsilon)
                        theta2[i] = theta[i]
                    return grad

            if optimizer == 'COBYLA':
                result_vqe = minimize(cost,
                                      theta,
                                      method='COBYLA',
                                      options={
                                          'rhobeg': 1.0,
                                          'maxiter': 10000,
                                          'disp': False,
                                          'tol': 1e-6
                                      })
                E_current = result_vqe.fun
                theta = result_vqe.x.tolist()
                if verbose:
                    print('Optmized Energy: ', result_vqe.fun, flush=True)
                if verbose:
                    print('Optimizer exited successfully: ',
                          result_vqe.success,
                          flush=True)

            elif optimizer == 'BFGS':
                if parameter_shift:
                    result_vqe = minimize(cost,
                                          theta,
                                          method='BFGS',
                                          jac=parameter_shift,
                                          tol=1e-5)
                    E_current = result_vqe.fun
                    theta = result_vqe.x.tolist()
                    if verbose:
                        print('Optmized Energy: ', result_vqe.fun, flush=True)
                    if verbose:
                        print('Optimizer exited successfully: ',
                              result_vqe.success,
                              flush=True)
                else:
                    result_vqe = minimize(cost,
                                          theta,
                                          method='BFGS',
                                          jac='2-point',
                                          options={'gtol': 1e-4})
                    E_current = result_vqe.fun
                    theta = result_vqe.x.tolist()
                    if verbose:
                        print('Optmized Energy: ', result_vqe.fun, flush=True)
                    if verbose:
                        print('Optimizer exited successfully: ',
                              result_vqe.success,
                              flush=True)

            elif optimizer == 'L-BFGS-B':
                if parameter_shift:
                    result_vqe = minimize(cost,
                                          theta,
                                          method='L-BFGS-B',
                                          jac=parameter_shift,
                                          tol=1e-5)
                    E_current = result_vqe.fun
                    theta = result_vqe.x.tolist()
                    if verbose:
                        print('Optmized Energy: ', result_vqe.fun, flush=True)
                    if verbose:
                        print('Optimizer exited successfully: ',
                              result_vqe.success,
                              flush=True)
                else:
                    result_vqe = minimize(cost,
                                          theta,
                                          method='L-BFGS-B',
                                          jac='2-point',
                                          tol=1e-5)
                    E_current = result_vqe.fun
                    theta = result_vqe.x.tolist()
                    if verbose:
                        print('Optmized Energy: ', result_vqe.fun, flush=True)
                    if verbose:
                        print('Optimizer exited successfully: ',
                              result_vqe.success,
                              flush=True)

            energy_list.append(E_current)

            if verbose:
                print('Result from the step ', istep)
            if verbose:
                print('Optmized Energy: ', result_vqe.fun)

            dE = np.abs(E_current - E_prev)
            E_prev = E_current

            if verbose:
                print('dE= :', dE)

            ratio = E_current / true_energy
            if verbose:
                print('Ratio of the energy: ', ratio)

            gamma = theta[:num_layer]
            beta = theta[num_layer:]

            if dE <= energy_threshold:
                if verbose:
                    print('\n', 'Final Result: ', '\n')
                if verbose:
                    print('dE below the threshold is satisfied: ', dE)
                if verbose:
                    print('Final mixer_pool: ', mixer_pool_str)
                if verbose:
                    print('Number of layers: ', len(layer))
                if verbose:
                    print('Number of mixer pool in each layer: ', layer)
                if verbose:
                    print('Final Energy= ', E_current)
                if verbose:
                    print('Ratio of the energy: ', ratio)

                break

            elif ratio >= approx_ratio:
                if verbose:
                    print('\n', 'Final Result: ', '\n')
                if verbose:
                    print('Approximation ratio is satisfied', ratio)
                if verbose:
                    print('Final mixer_pool: ', mixer_pool_str)
                if verbose:
                    print('Number of layers: ', len(layer))
                if verbose:
                    print('Number of mixer pool in each layer: ', layer)
                if verbose:
                    print('Final Energy= ', E_current)
                if verbose:
                    print('Ratio of the energy: ', ratio)
                break

            else:

                state = cudaq.get_state(kernel_qaoa, qubits_num, ham_words,
                                        ham_coeffs, mixer_pool, gamma, beta,
                                        num_layer)
                if verbose:
                    print('State at step ', istep)
                istep += 1
                if verbose:
                    print('\n')

    if iter == max_iter - 1:
        if verbose:
            print('\n', 'Final Result: ', '\n')
        if verbose:
            print(
                'Maximum number of iterations reached without satisfying the convergence criteria.'
            )
        if verbose:
            print('Final mixer_pool: ', mixer_pool_str)
        if verbose:
            print('Number of layers: ', len(layer))
        if verbose:
            print('Number of mixer pool in each layer: ', layer)
        if verbose:
            print('Final Energy= ', E_current)

    if verbose:
        print('\n', 'Sampling the Final ADAPT QAOA circuit', '\n')
    count = cudaq.sample(kernel_qaoa,
                         qubits_num,
                         ham_words,
                         ham_coeffs,
                         mixer_pool,
                         gamma,
                         beta,
                         num_layer,
                         shots_count=5000)
    if verbose:
        print('The most probable max cut: ', count.most_probable())
    if verbose:
        print('All bitstring from circuit sampling: ', count)

    return (energy_list, mixer_pool_str, pool_list_index, gamma, beta, ratio,
            str(count.most_probable()), len(layer), result_vqe.success)

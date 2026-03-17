from cudaq import spin
import cudaq

def max_cut_ham(edges):
    ham = 0.0

    for edge in range(len(edges)):

        qubitu = edges[edge][0]
        qubitv = edges[edge][1]
        weight = edges[edge][2]
        ham += 0.5 * (weight * spin.z(qubitu) * spin.z(qubitv) -
                      weight * spin.i(qubitu) * spin.i(qubitv))

    return ham

def term_coefficients(ham: cudaq.SpinOperator) -> list[complex]:
    result = []
    for term in ham:
        result.append(term.evaluate_coefficient())
    return result

def term_words(ham: cudaq.SpinOperator, qubits_num) -> list[str]:

    result = []
    for term in ham:
        result.append(term.get_pauli_word(qubits_num))
    return result

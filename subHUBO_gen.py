import numpy as np
import random
import os

def load_QUBO(HUBO_size):
    filename = os.getcwd()  + "/" + str(HUBO_size)+'_QUBO.txt'
    QUBO_file = open(filename, 'r')
    A = QUBO_file.read()
    A = np.asmatrix(A)
    m = int(np.size(A)**(1/2))
    A = np.reshape(A, (m,m))
    Q = A
    Q = np.array(Q)
    return Q

def load_k_third(HUBO_size):
    filename = os.getcwd()  +  "/" + str(HUBO_size)+'_k_third.txt'
    file = open(filename, 'r')
    A = file.read()
    A = np.asmatrix(A)
    A = np.reshape(A, (HUBO_size,8))
    k_third = A
    k_third = np.array(k_third)
    return k_third

def formulate_HUBO(HUBO_size):
    HUBO = np.zeros((HUBO_size,HUBO_size,HUBO_size))
    QUBO_2nd = load_QUBO(HUBO_size)
    k_third = load_k_third(HUBO_size)
    for i in range(HUBO_size):
        HUBO[i,i,i] = QUBO_2nd[i,i]

    for i in range(HUBO_size):
        for j in range(i+1,HUBO_size):
            HUBO[i,i,j] = QUBO_2nd[i,j]

    for i in range(HUBO_size):
        for j in range(i+1,HUBO_size):
            for k in range(j+1,HUBO_size):
                for f in range(8):
                    HUBO[i,j,k]+=k_third[i,f]*k_third[j,f]*k_third[k,f]

    return HUBO

def make_sub_HUBO(HUBO, sub_size):
    N = HUBO.shape[0]
    assert sub_size <= N, "sub_size cannot exceed HUBO size."

    selected_nodes = sorted(random.sample(range(N), sub_size))

    idx = np.ix_(selected_nodes, selected_nodes, selected_nodes)
    sub_HUBO = HUBO[idx]

    return sub_HUBO, selected_nodes

if __name__ == "__main__":
    HUBO_size = 100
    sub_HUBO_size = 4
    num_sub_HUBOs = 50

    for ttn in range(num_sub_HUBOs):
        HUBO = formulate_HUBO(HUBO_size)
        sub_HUBO, selected_nodes = make_sub_HUBO(HUBO, sub_HUBO_size)

        print("HUBO shape:", HUBO.shape)
        print("Generated sub_HUBO:", sub_HUBO)
        print("Selected nodes:", selected_nodes)

        np.save(os.getcwd() + '/subhubo_npy/' +f"subHUBO_size{sub_HUBO_size}_{ttn}.npy", sub_HUBO)

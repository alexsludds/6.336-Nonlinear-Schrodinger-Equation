import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def steady_state():
    delta_x = 0.1
    L = 1
    number_of_nodes = int(L/delta_x)
    V = 10
    sum_mat = hamiltonian(number_of_nodes)  # Construct the Hamiltonian matrix

    w, v = np.linalg.eig(sum_mat)
    print(w)
    v = np.transpose(v)
    small = w.argsort()[:3]
    w = np.sort(w)
    plt.plot(v[small[0]])
    plt.plot(v[small[1]])
    plt.plot(v[small[2]])
    plt.figure()
    plt.plot(w)
    plt.figure()
    plt.plot(w[:10])
    plt.show()


def hamiltonian(n_nodes, v=None):
    """Return a matrix representing the Hamiltonian with first-type boundary conditions at 0"""
    if v is None:
        V_mat = np.zeros((n_nodes, n_nodes))
    else:
        V_mat = np.diag(v)

    kinetic_mat = np.diag(np.ones(n_nodes)*2) - \
                  np.diag(np.ones(n_nodes-1), 1) - \
                  np.diag(np.ones(n_nodes-1), -1)

    return kinetic_mat + V_mat


def schrodinger_f(x, u, p):
    """Function for forward step of Schrodinger equation"""
    return p["A"].dot(x) + p["B"].dot(u)


def forward_euler(f, u, x_start, p, t_start, t_stop, delta_t):
    x = x_start
    t = t_start
    n = 0

    while t < t_stop:
        x = x + delta_t * f(x, u(t), p)
        n += 1
        t = t + delta_t
        plt.plot(x)
        plt.show()
    return x


if __name__ == '__main__':
    steady_state()
    n_nodes = 100
    p = {'A': hamiltonian(n_nodes=100),
         'B': np.zeros((n_nodes, n_nodes))}
    u = lambda x: np.zeros(n_nodes)
    x_start = np.ones(n_nodes)
    t_start = 0.
    t_stop = 10.
    delta_t = 0.1
    x_final = forward_euler(schrodinger_f, u, x_start, p, t_start, t_stop, delta_t)
    plt.plot(x_final)
    plt.show()

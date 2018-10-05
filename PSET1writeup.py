import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def hamiltonian(n_nodes, v=None, dx=1., alpha=1.):
    """Return a matrix representing the Hamiltonian with first-type boundary conditions at 0"""
    if v is None:
        V_mat = np.zeros((n_nodes, n_nodes))
    else:
        V_mat = np.diag(v)

    # Matrix of 2nd derivatives
    partial2_mat = np.diag(np.ones(n_nodes)*-2) + \
                  np.diag(np.ones(n_nodes-1), 1) + \
                  np.diag(np.ones(n_nodes-1), -1)
    # partial2_mat[0,:4] = np.array([-20, 6, 4, -1])/12.
    # partial2_mat[-1, -4:] = np.array([-1, 4, 6, -20])/12.
    partial2_mat *= alpha/dx**2
    kinetic_mat = -partial2_mat
    # print(kinetic_mat)
    return 1j*(kinetic_mat + V_mat)


def dxdt_f(x, u, p):
    """Function for forward step of a dynamical system with dx/dt = A*x(t) + B*u(t)
    Parameters
        x : unknown variable
        u : input
        p : dict containing "A" (nodal analysis matrix) and "B" (input matrix)
    """
    return p["A"].dot(x) + p["B"].dot(u)


def forward_euler(f, u, x_start, p, t_start, t_stop, delta_t):
    x = x_start
    t = t_start
    n = 0
    x_arr = []

    while t < t_stop:
        x = x + delta_t * f(x, u(t), p)
        # TODO: Normalize
        n += 1
        t = t + delta_t
        x_arr.append(x)

    return x, x_arr


if __name__ == '__main__':
    # steady_state()
    n_nodes = 10   # Number of nodes to solve for. Total number of nodes = n_nodes+2 because of boundary nodes
    frame_interval = 5  # Interval of frame to print (i.e., print every 5th frame)
    p = {'A': hamiltonian(n_nodes=n_nodes, dx=1/(n_nodes+2), alpha=1e-2),
         'B': np.zeros((n_nodes, n_nodes))}
    u = lambda _: np.zeros(n_nodes)
    x = np.linspace(0, 1, n_nodes+2)[1:-1]  # Note this x is spatial position, not solution
    # Initial state is combination of 1st and 2nd mode
    stationary_state_1 = np.sqrt(2)*np.sin(np.pi*x)
    stationary_state_2 = np.sqrt(2) * np.sin(2*np.pi*x)
    # psi_start = 1/np.sqrt(2)*stationary_state_1 + 1/np.sqrt(2)*stationary_state_2
    psi_start = stationary_state_1
    t_start = 0.
    t_stop = 50.
    delta_t = 0.01
    x_final, x_arr = forward_euler(dxdt_f, u, psi_start, p, t_start, t_stop, delta_t)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
    l1, = plt.plot([], [], color='b')
    l2, = plt.plot([], [], color='g')

    def init():
        l1.set_data([], [])
        l2.set_data([], [])

    def animate(i):
        l1.set_data(x, np.real(x_arr[i*frame_interval]))
        l2.set_data(x, np.imag(x_arr[i*frame_interval]))
        return [l1, l2]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=int(t_stop/delta_t/frame_interval), interval=1)

    plt.show()


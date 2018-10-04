import numpy as np
import scipy as sp

def steady_state():
    delta_x = 0.1
    L = 1
    number_of_nodes = int(L/delta_x)
    V = 10
    con_mat = np.zeros((number_of_nodes,number_of_nodes))
    boundary_mat = np.zeros((number_of_nodes,number_of_nodes))
    for i in range(number_of_nodes):
        con_mat[i,i] = -2
        try:
            con_mat[i,i-1] = 1
        except:
            pass
        try:
            con_mat[i,i+1] = 1
        except:
            pass

        boundary_mat[i,i] = -V

    boundary_mat *= delta_x**2
    print(np.linalg.eig(con_mat)[0])
    eigenvalues = np.linalg.eig(boundary_mat - con_mat)
    print(eigenvalues[0])

def forward_euler(f,u,x_start,p,t_start,t_stop,delta_t):
    x = x_start
    t = t_start
    n = 0
    while t < t_stop:
        x = x + delta_t * f(x,u(t),p)
        n += 1
        t = t + delta_t
    return x

if __name__ == '__main__':
    steady_state()

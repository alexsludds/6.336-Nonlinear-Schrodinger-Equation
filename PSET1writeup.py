import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def steady_state():
    delta_x = 0.1
    L = 1
    number_of_nodes = int(L/delta_x)
    V = 10
    con_mat = np.zeros((number_of_nodes,number_of_nodes))
    boundary_mat = np.zeros((number_of_nodes,number_of_nodes))
    # for i in range(number_of_nodes):
    #     con_mat[i,i] = 2
    #     try:
    #         if(i!=0):
    #             con_mat[i,i-1] = -1
    #     except:
    #         pass
    #     try:
    #         con_mat[i,i+1] = -1
    #     except:
    #         pass
    con_mat = np.diag(np.ones(number_of_nodes)*2) - \
              np.diag(np.ones(number_of_nodes-1),1) - \
              np.diag(np.ones(number_of_nodes-1),-1)

    sum_mat = con_mat + boundary_mat
    w, v = np.linalg.eig(sum_mat)
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
    return sum_mat


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
    # forward_euler()

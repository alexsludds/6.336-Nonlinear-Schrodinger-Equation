import numpy as np
import scipy as sp
import scipy.constants
import matplotlib
import matplotlib.pyplot as plt
import problems
from main import Simulation
# from scipy.optimize import newton
from scipy import optimize


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def shooting_function(x):
    init_state = x
    xf, _ = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
                      delta_t=delta_t, animation_timestep=1e10)
    return xf - x


def find_stationary_state(problem, x0):
    xf = optimize.root(problem.calc_F_stationary, x0=x0, jac=problem.calc_F_stationary_Jacobian)
    return xf


def newton(fun, x0, jac):
    tol = 1e-4
    max_iter = 500
    for i in range(max_iter):
        f_i = fun(x0)
        j_i = jac(x0)
        dx = -np.linalg.solve(j_i, f_i)
        x0 = x0 + dx

        if np.amax(np.abs(f_i)) < tol and np.amax(np.abs(dx)) < tol:
            print("Converged in %d iterations" % i)
            break
    return x0


if __name__ == "__main__":
    number_of_psi = 1000  # This is the total number of nodes.
    # We are solving for number_of_psi-2 because of boundary conditions
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = -100
    stop_x = 100
    periodic_boundary_conditions = True
    gif_name = "test"
    time_start = 0
    time_stop = 10
    delta_t = 1e-5

    sim = Simulation(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
                     number_of_spatial_dimensions=number_of_spatial_dimensions, nonlinear=True)

    NLSE = problems.NLSE(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi, periodic=True)

    init_state = NLSE.get_stationary_state()
    init_state = 0.7*np.exp(-NLSE.linspace**2/2)

    # xf = find_stationary_state(NLSE, x0=init_state)
    # xf = xf.x
    # np.savetxt('xf', xf.x)

    # xf = np.loadtxt('xf')

    xf2 = newton(NLSE.calc_F_stationary, x0=init_state, jac=NLSE.calc_F_stationary_Jacobian)

    plt.figure()
    plt.plot(NLSE.linspace, init_state, label='Init guess')
    plt.plot(NLSE.linspace, NLSE.soliton(NLSE.linspace), label='Analytical')
    # plt.plot(NLSE.linspace, xf, linestyle=':', label='Scipy root', markersize=20, linewidth=2)
    plt.plot(NLSE.linspace, xf2, linestyle='-.', label='Newton', markersize=20, linewidth=2)
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()

    # x_final, x_arr = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
    #                                    delta_t=delta_t, animation_timestep=animation_timestep)

    # x_final, x_arr = sim.trapezoidal(sim.dxdt_f_trapezoid, u, init_state, p, t_start=time_start, t_stop=time_stop,
    #                                  delta_t=delta_t, animation_timestep=animation_timestep)


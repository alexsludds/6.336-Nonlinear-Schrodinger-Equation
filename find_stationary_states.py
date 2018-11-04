import numpy as np
import scipy as sp
import scipy.constants
import matplotlib
import matplotlib.pyplot as plt
import problems
from main import Simulation
from scipy.optimize import newton
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


def find_stationary_state(problem):
    xf = optimize.root(problem.calc_F_stationary, problem.soliton(problem.linspace),
                jac=problem.calc_F_stationary_Jacobian)
    return xf


if __name__ == "__main__":
    number_of_psi = 1000  # This is the total number of nodes.
    # We are solving for number_of_psi-2 because of boundary conditions
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = -100
    stop_x = 100
    # This constant allows for there to be runtime_in_seconds / animation_constant number of frames output animation.
    # Higher constant = faster animation.
    plot_stationary_solution = False
    periodic_boundary_conditions = True
    gif_name = "test"
    time_start = 0
    time_stop = 10
    delta_t = 1e-5

    sim = Simulation(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
                     number_of_spatial_dimensions=number_of_spatial_dimensions, nonlinear=True)

    # quantum = problems.Quantum(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
    #                            number_of_spatial_dimensions=number_of_spatial_dimensions,
    #                            periodic=periodic_boundary_conditions)
    # hamiltonian = quantum.calc_A()
    #
    # # plot the stationary solution
    # if plot_stationary_solution:
    #     eigenvalues, eigenvectors = sim.find_eigenvalues(hamiltonian/1j)
    #     print(eigenvalues)
    #     sim.plot_stationary(eigenvectors[mode-1])
    #
    # # Initial state is one of the stationary states
    # init_state = quantum.get_stationary_state(mode)
    # # If we want to put something else inside of the line and see how it evolves we can do it here
    # init_state = signal.gaussian(number_of_psi - 2, std=1)

    NLSE = problems.NLSE(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi, periodic=True)

    init_state = NLSE.get_stationary_state()
    # plt.figure()
    # plt.plot(init_state)
    # plt.show()

    u = NLSE.get_u()
    p = NLSE.get_P()

    # print(NLSE.calc_F_stationary(x=init_state, theta=1))
    # print(init_state)
    # print(NLSE.calc_A(x=init_state))

    xf = find_stationary_state(NLSE)
    xf = xf.x
    # np.savetxt('xf', xf.x)

    # xf = np.loadtxt('xf')

    plt.figure()
    plt.plot(xf, label='Root')
    plt.plot(NLSE.soliton(NLSE.linspace), label='Analytical')
    plt.legend()
    plt.show()

    # x_final, x_arr = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
    #                                    delta_t=delta_t, animation_timestep=animation_timestep)

    # x_final, x_arr = sim.trapezoidal(sim.dxdt_f_trapezoid, u, init_state, p, t_start=time_start, t_stop=time_stop,
    #                                  delta_t=delta_t, animation_timestep=animation_timestep)


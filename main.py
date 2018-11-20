import numpy as np
import scipy as sp
from scipy.misc import imread
import scipy.constants
import matplotlib.pyplot as plt
# We are using the qt backend because tkinter backend does not allow for gif creation without an open window
# matplotlib.use("Qt4Agg")
from progress.bar import Bar
import problems
from animation import AnimationClass
from simulation import Simulation

def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


if __name__ == "__main__":
    number_of_psi = 100  # This is the total number of nodes.
    # We are solving for number_of_psi-2 because of boundary conditions
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = -10
    stop_x = 10
    hbar = sp.constants.h / (2*sp.pi)   # Reduced Planck constant = 1.055e-34 J s/rad
    # This constant allows for there to be runtime_in_seconds / animation_constant number of frames output animation.
    # Lower constant = faster animation.
    animation_interval = 0.01  # Animation constant associated with coarseness.
    animation_speed = 1./2  # Animation speed multiplier
    display_animation = True
    plot_stationary_solution = False
    periodic_boundary_conditions = False
    gif_name = "trap_with_disturbance"
    time_start = 0
    time_stop = 5
    delta_t = 2e-3

    animation_timestep = int(animation_interval / delta_t)

    sim = Simulation(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
                     number_of_spatial_dimensions=number_of_spatial_dimensions, nonlinear=True)

    quantum = problems.Quantum(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
                               number_of_spatial_dimensions=number_of_spatial_dimensions,
                               periodic=periodic_boundary_conditions)

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

    NLSE = problems.NLSE(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi)

    init_state = NLSE.get_stationary_state()

    # plt.plot(init_state)

    # init_state = 0.5*np.exp(-(NLSE.linspace**2)/2)

    # plt.plot(init_state)
    # plt.show()
    #
    # quit()

    u = NLSE.get_u()
    p = NLSE.get_P()


    # x_final, x_arr = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                       # delta_t=delta_t, animation_timestep=animation_timestep)

    # x_final, x_arr = sim.trapezoidal(sim.dxdt_f_trapezoid, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                     # delta_t=delta_t, animation_timestep=animation_timestep)

    # x_final, x_arr = sim.trapezoidal_nl(sim.dxdt_f_trapezoid, u, init_state, p, t_start = time_start, t_stop = time_stop,
                                        # delta_t=delta_t, animation_timestep=animation_timestep)

    x_final, x_arr = sim.trapezoidal_nl_iterative(sim.dxdt_f,u,init_state,p,t_start = time_start, t_stop = time_stop,
                                                  delta_t = delta_t, animation_timestep = animation_timestep)

    # Display animation
    if display_animation:
        ani = AnimationClass(animation_interval=animation_timestep,
                             x=np.linspace(start_x, stop_x, number_of_psi),
                             x_arr=x_arr, runtime_seconds=time_stop, delta_t=delta_t,
                             gif_name=gif_name, speed=animation_speed)
        ani.run_animation_with_propagation()
        ani.create_gif()
        ani.create_video()

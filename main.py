import numpy as np
import scipy as sp
from scipy.misc import imread
import scipy.constants
import matplotlib.pyplot as plt
# We are using the qt backend because tkinter backend does not allow for gif creation without an open window
from progress.bar import Bar
import problems
from animation import AnimationClass
from simulation import Simulation


def main(display_animation=True):
    number_of_psi = 100  # This is the total number of nodes.
    # We are solving for number_of_psi-2 because of boundary conditions
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = -10
    stop_x = 10
    hbar = sp.constants.h / (2*sp.pi)   # Reduced Planck constant = 1.055e-34 J s/rad
    # This constant allows for there to be runtime_in_seconds / animation_constant number of frames output animation.
    # Lower constant = faster animation.
    animation_interval = 0.1  # Animation constant associated with coarseness.
    animation_speed = 10.  # Animation speed multiplier
    plot_stationary_solution = False
    periodic_boundary_conditions = False
    gif_name = "power_plot"
    time_start = 0
    time_stop = 10*np.pi
    delta_t = 1e-2

    animation_timestep = int(animation_interval / delta_t)

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

    #This is where Gamma is defined. For our purposes Gamma is a list in time. If you wish for a constant Gamma create a list of constant values. Note that the standard value we have been using for gamma is -2
    num_timesteps = (time_stop-time_start)/delta_t + 1
    num_timesteps = int(num_timesteps)
    gamma = np.ones(num_timesteps)
    gamma = gamma * (-2)
    # gamma[int(0.05*num_timesteps):int(0.10*num_timesteps)] *= 1.25

    # gamma[int(0.15*num_timesteps):int(0.20*num_timesteps)] *= 1.25
    # gamma[int(0.25*num_timesteps):int(0.30*num_timesteps)] *= 1.25
    # gamma[int(0.35*num_timesteps):int(0.40*num_timesteps)] *= 1.25
    # gamma[int(0.45*num_timesteps):int(0.50*num_timesteps)] *= 1.25
    # gamma[int(0.55*num_timesteps):int(0.60*num_timesteps)] *= 1.25
    # gamma[int(0.65*num_timesteps):int(0.70*num_timesteps)] *= 1.25
    # gamma[int(0.75*num_timesteps):int(0.80*num_timesteps)] *= 1.25
    # gamma[int(0.85*num_timesteps):int(0.90*num_timesteps)] *= 1.25
    # gamma[int(0.95*num_timesteps):int(1.00*num_timesteps)] *= 1.25
    # #gamma = -2

    NLSE = problems.NLSE(gamma = gamma,x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi)

    init_state = NLSE.get_stationary_state()
    # init_state = np.loadtxt('xf_shooting2.txt').view(complex)

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

    # x_final, x_arr = sim.trapezoidal_nl(sim.dxdt_f_trapezoid, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                        # delta_t=delta_t, animation_timestep=animation_timestep)

    x_final, x_arr = sim.trapezoidal_nl_iterative(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                                  delta_t=delta_t, animation_timestep=animation_timestep, NLSE=NLSE)

    # print(x_arr[0].shape)
    x_arr = [np.abs(i)**2 for i in x_arr ]
    # Display animation
    if display_animation:
        ani = AnimationClass(animation_interval=animation_timestep,
                             x=np.linspace(start_x, stop_x, number_of_psi),
                             x_arr=x_arr, gamma = gamma ,runtime_seconds=time_stop, delta_t=delta_t,
                             gif_name=gif_name, speed=animation_speed)
        ani.run_animation_with_propagation()
        ani.create_gif()
        ani.create_video()


if __name__ == "__main__":
    main()

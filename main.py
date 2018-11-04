import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.misc import imread
import scipy.constants
import matplotlib
# We are using the qt backend because tkinter backend does not allow for gif creation without an open window
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
# from array2gif import write_gif
import matplotlib.animation as animation
from progress.bar import Bar
import os, sys, time
from benchmark import benchmark
import problems


class Simulation:
    def __init__(self, x_start=0, x_stop=1, number_of_psi=100, number_of_spatial_dimensions=1,
                 nonlinear=False):
        self.number_of_psi = number_of_psi
        self.number_of_spatial_dimensions = number_of_spatial_dimensions
        self.constituent_matrix = None
        self.potential_matrix = None
        self.x_start = x_start
        self.x_stop = x_stop
        self.linspace = np.linspace(self.x_start, self.x_stop, num=self.number_of_psi)[1:-1]
        self.dx = (self.x_stop - self.x_start)/(self.number_of_psi-1)
        self.nonlinear = nonlinear

    """
    Given the Hermitian matrix return the eigenvalues and eigenvectors
    """
    @benchmark
    def find_eigenvalues(self, H_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(H_matrix)
        # sort eigenvectors and eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors.T  # possibly the most cursed thing in numpy
        return eigenvalues, eigenvectors

    """
    Performs the forward_euler method. We use this as a way of advancing time on the wave-form
    """
    @benchmark
    # Forward euler is faster, but unstable
    def forward_euler(self, f, u, x_start, p, t_start, t_stop, delta_t, animation_timestep):
        x = x_start
        t = t_start
        n = 0
        x_arr = []
        bar = Bar('Processing', max=int((t_stop-t_start)/delta_t))
        # t_stop = 2*delta_t
        # plt.figure()
        # plt.plot(np.real(x))
        # plt.plot(np.imag(x))
        while t < t_stop:
            x = x + delta_t * f(x, u(t), p)
            n += 1
            t = t + delta_t
            # plt.figure()
            # plt.plot(np.real(x))
            # plt.plot(np.imag(x))
            # plt.show()
            # normalize the wave function for the quantum problem
            if not self.nonlinear:
                norm = np.linalg.norm(x)
                x = x/norm
            # Save only usable frames
            if n % animation_timestep == 0:
                x_arr.append(x)
            bar.next()
        bar.finish()
        return x, x_arr

    @benchmark
    # Trapezoidal rule is slower, but stable
    def trapezoidal(self, f, u, x_start, p, t_start, t_stop, delta_t, animation_timestep):
        x = x_start
        t = t_start
        x_arr = []
        n = 0
        # TODO This is the slow version of the trapezoidal rule, we should get around to implementing the iterative
        # version at some point
        inverse = np.linalg.inv(np.eye(x.shape[0],x.shape[0])-delta_t/2. * p['A'])
        bar = Bar("Processing", max=int((t_stop-t_start)/delta_t), suffix='%(percent).1f%% - %(eta)ds')
        while t < t_stop:
            if t == 0:
                x = f(x, u(0), u(0), p, delta_t,inverse)
            else:
                x = x + f(x, u(t-1), u(t), p, delta_t, inverse)
            t = t + delta_t
            n += 1
            # normalize
            if not self.nonlinear:
                norm = np.linalg.norm(x)
                x = x/norm
            # Save only usable frames
            if n % animation_timestep == 0:
                x_arr.append(x)
            bar.next()
        bar.finish()
        return x, x_arr

    @benchmark
    def plot_stationary(self, eigenvector):
        plt.plot(self.linspace, eigenvector)
        plt.show()

    # Don't benchmark this function, it is fast and gets called A LOT by forward euler
    def dxdt_f(self, x, u, p):
        if not self.nonlinear:
            return p["A"].dot(x) + p["B"].dot(u)
        else:
            return p["A"](x).dot(x) + p["B"].dot(u)

    def dxdt_f_trapezoid(self,x,u_previous,u_current,p,delta_t,inverse):
        RHS = x + delta_t/2.*p['B'].dot(u_previous) + delta_t/2.*p['B'].dot(u_current) + delta_t/2.*p['A'].dot(x)
        return np.dot(inverse,RHS)


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


class AnimationClass:
    def __init__(self, animation_interval, x, x_arr, runtime_seconds=10, delta_t=0.005,
                 gif_name="test.gif"):
        self.fig = plt.figure(figsize=(8, 6), dpi=200)
        self.ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
        self.l1, = plt.plot([], [], color='b')
        self.l2, = plt.plot([], [], color='g')
        self.x = x
        self.gif_name = gif_name
        self.anim = None
        self.animation_interval = animation_interval

        # Append to x_arr such that we have the boundary conditions
        self.x_arr = x_arr
        self.include_boundary_conditions()

        # self.time_step = int(len(self.x_arr)/(self.fps*self.runtime_seconds))
        # TODO: Clean up all this mess regarding frames and playback speed
        self.n_frames = int(runtime_seconds/delta_t/animation_interval)

    def include_boundary_conditions(self):
        # TODO Add ability to have boundary condition other than just zero
        x_arr_temp = list(map(lambda x: np.append(x, 0), self.x_arr))
        self.x_arr = list(map(lambda x: np.insert(x, 0, 0), x_arr_temp))
        return self.x_arr

    def animate(self, i):
        self.l1.set_data(self.x, np.real(self.x_arr[i]))
        self.l2.set_data(self.x, np.imag(self.x_arr[i]))
        return [self.l1, self.l2]

    def animate_with_velocity(self, i):
        # We want to get the self.x_arr data and create a version which shifts over time by velocity at each timestep
        # First to do this we must get self.x and extend it. We know that the spacing in self.x is linear so we can find
        # the spacing by doing (last-first)/num_samples
        spacing = (self.x[-1]-self.x[0])/self.x.shape[0]
        # We update this by using an array of size   self.x.shape[0] + number_of_time_steps*velocity
        number_of_elements = self.x.shape[0] + len(self.x_arr)*self.velocity
        extended_x = np.linspace(start=self.x[0], stop=self.x[0] + spacing * number_of_elements ,num= number_of_elements)
        # We want to get x_arr and append time_step * velocity  zeros to the beginning of each sample
        new_x_arr = np.zeros(number_of_elements, dtype=np.complex128)
        new_x_arr[i*self.velocity: i*self.velocity + len(self.x_arr[i])] = self.x_arr[i]
        # We want to update self.ax such that it now has new x-limits
        self.ax.set_xlim((0, spacing * number_of_elements))
        self.l1.set_data(extended_x, np.real(new_x_arr))
        self.l2.set_data(extended_x, np.imag(new_x_arr))
        return [self.l1, self.l2]

    def initialize(self):
        self.l1.set_data([], [])
        self.l2.set_data([], [])

    def run_animation(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.initialize,
                                            frames=self.n_frames, interval=self.animation_interval)
        plt.show()

    def run_animation_with_propagation(self, velocity=1):
        self.velocity = velocity
        self.anim = animation.FuncAnimation(self.fig, self.animate_with_velocity, init_func=self.initialize,
                                            frames=self.n_frames, interval=self.animation_interval)
        spacing = (self.x[-1]-self.x[0])/self.x.shape[0]
        number_of_elements = self.x.shape[0] + len(self.x_arr)*self.velocity
        right_side_boundary = self.x[0] + spacing * number_of_elements
        plt.imshow(imread("fiber_optic.png"), zorder=0, extent=[0, right_side_boundary, -1, 1])
        plt.show()

    def create_gif(self):
        create_gif_input = input("Do you want to create a gif of the animation? (y/n)")
        if create_gif_input == "n":
            return
        print("Creating gif")
        if self.anim is None:
            print("Please run animate in order to create the animation object for gif creation")
            return
        else:
            self.anim.save("multimedia/" + str(self.gif_name) + ".gif", dpi=200, writer='ffmpeg', codec="libx265")
        print("Gif created with name: ", self.gif_name)

    def create_video(self):
        create_video_input = input("Do you want to create a video of the animation? (y/n)")
        if create_video_input == "n":
            return
        print("Creating video")
        if self.anim is None:
            print("Please run animate in order to create the animation object for video creation")
            return
        else:
            # TODO: Fix fps of the exported video to match that of the animation
            self.anim.save("multimedia/" + str(self.gif_name) + ".mp4", dpi=200, writer="ffmpeg", codec="libx265")
        print("Video created with name: ", self.gif_name)


if __name__ == "__main__":
    number_of_psi = 100  # This is the total number of nodes.
    # We are solving for number_of_psi-2 because of boundary conditions
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = -10
    stop_x = 10
    hbar = sp.constants.h / (2*sp.pi)   # Reduced Planck constant = 1.055e-34 J s/rad
    # This constant allows for there to be runtime_in_seconds / animation_constant number of frames output animation.
    # Higher constant = faster animation.
    animation_constant = 0.005
    animation_constant = 0.05
    display_animation = True
    plot_stationary_solution = False
    periodic_boundary_conditions = True
    gif_name = "test"
    time_start = 0
    time_stop = 10
    delta_t = 1e-5

    animation_timestep = int(animation_constant / delta_t)

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

    NLSE = problems.NLSE(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi)

    init_state = NLSE.get_stationary_state()
    # plt.figure()
    # plt.plot(init_state)
    # plt.show()

    u = NLSE.get_u()
    p = NLSE.get_P()

    x_final, x_arr = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                       delta_t=delta_t, animation_timestep=animation_timestep)

    # x_final, x_arr = sim.trapezoidal(sim.dxdt_f_trapezoid, u, init_state, p, t_start=time_start, t_stop=time_stop,
    #                                  delta_t=delta_t, animation_timestep=animation_timestep)

    # Display animation
    if display_animation:
        ani = AnimationClass(animation_interval=animation_timestep,
                             x=np.linspace(start_x, stop_x, number_of_psi),
                             x_arr=x_arr, runtime_seconds=time_stop, delta_t=delta_t,
                             gif_name=gif_name)
        ani.run_animation_with_propagation()
        ani.create_gif()
        ani.create_video()

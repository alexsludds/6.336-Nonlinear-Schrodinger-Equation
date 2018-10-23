import numpy as np
import scipy as sp
import matplotlib
matplotlib.use("Qt4Agg") #We are using the qt backend because tkiner backend does not allow for gif creation without an open window
import matplotlib.pyplot as plt
from array2gif import write_gif
import matplotlib.animation as animation
from progress.bar import Bar
import os,sys


class Simulation:
    def __init__(self, x_start=0, x_stop=1, number_of_psi=100, number_of_spatial_dimensions=1,
                 potential_function=lambda x: 0, non_linear=False, alpha=1e-12):
        self.number_of_psi = number_of_psi
        self.number_of_spatial_dimensions = number_of_spatial_dimensions
        self.constituent_matrix = None
        self.potential_matrix = None
        self.x_start = x_start
        self.x_stop = x_stop
        self.potential_function = potential_function    # Default is infinite square well
        self.linspace = np.linspace(self.x_start, self.x_stop, num=self.number_of_psi)[1:-1]
        self.alpha = alpha
        self.dx = (self.x_stop - self.x_start)/(self.number_of_psi-1)
        self.non_linear = non_linear

    """
    Returns a copy of matrix describing relationships between Psi's
    """
    def generate_constituent_matrix(self):
        # set off diagonal terms to -1
        D = -1*np.diag(np.ones(self.number_of_psi-3), 1) - np.diag(np.ones(self.number_of_psi-3), -1)
        # set diagonal to 2
        np.fill_diagonal(D, 2)
        if self.number_of_spatial_dimensions == 1:
            self.constituent_matrix = D
            return D

        elif self.number_of_spatial_dimensions == 2:
            # TODO: Clean this up and consider boundary conditions
            # This basically just generates a block matrix of the 2D case.
            # Best way to see this work is to swap all of the toblock.append(D) to something like toblock.append("D")
            # then print Q at the end
            D = -1*np.diag(np.ones(self.number_of_psi-1), 1) - 1* np.diag(np.ones(self.number_of_psi-1),-1)
            np.fill_diagonal(D,4)
            I = np.eye(self.number_of_psi)
            Z = np.zeros((self.number_of_psi,self.number_of_psi))
            toblock = []
            toblockarray = []
            for row in range(self.number_of_psi):
                toblock = []
                for col in range(self.number_of_psi):
                    if row == col:
                        toblock.append(D)
                    elif row == col-1:
                        toblock.append(I)
                    elif row == col+1:
                        toblock.append(I)
                    else:
                        toblock.append(Z)
                toblockarray.append(toblock)
            Q = np.block(toblockarray)
            self.constituent_matrix = Q
            return Q

        elif self.number_of_spatial_dimensions == 3:
            pass

    """
    Given the values of x_start, x_stop, and the potential function that is defined
    """
    def generate_potential_matrix(self):
        potential_values = np.array(list(map(self.potential_function, self.linspace)))
        diagonal_matrix = np.diag(potential_values)
        self.potential_matrix = diagonal_matrix
        return diagonal_matrix

    """
    Given the Hermitian matrix return the eigenvalues and eigenvectors
    """
    def find_eigenvalues(self, H_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(H_matrix)
        # eigenvectors = eigenvectors.T #possibly the more cursed thing numpy ever implemented
        #sort eigenvectors and eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = -1*eigenvectors.T  # possibly the most cursed thing in numpy
        return eigenvalues, eigenvectors

    """
    Calculates the Hamiltonian of the system, which is i*(constituent matrix + potential matrix)
    """
    def hamiltonian(self):
        hamiltonian = 1j*(self.constituent_matrix + self.potential_matrix)
        return hamiltonian

    """
    Performs the forward_euler method. We use this as a way of advancing time on the wave-form
    """
    def forward_euler(self, f, u, x_start, p, t_start, t_stop, delta_t):
        x = x_start
        t = t_start
        n = 0
        x_arr = []
        bar = Bar('Processing', max=int((t_stop-t_start)/delta_t))
        while t < t_stop:
            x = x + delta_t * f(x, u(t),p)
            n += 1
            t = t + delta_t
            # normalize
            if not self.non_linear:
                norm = np.linalg.norm(x)
                x = x/norm
            x_arr.append(x)
            bar.next()
        bar.finish()
        return x, x_arr

    def plot_stationary(self, eigenvector):
        plt.plot(self.linspace, eigenvector)
        plt.show()

    def dxdt_f(self, x, u, p):
        return p["A"].dot(x) + p["B"].dot(u)

    def nonlinear_matrix(self, x):
        D = self.hamiltonian
        D = D + np.diag(np.square(np.abs(x)))
        return D

    def calc_jacobian_numerical(self, f, x, u, p, epsilon):
        """Return the Jacobian calculated using finite-difference
        The Jacobian is size (n, 2n) where n is size of x because x is complex"""
        jacobian = np.zeros((self.number_of_psi-2, 2*self.number_of_psi), dtype=complex)
        f0 = f(x, u, p)
        # print(f0)
        for i in range(len(x)):
            delta_x = np.zeros(self.number_of_psi-2)
            delta_x[i] = epsilon
            j_i_real = (f(x + delta_x, u, p) - f0)/epsilon
            j_i_imag = (f(x + 1.0j*delta_x, u, p) - f0)/(epsilon*1.0j)
            jacobian[:, 2*i] = j_i_real
            jacobian[:, 2*i+1] = j_i_imag
        return jacobian


class AnimationClass:
    def __init__(self, fps, x, x_arr, runtime_seconds=10, delta_t=0.005, playback_speed=1,
                 gif_name = "test.gif"):
        self.fig = plt.figure(figsize=(8,6),dpi=200)
        self.ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
        self.l1, = plt.plot([], [], color='b')
        self.l2, = plt.plot([], [], color='g')
        self.fps = fps
        self.x = x
        self.gif_name = gif_name
        self.anim = None

        # Append to x_arr such that we have the boundary conditions
        self.x_arr = x_arr
        self.include_boundary_conditions()

        # self.time_step = int(len(self.x_arr)/(self.fps*self.runtime_seconds))
        # TODO: Clean up all this mess regarding frames and playback speed
        self.frame_step = int(playback_speed / delta_t / fps)
        self.n_frames = int(runtime_seconds*self.fps/playback_speed)

    def include_boundary_conditions(self):
        # TODO Add ability to have boundary condition other than just zero
        x_arr_temp = list(map(lambda x: np.append(x, 0), self.x_arr))
        self.x_arr = list(map(lambda x: np.insert(x, 0, 0), x_arr_temp))
        return self.x_arr

    def animate(self, i):
        self.l1.set_data(self.x, np.real(self.x_arr[self.frame_step*i]))
        self.l2.set_data(self.x, np.imag(self.x_arr[self.frame_step*i]))
        return [self.l1, self.l2]

    def initialize(self):
        self.l1.set_data([], [])
        self.l2.set_data([], [])

    def run_animation(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.initialize,
                                       frames=self.n_frames, interval=int(1000/self.fps))
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
        print("Gif created with name: ",self.gif_name)

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
            self.anim.save("multimedia/" + str(self.gif_name) + ".mp4", dpi=200, writer="ffmpeg", codec = "libx265")
        print("Video created with name: ", self.gif_name)


if __name__ == "__main__":
    number_of_psi = 10  # This is the total number of nodes. We are solving for number of nodes -2
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = 0
    stop_x = 1
    hbar = 1.0545718E-34
    gif_animation_frames_per_second = 100
    display_animation = False
    plot_stationary_solution = False
    gif_name = "test"
    time_start = 0
    time_stop = 100
    delta_t = 0.005

    sim = Simulation(x_start=start_x, x_stop=stop_x, number_of_psi=number_of_psi,
                     number_of_spatial_dimensions=number_of_spatial_dimensions)
    constituent_matrix = sim.generate_constituent_matrix()
    potential_matrix = sim.generate_potential_matrix()
    hamiltonian = sim.hamiltonian()
    # hamiltonian = constituent_matrix + potential_matrix
    eigenvalues, eigenvectors = sim.find_eigenvalues(hamiltonian)

    # plot the stationary solution
    if plot_stationary_solution:
        sim.plot_stationary(eigenvectors[mode-1])


    def u(_):
        return np.zeros(number_of_psi - 2)


    # Initial state is one of the stationary states
    init_state = np.sqrt(2) * np.sin(mode * np.pi * np.linspace(start_x, stop_x, number_of_psi)[1:-1])
    p = {'A': hamiltonian,
         'B': np.zeros((number_of_psi - 2, number_of_psi - 2))}

    # Testing Jacobian
    jacobian = sim.calc_jacobian_numerical(sim.dxdt_f, init_state, u(0), p, 1e-3)
    # print(jacobian)

    x_final, x_arr = sim.forward_euler(sim.dxdt_f, u, init_state, p, t_start=time_start, t_stop=time_stop,
                                       delta_t=delta_t)

    # stationary_state = eigenvectors[mode-1]

    # Display animation
    if display_animation:
        ani = AnimationClass(fps=gif_animation_frames_per_second,
                             x=np.linspace(start_x, stop_x, number_of_psi),
                             x_arr=x_arr, runtime_seconds=time_stop, delta_t=delta_t, playback_speed=10,
                             gif_name =  gif_name)
        ani.run_animation()
        ani.create_gif()
        ani.create_video()

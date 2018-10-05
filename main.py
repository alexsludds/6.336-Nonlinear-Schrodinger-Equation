import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from array2gif import write_gif
import matplotlib.animation as animation
from progress.bar import Bar


class simulation():
    def __init__(self,x_start=0, x_stop=1, number_of_psi = 100, number_of_spatial_dimensions = 1, potential_function = lambda x:0):
        self.number_of_psi = number_of_psi
        self.number_of_spatial_dimensions = number_of_spatial_dimensions
        self.constituent_matrix = None
        self.potential_matrix = None
        self.x_start = x_start
        self.x_stop = x_stop
        self.potential_function = potential_function
        self.linspace = np.linspace(self.x_start,self.x_stop,num=self.number_of_psi-2)

    """
    Returns a copy of matrix describing relationships between Psi's
    """
    def generate_constituent_matrix(self):
        #set off diagonal terms to -1
        D = -1*np.diag(np.ones(self.number_of_psi-3), 1) - 1* np.diag(np.ones(self.number_of_psi-3),-1)
        #set diagonal to 2
        np.fill_diagonal(D,2)
        if(self.number_of_spatial_dimensions==1):
            self.constituent_matrix = D
            return D

        elif(self.number_of_spatial_dimensions==2):
            tensor_product = np.tensordot(D,D,axes=0)
            self.constituent_matrix = tensor_product
            return tensor_product

        elif(self.number_of_spatial_dimensions==3):
            pass
    """
    Given the values of x_start, x_stop, and the potential function that is defined
    """
    def generate_potential_matrix(self):
        potential_values = np.array(list(map(self.potential_function,self.linspace)))
        diagonal_matrix = np.diag(potential_values)
        self.potential_matrix = diagonal_matrix
        return diagonal_matrix

    """
    Given the hermatian matrix return the eigenvalues and eigenvectors
    """
    def find_eigen_values(self,H_matrix):
        eigenValues,eigenVectors = np.linalg.eig(H_matrix)
        # eigenvectors = eigenVectors.T #possibly the more cursed thing numpy ever implemented
        #sort eigenvectors and eigenvalues
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        eigenVectors = -1*eigenVectors.T #possibly the most cursed thing in numpy
        return eigenValues, eigenVectors

    def hamiltonian(self):
        hamiltonian = 1j*(self.constituent_matrix + self.potential_matrix)
        return hamiltonian

    def forward_euler(self,f, u, x_start, p, t_start, t_stop, delta_t):
        x = x_start
        t = t_start
        n = 0
        x_arr = []
        bar = Bar('Processing', max=int((t_stop-t_start)/delta_t))
        while t < t_stop:
            x = x + delta_t * f(x, u(t),p)
            n += 1
            t = t + delta_t
            #normalize
            norm = np.linalg.norm(x)
            x = x/norm
            x_arr.append(x)
            bar.next()
        bar.finish()
        return x, x_arr

    def plot_stationary(self,eigenvector):
        plt.plot(self.linspace,eigenvector)
        plt.show()

    def dxdt_f(self,x, u, p):
        return p["A"].dot(x) + p["B"].dot(u)

class animation_class():
    def __init__(self,fps,x,x_arr,runtime_seconds=10):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
        self.l1, = plt.plot([], [], color='b')
        self.l2, = plt.plot([], [], color='g')
        self.fps = fps
        self.x = x
        self.x_arr = x_arr
        self.runtime_seconds = runtime_seconds
        # self.time_step = int(len(self.x_arr)/(self.fps*self.runtime_seconds))
        self.time_step = 700
        if(self.time_step==0):
            self.time_step = 1

    def animate(self,i):
        self.l1.set_data(self.x, np.real(self.x_arr[self.time_step*i]))
        self.l2.set_data(self.x, np.imag(self.x_arr[self.time_step*i]))
        return [self.l1, self.l2]

    def initialize(self):
        self.l1.set_data([],[])
        self.l2.set_data([],[])

    def run_animation(self):
        anim = animation.FuncAnimation(self.fig,self.animate,init_func=self.initialize,frames=self.runtime_seconds*self.fps,interval = 2)
        plt.show()

if __name__ == "__main__":
    number_of_psi = 50 #This is the total number of nodes. We are solving for number of nodes -2
    number_of_spatial_dimensions = 1
    mode = 1
    start_x = 0
    stop_x = 1
    hbar = 1.0545718E-34
    gif_animation_frames_per_second = 3
    write_gif_bool = False
    display_animation = True
    plot_stationary_solution = True
    gif_name = "test.gif"
    time_start = 0
    time_stop = 2000
    delta_t = 0.0005

    sim = simulation(x_start = start_x, x_stop = stop_x ,number_of_psi=number_of_psi, number_of_spatial_dimensions=number_of_spatial_dimensions)
    constituent_matrix = sim.generate_constituent_matrix()
    potential_matrix = sim.generate_potential_matrix()
    hamiltonian = sim.hamiltonian()
    # hamiltonian = constituent_matrix + potential_matrix
    eigenvalues, eigenvectors  = sim.find_eigen_values(hamiltonian)

    #plot the stationary solution
    if(plot_stationary_solution==True):
        sim.plot_stationary(eigenvectors[mode-1])

    # stationary_state = eigenvectors[mode-1]
    if(write_gif_bool==True or display_animation==True):
        stationary_state = np.sqrt(2)*np.sin(mode*np.pi*np.linspace(start_x,stop_x,number_of_psi-2))
        u = lambda _: np.zeros(number_of_psi-2)
        p = {'A': hamiltonian,
             'B': np.zeros((number_of_psi-2, number_of_psi-2))}
        x_final,x_arr = sim.forward_euler(sim.dxdt_f,u, stationary_state, p, t_start = time_start,t_stop = time_stop,delta_t=delta_t)

    #Display animation
    if(display_animation==True):
        ani = animation_class(fps=gif_animation_frames_per_second,x=np.linspace(start_x,stop_x,number_of_psi-2),x_arr=x_arr,runtime_seconds=time_stop)
        ani.run_animation()

    #Create gif from data
    if(write_gif_bool==True):
        print(x_arr[0])
        # write_gif(,gif_name,fps=30)

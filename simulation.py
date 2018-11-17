import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.misc import imread
import scipy.constants
import matplotlib
import matplotlib.pyplot as plt
from benchmark import benchmark
import problems
from progress.bar import Bar
import os, sys, time

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
        while t < t_stop:
            x = x + delta_t * f(x, u(t), p)
            n += 1
            t = t + delta_t
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
                x = x  + f(x, u(t-1), u(t), p, delta_t, inverse)
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
    #Trapezoidal Nonlinearity
    def trapezoidal_nl(self, f, u, x_start, p, t_start, t_stop, delta_t, animation_timestep):
        x = x_start
        t = t_start
        x_arr = []
        n = 0
        # TODO This is the slow version of the trapezoidal rule, we should get around to implementing the iterative
        # version at some point

        bar = Bar("Processing", max=int((t_stop-t_start)/delta_t), suffix='%(percent).1f%% - %(eta)ds')
        while t < t_stop:
            if t == 0:
                inverse = np.eye(len(u(0))) - delta_t/2*p["A"](x_start)
                np.linalg.inv(inverse)
                x = f(x,u(0),u(0),p, delta_t,inverse)
            else:
                inverse = np.eye(x.shape[0],x.shape[0]) - delta_t/2*p["A"](x)
                inverse = np.linalg.inv(inverse)
                x = f(x,u(t-1),u(t),p, delta_t, inverse)
            t = t + delta_t
            n += 1

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
        if not self.nonlinear:
            RHS = x + delta_t/2.*p['B'].dot(u_previous) + delta_t/2.*p['B'].dot(u_current) + delta_t/2.*p['A'].dot(x)
            return np.dot(inverse,RHS)
        elif self.nonlinear:
            RHS = x + delta_t/2.*p['B'].dot(u_previous) + delta_t/2.*p['B'].dot(u_current) + delta_t/2.*p['A'](x).dot(x)
            assert np.any(np.isfinite(x))
            return np.dot(inverse,RHS)

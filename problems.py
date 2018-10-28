import numpy as np
import scipy as sp
import scipy.constants
from benchmark import benchmark

hbar = sp.constants.h / (2*sp.pi)   # Reduced Planck constant = 1.055e-34 J s/rad


class Problem:
    def __init__(self, x_start=0, x_stop=1, number_of_psi=100, number_of_spatial_dimensions=1,
                 non_linear=False):
        self.number_of_psi = number_of_psi
        self.number_of_spatial_dimensions = number_of_spatial_dimensions
        self.constituent_matrix = None
        self.potential_matrix = None
        self.x_start = x_start
        self.x_stop = x_stop
        self.linspace = np.linspace(self.x_start, self.x_stop, num=self.number_of_psi)[1:-1]
        self.dx = (self.x_stop - self.x_start)/(self.number_of_psi-1)
        self.non_linear = non_linear

    @benchmark
    def second_derivative(self):
        """Matrix calculating second derivative with 0 boundary condition"""
        D = None
        if self.number_of_spatial_dimensions == 1:
            D = np.diag(np.ones(self.number_of_psi - 3), 1) + np.diag(np.ones(self.number_of_psi - 3), -1)
            np.fill_diagonal(D, -2)
            D /= self.dx ** 2
        elif self.number_of_spatial_dimensions == 2:
            # TODO: Clean this up and consider boundary conditions
            # This basically just generates a block matrix of the 2D case.
            # Best way to see this work is to swap all of the toblock.append(D) to something like toblock.append("D")
            # then print Q at the end
            D = np.diag(np.ones(self.number_of_psi - 1), 1) + 1 * np.diag(np.ones(self.number_of_psi - 1), -1)
            np.fill_diagonal(D, 4)
            I = -np.eye(self.number_of_psi)
            Z = np.zeros((self.number_of_psi, self.number_of_psi))
            toblockarray = []
            for row in range(self.number_of_psi):
                toblock = []
                for col in range(self.number_of_psi):
                    if row == col:
                        toblock.append(D)
                    elif row == col - 1:
                        toblock.append(I)
                    elif row == col + 1:
                        toblock.append(I)
                    else:
                        toblock.append(Z)
                toblockarray.append(toblock)
            D = np.block(toblockarray)
        else:
            pass

        self.constituent_matrix = D
        return D

    def calc_A(self):
        pass

    def get_P(self):
        pass

    def get_u(self):
        pass


class Quantum(Problem):
    def __init__(self, x_start=0, x_stop=1, number_of_psi=100, number_of_spatial_dimensions=1,
                 potential_function=lambda x: 0, non_linear=False, mass=9.109e-31):
        super().__init__(x_start=x_start, x_stop=x_stop, number_of_psi=number_of_psi,
                         number_of_spatial_dimensions=number_of_spatial_dimensions, non_linear=non_linear,)
        self.potential_matrix = self.generate_potential_matrix(potential_function)
        self.mass = mass
        self.time_multiplier = 1e5
        self.A = self.calc_A()

    @benchmark
    def generate_potential_matrix(self, potential_function):
        """Generate matrix V of potential using a function for potential as a function of coordinate"""
        potential_values = np.array(list(map(potential_function, self.linspace)))
        return np.diag(potential_values)

    """
    Calculates the Hamiltonian of the system, which is i*(constituent matrix + potential matrix)
    """
    @benchmark
    def calc_hamiltonian(self):
        hamiltonian = -1j*(-hbar**2/(2*self.mass)*self.second_derivative() + self.potential_matrix)/hbar
        return hamiltonian*self.time_multiplier

    def calc_A(self):
        return self.calc_hamiltonian()

    def get_P(self):
        p = {'A': self.A,
             'B': np.zeros((self.number_of_psi - 2, self.number_of_psi - 2))}
        return p

    def get_u(self):
        def u(_):
            return np.zeros(self.number_of_psi - 2)
        return u


class NLSE(Problem):
    """Nonlinear Schrodinger equation for nonlinear fiber optics"""
    def __init__(self, x_start=0, x_stop=1, number_of_psi=100, number_of_spatial_dimensions=1,
                 non_linear=False, alpha=1e-12):
        super().__init__(x_start=x_start, x_stop=x_stop, number_of_psi=number_of_psi,
                         number_of_spatial_dimensions=number_of_spatial_dimensions, non_linear=non_linear, )
        self.alpha = alpha

    def calc_A(self, x=None):
        if x is None:
            return self.alpha*self.second_derivative()/1j
        else:
            return (self.alpha*self.second_derivative() + self.nonlinear_matrix(x)) / 1j

    @benchmark
    def nonlinear_matrix(self, x):
        return np.diag(np.square(np.abs(x)))

    @benchmark
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

    @benchmark
    def stationary_matrix(self, omega):
        right = (self.constituent_matrix + self.potential_matrix)
        right += omega*np.identity(self.number_of_psi-2)


if __name__ == "__main__":
    problem = Problem(x_start=0, x_stop=1, number_of_psi=10, number_of_spatial_dimensions=1,
                      non_linear=False)
    print(problem.second_derivative())

    problem = Problem(x_start=0, x_stop=1, number_of_psi=10, number_of_spatial_dimensions=2,
                      non_linear=False)
    print(problem.second_derivative())

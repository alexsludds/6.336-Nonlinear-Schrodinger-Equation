import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

"""
Returns a copy of matrix describing relationships between Psi's
"""
def generate_constituent_matrix(number_of_psi = 100,number_of_spatial_dimensions=1):
    #set off diagonal terms to -1
    D = -1*np.roll(np.eye(number_of_psi-2),1) - 1*np.roll(np.eye(number_of_psi-2),-1)
    #set diagonal to 2
    np.fill_diagonal(D,2)
    if(number_of_spatial_dimensions==1):
        return D
    elif(number_of_spatial_dimensions==2):
        pass
    elif(number_of_spatial_dimensions==3):
        pass
    else:
        print("Number of spatial dimensions incorrect:",number_of_spatial_dimensions)

"""
Given a function, create a potential matrix
"""
def generate_potential_matrix(potential_function,number_of_psi = 100,start=0,stop=1):
    #define a function here to approximate
    linspace = np.linspace(start,stop,num=number_of_psi-2)
    print()
    potential_values = np.array(list(map(potential_function,linspace)))
    diagonal_matrix = np.diag(potential_values)
    return diagonal_matrix

"""
Given the hermatian matrix return the eigenvalues and eigenvectors
"""
def find_eigen_values(H_matrix):
    eigenValues,eigenVectors = np.linalg.eig(H_matrix)
    # eigenvectors = eigenVectors.T #possibly the more cursed thing numpy ever implemented
    #sort eigenvectors and eigenvalues
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    eigenVectors = eigenVectors.T #possibly the most cursed thing in numpy
    return eigenValues, eigenVectors



if __name__ == "__main__":
    number_of_psi = 100
    hbar = 1.0545718E-34
    constituent_matrix = generate_constituent_matrix(number_of_psi=number_of_psi)
    print(constituent_matrix)
    #create a function that returns zero
    def potential(x):
        return (x-0.5)**2
    potential_matrix = generate_potential_matrix(potential,number_of_psi=number_of_psi)
    eigenvalues, eigenvectors  = find_eigen_values(constituent_matrix + potential_matrix)
    plt.plot(eigenvectors[0])
    plt.plot(eigenvectors[1])
    plt.plot(eigenvectors[2])
    plt.show()

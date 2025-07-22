#!/usr/bin/env python
#
####  Uncomment and activate the code for graphs with MatPlotLib 
import matplotlib.pyplot as plt
####  Uncomment to display graphs in a Jupyter Notebook
#%matplotlib inline
####  Uncomment to import/process/export data with Pandas
#import pandas as pd
####  Uncomment to optimize CPU execution with Numba
from numba import njit
#
# Load NumPy (Numerical Python)
import numpy as np

# Attiva generatore di numeri pseudocasuali  (default PCG64)
#rng = np.random.default_rng(seed=42)

# An implementation of a Metropolis-Hastings algorithm
# with simulated annealing applied to a 2D Ising spin glass
# Fix random seed
np.random.seed(123)


def init_state_SA(spins,mag,energy,ts):

    # Initialize interaction arrays
    # Have 4 arrays: up, down, left right
    # These represent the interaction strengths to the
    # up/down/left/right neighbours of a site
    # There is a symmetry between these matrices
    # This is not the most memory efficient solution
    s_h = 1
    s_v = 1
    L = spins.shape[0]
    up = np.zeros((L,L))
    down = np.zeros((L,L))
    left = np.zeros((L,L))
    right = np.zeros((L,L))
    
    up[1:L,:] = np.random.rand(L-1,L) * s_v
    down[0:L-1,:] = up[1:L,:]
    left[:,1:L] = np.random.rand(L,L-1) * s_h
    right[:,0:L-1] = left[:,1:L]
    
    
    mag[0] = spins.sum()
    
    for i in range(L):
        for j in range(L):
            if i == 0:
                up_neighbour = 0
                down_neighbour = spins[i+1,j]
            elif i == L-1:
                up_neighbour = spins[i-1,j]
                down_neighbour = 0
            else:
                up_neighbour = spins[i-1,j]
                down_neighbour = spins[i+1,j]
            if j == 0:
                left_neighbour = 0
                right_neighbour = spins[i,j+1]
            elif j == L-1:
                left_neighbour = spins[i,j-1]
                right_neighbour = 0
            else:
                left_neighbour = spins[i,j-1]
                right_neighbour = spins[i,j+1]
    
            energy[0] += spins[i,j]*(up[i,j]*up_neighbour + down[i,j]*down_neighbour + left[i,j]*left_neighbour + right[i,j]*right_neighbour)
    
    # Avoid double count - each neighbour pair
    # counted twice in above since loop over each site
    energy[0] /= 2
    
    return spins,mag,energy,up, down, left, right

# Define proposal step
def proposal(s_array):
    _L = s_array.shape[0]
    return np.random.choice(_L, 2)

def energy_change(spin_site, bt, s_array, up_array, down_array, left_array, right_array):
    i = spin_site[0]
    j = spin_site[1]

    L = s_array.shape[0]

    if i == 0:
        up_neighbour = 0
        down_neighbour = s_array[i+1,j]
    elif i == L-1:
        up_neighbour = s_array[i-1,j]
        down_neighbour = 0
    else:
        up_neighbour = s_array[i-1,j]
        down_neighbour = s_array[i+1,j]
    if j == 0:
        left_neighbour = 0
        right_neighbour = s_array[i,j+1]
    elif j == L-1:
        left_neighbour = s_array[i,j-1]
        right_neighbour = 0
    else:
        left_neighbour = s_array[i,j-1]
        right_neighbour = s_array[i,j+1]

    dE_tmp = 2*s_array[i,j]*(up_array[i,j]*up_neighbour + down_array[i,j]*down_neighbour + left_array[i,j]*left_neighbour + right_array[i,j]*right_neighbour)
    return dE_tmp

def acceptance(bt, energy):
    if energy <= 0:
        return -1
    else:
        prob = np.exp(-bt*energy)
        if prob > np.random.random():
            return -1
        else:
            return 1


# Define update step
dE = 0
dM = 0

def update(bt, s_array, up_array, down_array, left_array, right_array):
    global dE
    global dM

    # Proposal Step
    site = proposal(s_array)

    # Calculate energy change
    dE = energy_change(site, bt, s_array, up_array, down_array, left_array, right_array)
    dM = -2*s_array[site[0],site[1]]

    # Acceptance step
    accept = acceptance(bt, dE)

    if accept == -1:
        s_array[site[0], site[1]] *= -1
    else:
        dE = 0
        dM = 0

    return s_array


def plot_energy_magnetization(mag,energy):
    # plot magnetism and energy evolving in time
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Magnetism", color='blue')
    ax1.plot(mag, color='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy", color='red')
    ax2.plot(energy, color='red')
    
    plt.show()




#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Main parameters
L = 50                # Lattice dimension (50x50)
T = 2.0               # Initial temperature (near critical T ~2.27), Boltzmann constant = 1
num_steps = 2000      # Number of evolution steps
equilibration = 200   # Number of thermal equilibrium steps (to reach a steady state)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
seed = 42             # fixed rng seed

def glauber_lattice_step(lattice, T):
    """
    Performs a Glauber step on the entire lattice
    """
    L = len(lattice)
    for _ in range(L**2):

        # Select a random site
        i, j = np.random.randint(0, L, 2)

        lattice = glauber_spin_update(lattice,i,j,J,T)

    return lattice


if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(L,p)

    print_parameter(L, J, p,T,num_steps)

    print_system_info(lattice, J, p)
    
    # List to store magnetization over time
    magnetization = []
    energy        = []
    
    # System evolution
    for step in range(num_steps):
        lattice = glauber_lattice_step(lattice, T)
    
        # After a certain number of thermal equilibrium steps, record the magnetization
        if step >= equilibration:
            mag = compute_average_magnetization(lattice)
            magnetization.append(mag)
            E_avg = calculate_total_energy(lattice, J) / lattice.size
            energy.append(E_avg)

    
        if step % 100 == 0:
            print("step: ",step)

    print("step: ",num_steps)

    print_system_info(lattice, J, p)

    # Plot average magnetization over time
    plot_magnetization(magnetization,T)
    plot_energy(energy,T)
    
    
    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — T={T} "
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)

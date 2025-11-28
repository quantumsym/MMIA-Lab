#!/usr/bin/env python
#
import time
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Main parameters
L = 50                # Lattice side dimension (50x50)
T = 2.0               # Initial temperature (T_critical ~ 2.27), Boltzmann constant = 1
num_steps = 2000      # Number of time evolution steps
equilibration = 200   # Number of steps to reach thermal equilibrium (estimate)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
B = 0.0               # optional magnetic field

# ---------------------------------------------

if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(L,p)

    print_parameter(L, J, p,T,num_steps)

    print_system_info(lattice, J, p)
    
    # Save start time for performance comparison
    start = time.time()

    # List to store magnetization over time
    magnetizations = []
    energies = []

    beta = 1.0 / T
    
    # System evolution - Gibbs Sampling
    for step in range(num_steps):

        # Systematic scan of all spins
        for i in range(L):
            for j in range(L):
                lattice = gibbs_sampling(lattice, i, j, J, B, beta)

        # After a certain number of steps, at thermal equilibrium, calculate the observables
        if step >= equilibration:
            # magnetization over time
            mag = compute_average_magnetization(lattice)
            magnetizations.append(mag)
            # energy over time
            energy = calculate_total_energy(lattice, J) / lattice.size
            energies.append(energy)

        if step % 100 == 0:
            print("step: ",step)

    print("step: ",num_steps)
    time1 = time.time() - start
    print("execution time: ",time1)

    print_system_info(lattice, J, p)

    # Plot average magnetization over time
    plot_magnetization(magnetizations,T)
    plot_energy(energies,T)
    
    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — T={T} "
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)

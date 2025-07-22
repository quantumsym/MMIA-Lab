#!/usr/bin/env python
#
import time
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Parametri principali
L =  50               # Dimensione lato del reticolo (50x50)
T = 2.0               # Temperatura iniziale (T_critica ~ 2.27), costante Boltzmann = 1
num_steps = 2000      # Numero di passi di evoluzione temporale
equilibration = 200   # Numero di passi per raggiungere equilibrio termico (stima)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
B = 0.0               # optional magnetic field

# --------------------------------------------------------------------------------------

def gibbs_sampling(spins, i, j, J=1.0, B=0.0, beta= 1.0):
    # Calcola il campo locale
    h_local = calculate_local_field(spins, i, j, J, B)

    # Probabilità condizionale per spin up
    P_up = 1.0 / (1.0 + np.exp(-2 * beta * h_local))

    # Campionamento dalla distribuzione condizionale
    if rng.random() < P_up : 
        spins[i, j] = 1 
    else:
        spins[i, j] = -1

    return spins

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(L,p)

    print_parameter(L, J, p,T,num_steps)

    print_system_info(lattice, J, p)
    
    # Save start time for performance comparison
    start = time.time()

    # Lista per memorizzare magnetizzazione nel tempo
    magnetizations = []
    energies = []

    beta = 1.0 / T
    
    # Evoluzione del sistema - Campionamento di Gibbs
    for step in range(num_steps):

        # Scansione sistematica di tutti gli spin
        for i in range(L):
            for j in range(L):
                lattice = gibbs_sampling(lattice, i, j, J, B, beta)

        # Dopo un certo numero di passi, di equilibrio termico, calcola le osservabili
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



#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Parametri principali
L =  50               # Dimensione del lattice (50x50)
T = 2.0               # Temperatura iniziale (vicino alla T critica ~2.27), costante Boltzmann = 1
num_steps = 2000      # Numero di passi di evoluzione
equilibration = 200   # Numero di passi di equilibrio termico (per raggiungere stato stazionario)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
seed = 42             # fixed rng seed

def glauber_lattice_step(lattice, T):
    """
    Esegue un passo di Glauber su tutto il lattice
    """
    L = len(lattice)
    for _ in range(L**2):

        # Seleziona un sito casuale
        i, j = np.random.randint(0, L, 2)

        lattice = glauber_spin_update(lattice,i,j,J,T)

    return lattice


if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(L,p)

    print_parameter(L, J, p,T,num_steps)

    print_system_info(lattice, J, p)
    
    # Lista per memorizzare magnetizzazione nel tempo
    magnetization = []
    energy        = []
    
    # Evoluzione del sistema
    for step in range(num_steps):
        lattice = glauber_lattice_step(lattice, T)
    
        # Dopo un certo numero di passi di equilibrio termico, registra la magnetizzazione
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


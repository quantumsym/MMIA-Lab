#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Parametri principali
L =  50               # Dimensione del lattice (50x50)
T = 3.0               # Temperatura iniziale (vicino alla T critica ~2.27), costante Boltzmann = 1
num_steps = 2000      # Numero di passi di evoluzione
equilibration = 200   # Numero di passi di equilibrio termico (per raggiungere stato stazionario)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
seed = 42             # fixed rng seed

def metropolis_lattice_step(lattice, T):
    """
    Esegue un passo di Metropolis su tutto il lattice
    """
    L = len(lattice)
    for _ in range(L**2):

        # Seleziona un sito casuale
        i, j = np.random.randint(0, L, 2)

        lattice = metropolis_spin_update(lattice,i,j,J,T)

    return lattice


if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(seed,L,p)
    print_system_info(lattice, J, p)
    print("Temperature: ",T)
    print("Steps: ",num_steps)
    
    # Lista per memorizzare magnetizzazione nel tempo
    magnetizations = []
    
    # Evoluzione del sistema
    for step in range(num_steps):
        lattice = metropolis_lattice_step(lattice, T)
    
        # Dopo un certo numero di passi di equilibrio termico, registra la magnetizzazione
        if step >= equilibration:
            mag = compute_average_magnetization(lattice)
            magnetizations.append(mag)
    
        if step % 200 == 0:
            print("step: ",step)


    print_system_info(lattice, J, p)

    # Plot average magnetization over time
    plot_magnetization(magnetizations,T)
    
    
    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — p={p}, J={J}"
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)


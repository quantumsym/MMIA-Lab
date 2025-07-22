#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *


# Parametri principali
L =  50               # Dimensione del lattice (50x50)
T = 1.0               # Temperatura iniziale (vicino alla T critica ~2.27)
num_steps = 1000      # Numero di passi di evoluzione
equilibration = 200   # Numero di passi di equilibrio termico (per raggiungere stato stazionario)
p = 0.5               # probability of spin −1
J = 1.0               # coupling constant
seed = 42             # fixed rng seed


def metropolis_step(lattice, T):
    """
    Esegue un passo di Metropolis su tutto il lattice
    """
    L = len(lattice)
    for _ in range(L**2):

        # Seleziona un sito casuale
        i, j = np.random.randint(0, L, 2)

        # Calcola la variazione di energia se si flippa lo spin
        # -2 for local site energy
        delta_E = -2 * local_bond_energy_all(lattice,i,j,J)

        # Accetta il flip se diminuisce energia o con probabilità Boltzmann
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            lattice[i, j] *= -1  # Flip dello spin

    return lattice


if __name__ == "__main__":
    
    # Initialize lattice and analyze
    lattice = initialize_lattice(seed,L,p)
    print_system_info(lattice, J, p)
    
    # Lista per memorizzare magnetizzazione nel tempo
    magnetizations = []
    
    # Evoluzione del sistema
    for step in range(num_steps):
        lattice = metropolis_step(lattice, T)
    
        # Dopo un certo numero di passi di equilibrio termico, registra la magnetizzazione
        if step >= equilibration:
            mag = compute_average_magnetization(lattice)
            magnetizations.append(mag)
    
    # Plot average magnetization over time
    plot_magnetization(magnetizations,T)
    
    
    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — p={p}, J={J}"
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)


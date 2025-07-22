#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# Parametri principali
size = 50             # Dimensione del lattice (50x50)
temperature = 2.25   # Temperatura iniziale (vicino alla T critica ~2.27)
num_steps = 2000      # Numero di passi di evoluzione
equilibration = 500 # Numero di passi di equilibrio termico (per raggiungere stato stazionario)

def initialize_lattice(size):
    """Inizializza il lattice con spin casuali (+1 o -1)"""
    return np.random.choice([-1, 1], size=(size, size))

def calculate_energy(lattice, i, j):
    """Calcola l'energia locale di uno spin (i,j) considerando i vicini"""
    size = len(lattice)

    # Somma degli spin vicini con condizioni periodiche (bordo toroidale)
    neighbors_sum = (
        lattice[(i+1) % size, j] +
        lattice[(i-1) % size, j] +
        lattice[i, (j+1) % size] +
        lattice[i, (j-1) % size]
    )

    # Energia locale: -spin * somma vicini
    return -lattice[i, j] * neighbors_sum


def metropolis_step(lattice, temperature):

    """Esegue un passo di Metropolis su tutto il lattice"""
    size = len(lattice)
    for _ in range(size**2):
        # Seleziona un sito casuale
        i, j = np.random.randint(0, size, 2)

        # Calcola la variazione di energia se si flippa lo spin
        delta_E = -2 * calculate_energy(lattice, i, j)

        # Accetta il flip se diminuisce energia o con probabilità Boltzmann
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / temperature):
            lattice[i, j] *= -1  # Flip dello spin

    return lattice

def compute_magnetization(lattice):
    """Calcola la magnetizzazione media del sistema"""
    return np.abs(np.sum(lattice)) / (size * size)

def plot_magnetization(magnetizations):
    """ Plot della magnetizzazione nel tempo  """
    plt.ylim(-0.5,1.5)
    plt.plot(magnetizations)
    plt.xlabel('Numero di passi')
    plt.ylabel('Magnetizzazione assoluta')
    plt.title(f'Simulazione Ising 2D a T={temperature}')
    plt.show()
    


# Inizializza il lattice
lattice = initialize_lattice(size)

# Lista per memorizzare magnetizzazione nel tempo
magnetizations = []

# Evoluzione del sistema
for step in range(num_steps):
    lattice = metropolis_step(lattice, temperature)

    # Dopo un certo numero di passi di equilibrio termico, registra la magnetizzazione
    if step >= equilibration:
        mag = compute_magnetization(lattice)
        magnetizations.append(mag)

plot_magnetization(magnetizations)



#!/usr/bin/env python
#
"""
    Campionamento di Gibbs per il modello di Ising 2D

    Parameters:
    - size: dimensione del reticolo (size x size)
    - J: costante di accoppiamento
    - B: campo magnetico esterno
    - beta: inverso della temperatura (1/kT)
    - n_steps: numero di passi di campionamento
"""
import numpy as np
import matplotlib.pyplot as plt

# Simulazione a diverse temperature
temperatures = [1.0, 2.0, 3.0, 4.0]
size = 20
n_steps = 500

def calculate_local_field(spins, i, j, J, B):
    """Calcola il campo locale per lo spin in posizione (i,j)"""
    size = spins.shape[0]

    # Condizioni al contorno periodiche
    neighbors = (
        spins[(i-1) % size, j] +
        spins[(i+1) % size, j] +
        spins[i, (j-1) % size] +
        spins[i, (j+1) % size]
    )

    return J * neighbors + B

def calculate_energy(spins, J, B):
    """Calcola l'energia totale del sistema"""
    size = spins.shape[0]
    energy = 0
    for i in range(size):
        for j in range(size):
            # Energia di interazione (evitando doppio conteggio)
            energy -= J * spins[i, j] * (
                spins[(i+1) % size, j] +
                spins[i, (j+1) % size]
            )
            # Energia nel campo magnetico
            energy -= B * spins[i, j]

    return energy


def gibbs_sampling(spins, i, j, J=1.0, B=0.0, beta= 1.0):
    # Calcola il campo locale
    h_local = calculate_local_field(spins, i, j, J, B)

    # Probabilità condizionale per spin up
    prob_up = 1.0 / (1.0 + np.exp(-2 * beta * h_local))

    # Campionamento dalla distribuzione condizionale
    spins[i, j] = 1 if rng.random() < prob_up else -1

    return spins

def gibbs_ising_2d(size, J=1.0, B=0.0, beta=1.0, n_steps=1000):
    # Campionamento di Gibbs per il modello di Ising 2D

    # Inizializzazione casuale degli spin
    spins = rng.choice([-1, 1], size=(size, size))

    # Liste per memorizzare magnetizzazione ed energia
    magnetizations = []
    energies = []

    # Campionamento di Gibbs
    for step in range(n_steps):
        # Scansione sistematica di tutti gli spin
        for i in range(size):
            for j in range(size):
                spins = gibbs_sampling(spins, i, j, J, B, beta)

        # Calcola osservabili
        magnetization = np.sum(spins) / (size * size)
        energy = calculate_energy(spins, J, B) / (size * size)
    
        magnetizations.append(magnetization)
        energies.append(energy)

    return spins, magnetizations, energies


def plot_energy_magnetization(idx,mags,energies):

    # Grafici
    ax = axes[idx]
    steps = range(len(mags))

    # Sottotrame per magnetizzazione ed energia
    ax2 = ax.twinx()

    line1 = ax.plot(steps, mags, 'b-', alpha=0.7, label='Magnetizzazione')
    line2 = ax2.plot(steps, energies, 'r-', alpha=0.7, label='Energia')

    ax.set_xlabel('Passo Monte Carlo')
    ax.set_ylabel('Magnetizzazione', color='b')
    ax2.set_ylabel('Energia per spin', color='r')
    ax.set_title(f'T = {T:.1f}')
    ax.grid(True, alpha=0.3)

    # Legenda
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')


def plot_end_config(temperatures):
    # Visualizzazione della configurazione finale
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, T in enumerate(temperatures):
        beta = 1.0 / T
        final_spins, _, _ = gibbs_ising_2d(size=size, beta=beta, n_steps=n_steps)
    
        axes[idx].imshow(final_spins, cmap='RdBu', interpolation='nearest')
        axes[idx].set_title(f'T = {T:.1f}')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    plt.tight_layout()
    plt.show()



# Esempio di utilizzo

# Inizializzazione generatore numeri casuali
rng = np.random.default_rng(seed=42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, T in enumerate(temperatures):
    beta = 1.0 / T

    # Esegui simulazione
    final_spins, mags, energies = gibbs_ising_2d(
        size=size, beta=beta, n_steps=n_steps
    )

    # grafici
    plot_energy_magnetization(idx,mags,energies)

plt.tight_layout()
plt.show()

# diagrammi configurazione finale
plot_end_config(temperatures)


    

#!/usr/bine/env python
#
import numpy as np

# Parametri
size = 50
J = 1.0
TEMPS = [0.5, 1.5, 2.3, 3.5]

def gibbs_update(spins, temperature):
    """Singolo aggiornamento Gibbs"""

    beta = 1.0 / temperature
    new_spins = spins.copy()
    for i in range(size):
        for j in range(size):
            # Somma vicini
            neighbors_sum = (
                spins[(i-1) % size, j] + spins[(i+1) % size, j] + 
                spins[i, (j-1) % size] + spins[i, (j+1) % size]
            )
            # Probabilità condizionale
            prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * J * neighbors_sum))
            # Aggiorna
            new_spins[i, j] = 1 if rng.random() < prob_up else -1
    return new_spins

def compute_energy(test_spins):
    energy = 0.0
    for i in range(size):
        for j in range(size):
            energy -= J * test_spins[i, j] * (
                test_spins[(i+1) % size, j] + test_spins[i, (j+1) % size]
            )
    energy /= (size * size)
    return energy

def compute_magnetization(test_spins):
    magnetization = np.sum(test_spins) / (size * size)
    return magnetization


def simple_ising_demo():
    """
    Versione semplificata per dimostrare i concetti base
    """

    # Inizializza il modello
    spins = rng.choice([-1, 1], size=(size, size))

    # Test a list of temperature
    for temp in TEMPS:
        test_spins = spins.copy()

        # Equilibrazione
        for _ in range(200):
            test_spins = gibbs_update(test_spins, temp)

        # Calcola proprietà
        energy = compute_energy(test_spins)
        magnetization = compute_magnetization(test_spins)

        print(f"T = {temp:.1f}: E = {energy:.4f}, M = {magnetization:.4f}")

# Esegui demo

if __name__ == "__main__":

    rng = np.random.default_rng(seed=42)

    simple_ising_demo()


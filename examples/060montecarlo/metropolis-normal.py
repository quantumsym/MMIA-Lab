#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import * 

# Numero di campioni 
n_samples=500

#----------------------------------------------------------------
# Distribuzione target: Gaussiana standard
def target_distribution(x):
    return np.exp(-x**2 / 2)

#----------------------------------------------------------------
# Algoritmo di Metropolis-Hastings
def metropolis_hastings_simple(n_samples=1000, sigma=1.0):
    """
    Implementazione semplice di Metropolis-Hastings per campionare
    da una distribuzione Gaussiana N(0,1)
    """
    samples = []
    x = 0.0  # valore iniziale
    n_accepted = 0

    rng = np.random.default_rng()


    for i in range(n_samples):
        # Proposta: passo casuale gaussiano
        x_new = x + rng.normal(0, sigma)

        # Rapporto di aggiornamento, tra la probabilità del nuovo e quella del vecchio
        update_ratio = target_distribution(x_new) / target_distribution(x)

        # Rapporto di accettazione, update_ratio se < 1,  1 se update_ratio > 1
        acceptance_ratio = min(1, update_ratio)

        # Accetta o rifiuta con probabilità  acceptance_ratio
        if rng.random() < acceptance_ratio:
            x = x_new
            n_accepted += 1

        #  aggiunge il risultato alla lista dei campioni
        samples.append(x)

    return np.array(samples), n_accepted / n_samples

#----------------------------------------------------------------
    
# Esecuzione
samples, acceptance_rate = metropolis_hastings_simple(n_samples,sigma=1.0)

# Stampa i risultati
print(f"Media campioni: {np.mean(samples):.3f} (teorica: 0.000)")
print(f"Std campioni: {np.std(samples):.3f} (teorica: 1.000)")

# grafico
plot_metropolis_normal(samples, acceptance_rate, n_samples)
#



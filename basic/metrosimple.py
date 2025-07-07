#!/bin/bash
#
import numpy as np
import matplotlib.pyplot as plt

# Distribuzione target: Gaussiana standard
def target_distribution(x):
    return np.exp(-x**2 / 2)

def metropolis_hastings_simple(n_samples=1000000, sigma=1.0):
    """
    Implementazione semplice di Metropolis-Hastings per campionare
    da una distribuzione Gaussiana N(0,1)
    """
    samples = []
    x = 0.0  # valore iniziale
    n_accepted = 0
    for i in range(n_samples):
        # Proposta: passo casuale gaussiano
        x_new = x + np.random.normal(0, sigma)
        # Rapporto di accettazione
        acceptance_ratio = min(1, target_distribution(x_new) / target_distribution(x))
        # Accetta o rifiuta
        if np.random.random() < acceptance_ratio:
            x = x_new
            n_accepted += 1
        samples.append(x)
    return np.array(samples), n_accepted / n_samples


# Esecuzione
samples, acceptance_rate = metropolis_hastings_simple(500000, sigma=1.0)

print(f"Media campioni: {np.mean(samples):.3f} (teorica: 0.000)")
print(f"Std campioni: {np.std(samples):.3f} (teorica: 1.000)")


# Visualizzazione
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.title(f'Trace Plot (primi 500 campioni)\nTasso accettazione: {acceptance_rate:.2f}')
plt.xlabel('Iterazione')
plt.ylabel('Valore campionato')
plt.subplot(1, 2, 2)
plt.hist(samples[100:], bins=50, density=True, alpha=0.7, label='Campioni MCMC')
x_theory = np.linspace(-4, 4, 100)
y_theory = np.exp(-x_theory**2 / 2) / np.sqrt(2 * np.pi)
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label='Distribuzione teorica N(0,1)')
plt.legend()
plt.title('Distribuzione dei campioni')
plt.xlabel('Valore')
plt.ylabel('Densità')
plt.tight_layout()
plt.show()



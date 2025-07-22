#!/usr/bin/env python3
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#----------------------------------------------------------------
def gibbs_sampling_bivariate_normal(rho, num_samples=10000, burn_in=1000):
    """
    Campionamento di Gibbs per distribuzione normale bivariata.
    Parameters:
    rho: coefficiente di correlazione
    num_samples: numero di campioni da generare
    burn_in: numero di campioni iniziali da scartare
    """
    # Inizializzazione
    samples = []
    x, y = 0.0, 0.0  # valori iniziali
    # Parametri delle distribuzioni condizionali
    conditional_std = np.sqrt(1 - rho**2)
    # Campionamento
    for i in range(num_samples + burn_in):
        # Campiona x condizionato su y
        x = np.random.normal(rho * y, conditional_std)
        # Campiona y condizionato su x
        y = np.random.normal(rho * x, conditional_std)
        # Salva il campione (dopo burn-in)
        if i >= burn_in:
            samples.append([x, y])
    return np.array(samples)

    
#----------------------------------------------------------------
def plot_gibbs_samples(gibbs_samples): 
    # Visualizzazione  Gibbs samples
    plt.figure(figsize=(5, 5))
    plt.scatter(gibbs_samples[:, 0], gibbs_samples[:, 1], alpha=0.6, s=1)
    plt.title('Gibbs Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
#----------------------------------------------------------------
def plot_bivariate_normal_samples(direct_samples):
    # Visualizzazione  Direct samples
    plt.figure(figsize=(5, 5))
    plt.scatter(direct_samples[:, 0], direct_samples[:, 1], alpha=0.6, s=1)
    plt.title('Bivariate Normal Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#----------------------------------------------------------------
def print_gibbs_stats(gibbs_samples):
    # Confronto statistiche
    print("Gibbs Sampling Stats:")
    print(f"Media: {np.mean(gibbs_samples, axis=0)}")
    print(f"Covarianza:\n{np.cov(gibbs_samples.T)}")
    
#----------------------------------------------------------------
def print_bivariate_normal_stats(true_mean,true_cov):
    print("\n Theoretical Stats (Bivariate Normal) :")
    print(f"Media: {true_mean}")
    print(f"Covarianza:\n{np.array(true_cov)}")
    

#-----------------------------------------------------------------------------------
#
# visualizzazione della simulazione e confronto con la soluzione analitica
#
def plot_radioactive_decay(t, N, N0, tau):

    plt.plot(t, N, '.')
    plt.xlabel('Tempo')
    plt.ylabel('Atomi residui')
    plt.title('Decadimento Monte Carlo')
    
    #  Soluzione analitica
    N = N0 * np.exp(-t/tau)
    # grafico
    plt.plot(t, N, color='red', linestyle='--')

    plt.show()
    
#-----------------------------------------------------------------------------------

def plot_metropolis_normal(samples):
    # Visualizzazione della simulazione
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


#!/usr/bin/python
#
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

# Parametri di base
#
N0       = 200       	# atomi iniziali
t_half   = 5.0       	# tempo di dimezzamento
dt       = 0.2          # passo temporale
t0       = 0.0			# tempo iniziale
t_max    = 5 * t_half   # tempo finale simulazione

def montecarlo_decay(N0=1000, tau=10.0, dt=0.1, t_max=20):

    p_decay  = dt / tau
    times, N = [0.0], [N0]

    while times[-1] < t_max:
        decays = sum(random() < p_decay for _ in range(int(N[-1])))

        N.append(N[-1] - decays)
        times.append(times[-1] + dt)

    return np.array(times), np.array(N)

#
n_steps   = int((t_max - t0 ) / dt)    # numero di passi della simulazione
tau      = t_half / np.log(2)   # vita media

t, N = montecarlo_decay(N0,tau,dt,t_max)

plt.plot(t, N, '.')
plt.xlabel('Tempo')
plt.ylabel('Atomi residui')
plt.title('Decadimento Monte Carlo')
#plt.suptitle('decadimento esponenziale tratteggiato in rosso', color='red')

#  Soluzione analitica

#tau      = t_half / np.log(2)     # vita media
t        = np.linspace(t0, t_max, n_steps)
# soluzione analitica
N = N0 * np.exp(-t/tau)
# grafico
plt.plot(t, N, color='red', linestyle='--')
plt.show()





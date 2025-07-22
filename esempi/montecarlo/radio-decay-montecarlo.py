#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import *

# Parametri di base
#
N0       = 100       	# atomi iniziali
t_half   = 5.0       	# tempo di dimezzamento
dt       = 0.2          # passo temporale
t0       = 0.0			# tempo iniziale
t_max    = 20.0         # tempo finale simulazione > t_half

n_steps   = int((t_max - t0 ) / dt)    # numero di passi della simulazione
tau      = t_half / np.log(2)          # costante decadimento esponenziale 

#----------------------------------------------------------------
#
# funzione  decadimento montecarlo
#
def montecarlo_radioactive_decay(N0=1000, tau=10.0, dt=0.1, t_max=20):

    p_decay  = dt / tau      # probabilità di decadimento costante
    times, N = [0.0], [N0]

    rng = np.random.default_rng()   # random number generator

    while times[-1] < t_max:
        decays = sum( rng.random() < p_decay for _ in range(int(N[-1])))

        N.append(N[-1] - decays)
        times.append(times[-1] + dt)

    return np.array(times), np.array(N)

#----------------------------------------------------------------
#
# Esecuzione simulazione
#
if __name__ == "__main__":
    
    t, N = montecarlo_radioactive_decay(N0,tau,dt,t_max)
    
    plot_radioactive_decay(t, N, N0,  tau)
    
    

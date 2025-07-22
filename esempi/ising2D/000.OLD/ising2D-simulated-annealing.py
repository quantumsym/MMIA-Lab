#!/usr/bin/env python
#
####  Uncomment and activate the code for graphs with MatPlotLib 
import matplotlib.pyplot as plt
####  Uncomment to display graphs in a Jupyter Notebook
#%matplotlib inline
####  Uncomment to import/process/export data with Pandas
#import pandas as pd
#
# Load NumPy (Numerical Python)
import numpy as np
#
from SA_Ising2D import *

# Attiva generatore di numeri pseudocasuali  (default PCG64)
#rng = np.random.default_rng(seed=42)

# An implementation of a Metropolis-Hastings algorithm
# with simulated annealing applied to a 2d Ising spin glass

L = 128         # Set size of model L 
T = 0.5         # 
nsteps = 100000 # Fix number of timesteps 




#---------------------------------------------------------------------------
#


def _main_loop_SA(ts, bt_initial, s_array, up_array, down_array, left_array, right_array):
    s_temp = s_array.copy()
    bt_live = bt_initial
    for i in range(ts):
        if ts % 500 == 0:
            bt_live *= 1/0.9
        update_step = update(bt_live, s_temp, up_array, down_array, left_array, right_array)
        s_temp = update_step
        energy[i+1] = energy[i] + dE
        mag[i+1] = mag[i] + dM

#---------------------------------------------------------------------------
#

# Initial beta,  system in glassy-phase for T<s so beta>1/s. Performance
# of random updates isn't good so don't select temperature too low
beta = 1 / T

# Initial spins configuration
spins = np.random.choice([-1, 1], (L,L))

# magnetization and energy containers
mag = np.zeros(nsteps+1)
energy = np.zeros(nsteps+1)

# inizialize some variables
spins, mag, energy, up, down, left, right  = init_state_SA(spins,mag,energy,nsteps)

#### Run Main Loop

_main_loop_SA(nsteps, beta, spins, up, down, left, right)

mag = mag / (L*L)

energy = energy / (L*L)

plot_energy_magnetization(mag,energy)


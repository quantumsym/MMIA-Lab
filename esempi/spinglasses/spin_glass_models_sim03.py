
# An implementation of a Sherrington-Kirkpatrick spin-glass of size N
# Connectivity is initialized as a Gaussian distribution N(0, s^2/N)
# Updates occur at randomly selected sites
"""
 Metropolis-Hastings algorithm. Given a configuration of spins 
 we will perform an update step by picking a site at random, 
 we will compute the probability of flipping the spin 
 and then accept/reject this change based on a random draw. 
 This process will be repeated for a set number of steps. 
 We will keep track of the energy of the system (the Hamiltonian) 
 and the overall magnetism

 https://lewiscoleblog.com/spin-glass-models-2

 """

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Fix random seed
np.random.seed(123)

# Set size of model N and initial spins
N = 1000
spins = np.random.choice([-1, 1], N)

# Fix number of timesteps and some containers
timesteps = 100000
mag = np.zeros(timesteps+1)
energy = np.zeros(timesteps+1)

# Initialize interaction array
s = 1
interaction = np.zeros((N, N))
for i in range(N):
    for j in range(i):
        interaction[i, j] = np.random.randn() * s / np.sqrt(N)
        interaction[j, i] = interaction[i, j]

# Calculate initial values
mag[0] = spins.sum()
energy[0] = -1 * np.dot(spins, np.dot(spins, interaction)) / 2

# Fix beta (inverse temerature) - from analysis we know that
# system in glassy-phase for T<s so beta>1/s. Performance
# of random updates isn't good so don't select temperature
# too low
beta = 1/(0.75*s)

# Define update step
dE = 0
dM = 0

def update(s_array, i_array):
    """
    update function performs 1 update step to the model

    inputs:
    s_array - an array of N spins (+-1)
    i_array - an array of interaction strengths NxN
    """
    global dE
    global dM

    _N = s_array.shape[0]
    old_s = s_array.copy()

    # Select a spin to update
    site = np.random.choice(_N, 1)[0]

    # Get interaction vector
    i_vector = i_array[site,:]

    # Calculate energy change associated with flipping site spin
    dE = 2*np.dot(i_vector, s_array)*s_array[site]
    dM = -2*s_array[site]

    # Calculate gibbs probability of flip
    prob = np.exp(-beta*dE)

    # Sample random number and update site
    if dE <= 0 or prob > np.random.random():
        s_array[site] *= -1
    else:
        dE = 0
        dM = 0

    return s_array

def _main_loop(ts , s_array, i_array):
    s_temp = s_array.copy()
    for i in range(ts):
        update_step = update(s_temp, i_array)
        s_temp = update_step
        energy[i+1] = energy[i] + dE
        mag[i+1] = mag[i] + dM

#### Run Main Loop
_main_loop(timesteps, spins, interaction)

# plot magnetism and energy evolving in time
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time step")
ax1.set_ylabel("Magnetism", color='blue')
ax1.plot(mag, color='blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Energy", color='red')
ax2.plot(energy, color='red')

plt.show()

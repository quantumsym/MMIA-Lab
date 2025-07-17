# An implementation of a Sherrington-Kirkpatrick spin-glass of size N
# Connectivity is initialized as a Gaussian distribution N(0, s^2/N)
# Very naive Monte-Carlo approach
#

"""
We will generate random configurations, calculate the energy 
and a Gibbs measure. With the results we will estimate 
the average energy, the average magnetism, ground state energy 
and the partition function. 
If we want to find the ground state this would be a very bad method, 
for an NxN size spin glass there will be 2^N possible configurations 
so finding any one will be difficult!

https://lewiscoleblog.com/spin-glass-models-2
"""


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Fix random seed
np.random.seed(123)

# Set size of model N
N = 1000

# Fix number of timesteps and some containers
timesteps = 10000
gibbs = np.zeros(timesteps)
energy = np.zeros(timesteps)
mag = np.zeros(timesteps)

# Initialize interaction array
s = 1
interaction = np.zeros((N, N))
for i in range(N):
    for j in range(i):
        interaction[i, j] = np.random.randn() * s / np.sqrt(N)
        interaction[j, i] = interaction[i, j]

# Fix Temperature for Gibbs distribution
beta = 1/(s*0.5)

for i in range(timesteps):
    configuration = np.random.choice([-1, 1], N)
    energy[i] = -1 * np.dot(configuration, np.dot(configuration, interaction)) / 2
    gibbs[i] = np.exp(-beta*energy[i])
    mag[i] = configuration.sum()

print("Estimated Ground State Energy: ", energy.min())
print("Estimated Average Energy:", energy.mean())
print("Estimated Partition Function:", gibbs.mean())
print("Estimated Average Magnetism:", mag.mean())

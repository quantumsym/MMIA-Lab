#!/usr/bin/env python3
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from  MonteCarlo import *

# Parametri
rho = 0.75
num_samples = 20000

# Genera campioni con Gibbs
gibbs_samples = gibbs_sampling_bivariate_normal(rho, num_samples)

# Confronta con campionamento diretto
true_mean = [0, 0]
true_cov = [[1, rho], [rho, 1]]
direct_samples = np.random.multivariate_normal(true_mean, true_cov, num_samples)

# Print Stats
print_gibbs_stats(gibbs_samples)
print_bivariate_normal_stats(true_mean,true_cov)

# Visualization
plot_gibbs_samples(gibbs_samples)
plot_bivariate_normal_samples(direct_samples)


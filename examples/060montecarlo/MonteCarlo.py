#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#----------------------------------------------------------------
def gibbs_sampling_bivariate_normal(rho, num_samples=10000, burn_in=1000):
    """
    Gibbs sampling for bivariate normal distribution.

    This function implements the Gibbs sampling algorithm to generate samples from 
    a bivariate normal distribution with zero means, unit variances, and correlation coefficient rho.

    The algorithm alternates between sampling from conditional distributions:
    - X|Y ~ N(rho*Y, sqrt(1-rho²))
    - Y|X ~ N(rho*X, sqrt(1-rho²))

    Parameters:
    -----------
    rho : float
        Correlation coefficient between X and Y variables (-1 ≤ rho ≤ 1)
    num_samples : int, default=10000
        Number of samples to generate after burn-in period
    burn_in : int, default=1000
        Number of initial samples to discard to allow the chain to reach stationarity

    Returns:
    --------
    numpy.ndarray
        Array of shape (num_samples, 2) containing the generated sample pairs [x, y]
    """
    # Initialization
    samples = []
    x, y = 0.0, 0.0  # initial values for the Markov chain

    # Parameters for conditional distributions
    # Standard deviation for conditional distributions derived from bivariate normal properties
    conditional_std = np.sqrt(1 - rho**2)

    # Sampling process
    for i in range(num_samples + burn_in):
        # Sample x conditional on current y value
        # X|Y=y ~ N(rho*y, sqrt(1-rho²))
        x = np.random.normal(rho * y, conditional_std)

        # Sample y conditional on new x value
        # Y|X=x ~ N(rho*x, sqrt(1-rho²))
        y = np.random.normal(rho * x, conditional_std)

        # Save the sample (after burn-in period)
        if i >= burn_in:
            samples.append([x, y])

    return np.array(samples)


#----------------------------------------------------------------
def plot_gibbs_samples(gibbs_samples): 
    """
    Visualization of Gibbs sampling results.

    Creates a scatter plot showing the joint distribution of samples generated
    by the Gibbs sampling algorithm.

    Parameters:
    -----------
    gibbs_samples : numpy.ndarray
        Array of shape (n_samples, 2) containing sample pairs from Gibbs sampling
    """
    # Visualization of Gibbs samples
    plt.figure(figsize=(5, 5))
    plt.scatter(gibbs_samples[:, 0], gibbs_samples[:, 1], alpha=0.6, s=1)
    plt.title('Gibbs Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gibbs_samples.svg")
    plt.show()


#----------------------------------------------------------------
def plot_bivariate_normal_samples(direct_samples):
    """
    Visualization of direct bivariate normal samples.

    Creates a scatter plot showing samples drawn directly from a bivariate normal
    distribution (for comparison with Gibbs sampling results).

    Parameters:
    -----------
    direct_samples : numpy.ndarray
        Array of shape (n_samples, 2) containing sample pairs from direct sampling
    """
    # Visualization of Direct samples
    plt.figure(figsize=(5, 5))
    plt.scatter(direct_samples[:, 0], direct_samples[:, 1], alpha=0.6, s=1)
    plt.title('Bivariate Normal Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bivariate_normal_samples.svg")
    plt.show()

#----------------------------------------------------------------
def print_gibbs_stats(gibbs_samples):
    """
    Display statistical summary of Gibbs sampling results.

    Prints the empirical mean vector and covariance matrix computed from
    the Gibbs sampling output.

    Parameters:
    -----------
    gibbs_samples : numpy.ndarray
        Array of shape (n_samples, 2) containing sample pairs from Gibbs sampling
    """
    # Statistical comparison
    print("Gibbs Sampling Stats:")
    print(f"Mean: {np.mean(gibbs_samples, axis=0)}")
    print(f"Covariance:\n{np.cov(gibbs_samples.T)}")

#----------------------------------------------------------------
def print_bivariate_normal_stats(true_mean, true_cov):
    """
    Display theoretical statistics of the target bivariate normal distribution.

    Prints the true mean vector and covariance matrix of the bivariate normal
    distribution that we're trying to sample from.

    Parameters:
    -----------
    true_mean : array-like
        True mean vector of the bivariate normal distribution
    true_cov : array-like
        True covariance matrix of the bivariate normal distribution
    """
    print("\nTheoretical Stats (Bivariate Normal):")
    print(f"Mean: {true_mean}")
    print(f"Covariance:\n{np.array(true_cov)}")


#-----------------------------------------------------------------------------------
#
# Visualization of radioactive decay simulation and comparison with analytical solution
#
def plot_radioactive_decay(t, N, N0, tau):
    """
    Plot radioactive decay simulation results with analytical solution overlay.

    This function visualizes Monte Carlo simulation results for radioactive decay
    and compares them with the exact analytical solution N(t) = N0 * exp(-t/tau).

    Parameters:
    -----------
    t : numpy.ndarray
        Time points for the simulation
    N : numpy.ndarray  
        Number of atoms remaining at each time point (from simulation)
    N0 : float
        Initial number of atoms
    tau : float
        Mean lifetime of the radioactive atoms (time constant)
    """
    # Plot simulation points
    plt.plot(t, N, '.')
    plt.xlabel('Time')
    plt.ylabel('Remaining Atoms')
    plt.title('Monte Carlo Radioactive Decay')

    # Analytical solution
    # Exponential decay law: N(t) = N0 * exp(-t/tau)
    N_analytical = N0 * np.exp(-t/tau)

    # Plot analytical solution as dashed red line
    plt.plot(t, N_analytical, color='red', linestyle='--')
    plt.savefig("radioactive_decay.svg")
    plt.show()

#-----------------------------------------------------------------------------------

def plot_metropolis_normal(samples, acceptance_rate, n_samples):
    """
    Visualization of Metropolis-Hastings sampling results for normal distribution.

    Creates two subplots:
    1. Trace plot showing the evolution of sample values over iterations
    2. Histogram comparing sampled distribution with theoretical normal distribution

    Note: This function references 'acceptance_rate' which should be defined 
    in the calling scope or passed as a parameter.

    Parameters:
    -----------
    samples : numpy.ndarray
        Array of samples generated by Metropolis-Hastings algorithm
    """
    # Visualization of the simulation
    plt.figure(figsize=(12, 5))

    # Trace plot (n_samples)
    plt.subplot(1, 2, 1)
    plt.plot(samples[:n_samples])
    plt.title(f'Trace Plot ( {n_samples} Samples)\nAcceptance Rate: {acceptance_rate:.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('Sampled Value')

    # Histogram comparison with theoretical distribution
    plt.subplot(1, 2, 2)
    plt.hist(samples[100:], bins=50, density=True, alpha=0.7, label='MCMC Samples')

    # Theoretical standard normal distribution N(0,1)
    x_theory = np.linspace(-4, 4, 100)
    y_theory = np.exp(-x_theory**2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x_theory, y_theory, 'r-', linewidth=2, label='Theoretical Distribution N(0,1)')

    plt.legend()
    plt.title('Sample Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig("metropolis_normal.svg")
    plt.show()

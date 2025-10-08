#!/usr/bin/env python
#
"""
Monte Carlo Estimation of π using the Gaussian Bell Surface

This script demonstrates how to estimate π using the Monte Carlo method
by exploiting the mathematical fact that the volume under the surface
z = exp(-x² - y²) over the entire xy-plane equals π.

Author: Python Monte Carlo Example
Mathematical background:
∫∫_{-∞}^{∞} exp(-x² - y²) dx dy = π
"""

import numpy as np

def monte_carlo_pi_gaussian(n_samples=10000000, sampling_radius=4.0):
    """
    Estimate π using Monte Carlo integration of the Gaussian bell surface.

    This function exploits the mathematical property that the integral
    of exp(-x² - y²) over the entire xy-plane equals π.

    Parameters:
    -----------
    n_samples : int, optional
        Number of random sample points to generate (default: 1,000,000)
    sampling_radius : float, optional
        Half-width of the square sampling region [-R, R]×[-R, R] (default: 4.0)

    Returns:
    --------
    float
        Monte Carlo approximation of π

    Mathematical Method:
    -------------------
    1. Generate random points (x, y) uniformly in [-R, R]×[-R, R]
    2. Evaluate z = exp(-x² - y²) at each point
    3. Use Monte Carlo integration: Integral ≈ Area × Average(z)
    4. The result approximates π since ∫∫ exp(-x² - y²) dx dy = π
    """

    # Generate random points uniformly distributed in the sampling square
    x_random = np.random.uniform(-sampling_radius, sampling_radius, n_samples)
    y_random = np.random.uniform(-sampling_radius, sampling_radius, n_samples)

    # Evaluate the Gaussian bell function at each random point
    # z = exp(-x² - y²) represents the height of the bell surface
    z_values = np.exp(-x_random**2 - y_random**2)

    # Apply Monte Carlo integration formula
    # Integral ≈ (Area of sampling region) × (Average function value)
    sampling_area = (2 * sampling_radius)**2
    average_height = np.mean(z_values)

    # The estimated integral should approximate π
    pi_estimate = sampling_area * average_height

    return pi_estimate

def demonstrate_convergence():
    """
    Demonstrate how the Monte Carlo estimate improves with more samples.
    Tests different sample sizes to show convergence behavior.
    """

    print("Monte Carlo π Estimation - Convergence Analysis")
    print("=" * 55)
    print()
    print("Mathematical principle:")
    print("∫∫_{-∞}^{∞} exp(-x² - y²) dx dy = π")
    print()

    # Test different sample sizes
    sample_sizes = [1000, 10000, 100000, 1000000, 5000000, 50000000, 100000000]
    true_pi = np.pi

    print(f"{'Samples':<10} {'π Estimate':<12} {'Error':<10} {'Error %':<8}")
    print("-" * 45)

    for n in sample_sizes:
        pi_est = monte_carlo_pi_gaussian(n)
        error = abs(pi_est - true_pi)
        error_percent = (error / true_pi) * 100

        print(f"{n:<10,} {pi_est:<12.6f} {error:<10.6f} {error_percent:<8.4f}")

    print()
    print(f"True value of π: {true_pi:.10f}")

def compare_sampling_regions():
    """
    Show how the choice of sampling region affects accuracy.
    Larger regions capture more of the infinite integral.
    """

    print("\nSampling Region Analysis")
    print("-" * 25)
    print("Effect of sampling region size on accuracy:")
    print()

    radii = [2.0, 3.0, 4.0, 5.0, 6.0]
    n_samples = 1000000
    true_pi = np.pi

    print(f"{'Region':<15} {'π Estimate':<12} {'Error':<10}")
    print("-" * 40)

    for R in radii:
        pi_est = monte_carlo_pi_gaussian(n_samples, R)
        error = abs(pi_est - true_pi)
        print(f"[-{R}, {R}]²{' ':<6} {pi_est:<12.6f} {error:<10.6f}")

if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_convergence()
    compare_sampling_regions()

    print("\n" + "="*55)
    print("Summary:")
    print("- The Monte Carlo method approximates π by random sampling")
    print("- Accuracy improves with more sample points (√n convergence)")
    print("- Larger sampling regions better approximate the infinite integral")
    print("- This method showcases the connection between geometry and probability")

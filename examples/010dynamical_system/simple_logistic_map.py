#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
r = 3.94          # Growth rate parameter (must be between 0 and 4)
x0 = 0.5         # Initial population value (between 0 and 1)
n_steps = 100    # Number of iterations to simulate


def simple_logistic_map():
    """
    Simplest possible logistic map simulation
    The logistic map equation: x_n+1 = r * x_n * (1 - x_n)

    This function demonstrates chaotic behavior in population dynamics
    """
    # Create arrays to store the results
    time = np.arange(n_steps)           # Time steps: 0, 1, 2, ..., n_steps-1
    population = np.zeros(n_steps)      # Array to store population values

    # Set the initial condition
    population[0] = x0

    # Main simulation loop - iterate the logistic map equation
    for i in range(1, n_steps):
        # Apply the logistic map formula: x_n+1 = r * x_n * (1 - x_n)
        population[i] = r * population[i-1] * (1 - population[i-1])

    return time, population


def plot_results(time, population):
    # Create the visualization
    plt.figure(figsize=(10, 6))        # Set the figure size
    plt.plot(time, population, 'b.-', markersize=4, linewidth=1)
    plt.title(f'Logistic Map: Population Evolution (r = {r})')
    plt.xlabel('Time Step (n)')
    plt.ylabel('Population (x_n)')
    plt.grid(True, alpha=0.3)          # Add a light grid for readability
    plt.show()

    return


if __name__ == "__main__":
    
    # Run the simulation
    time_steps, population = simple_logistic_map()
    
    # Print results
    print(f"Growth parameter r = {r}")
    print(f"Initial population = {x0}")
    print(f"Final population after {n_steps-1} steps = {population[-1]:.6f}")
    
    # plot simulation
    
    plot_results(time_steps,population)
    
    exit
    
    

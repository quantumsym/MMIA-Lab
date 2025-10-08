#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Initial conditions: starting populations
prey_initial = 30      # initial number of prey (e.g., rabbits)
predator_initial = 15   # initial number of predators (e.g., foxes)
initial_populations = [prey_initial, predator_initial]

# Time parameters
time_start = 0     # start time
time_end = 12      # end time (years)

# Model parameters (these control the dynamics)
alpha = 1.0    # prey birth rate (natural growth without predators)
beta = 0.1     # predation rate (how often prey gets eaten)
gamma = 1.5    # predator death rate (natural death without prey)
delta = 0.075  # predator efficiency (how many new predators per prey eaten)



###---------------------------------------

def lotka_volterra_system(t, populations):
    """
    Lotka-Volterra predator-prey model differential equations.

    The system models the interaction between prey and predator populations:
    - dx/dt = alpha*x - beta*x*y    (prey equation)
    - dy/dt = -gamma*y + delta*x*y  (predator equation)

    Args:
        t : float
            Current time (not used explicitly but required by solve_ivp)
        populations : array_like, shape (2,)
            Current populations [prey, predator]

    Returns:
        derivatives : list
            [dx/dt, dy/dt] - population change rates
    """
    # Unpack the current populations
    prey, predator = populations

    # Calculate the derivatives using Lotka-Volterra equations
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = -gamma * predator + delta * prey * predator

    return [dprey_dt, dpredator_dt]

    
def plot_volterra_lotka(time_array,prey_population,predator_population):
    # Create visualization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time series showing population evolution
    ax1.plot(time_array, prey_population, 'b-', linewidth=2, label='Prey (Rabbits)')
    ax1.plot(time_array, predator_population, 'r-', linewidth=2, label='Predator (Foxes)')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Population')
    ax1.set_title('Lotka-Volterra Predator-Prey Model: Population vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase space plot (predator vs prey)
    ax2.plot(prey_population, predator_population, 'g-', linewidth=2, alpha=0.8)
    ax2.plot(prey_initial, predator_initial, 'ro', markersize=8, label='Initial Point')
    ax2.plot(prey_population[-1], predator_population[-1], 'bs', markersize=8, label='Final Point')
    ax2.set_xlabel('Prey Population')
    ax2.set_ylabel('Predator Population')
    ax2.set_title('Phase Space: Predator vs Prey')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return

    
def print_basic_stats(prey_population,predator_population):
    # Print some basic statistics
    print(f"\nPopulation Statistics:")
    print(f"Initial prey: {prey_initial}, Initial predator: {predator_initial}")
    print(f"Maximum prey: {np.max(prey_population):.2f}")
    print(f"Minimum prey: {np.min(prey_population):.2f}")
    print(f"Maximum predator: {np.max(predator_population):.2f}")
    print(f"Minimum predator: {np.min(predator_population):.2f}")
    return


###-----------------------------------------------------------

if __name__ == "__main__":

    time_points = np.linspace(time_start, time_end, 1000)  # evaluation points
    
    # Solve the differential equation system using solve_ivp
    solution = solve_ivp(
        lotka_volterra_system,        # our ODE system function
        (time_start, time_end),       # time span for integration
        initial_populations,          # initial conditions
        t_eval=time_points,           # specific time points to evaluate
        method='RK45',                # Runge-Kutta method (default, works well)
        rtol=1e-6                     # relative tolerance for accuracy
    )
    
    # Extract the solutions for plotting
    time_array = solution.t           # time points
    prey_population = solution.y[0]   # prey population over time
    predator_population = solution.y[1]  # predator population over time
    
    # Check if integration was successful
    print(f"Integration successful: {solution.success}")
    print(f"Integration message: {solution.message}")
    
    
    # Plotting and printing results
    plot_volterra_lotka(time_array,prey_population,predator_population)
    print_basic_stats(prey_population,predator_population)
    
    exit

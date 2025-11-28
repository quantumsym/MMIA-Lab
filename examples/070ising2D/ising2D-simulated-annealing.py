#!/usr/bin/env python3
#
import time
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *

# Main parameters
L = 50               # Lattice dimension (50x50)
T_initial = 5.0      # Initial temperature (high)
T_final = 0.5        # Final temperature (low)
num_steps = 2000     # Number of evolution steps
equilibration = 200  # Number of thermal equilibrium steps
p = 0.5              # probability of spin −1
J = 1.0              # coupling constant
seed = 42            # fixed rng seed
cooling_rate = 0.999 # Cooling rate (α)

if __name__ == "__main__":

    # Initialize lattice and analyze
    lattice = initialize_lattice(L, p)

    # Initial temperature for simulated annealing
    T_current = T_initial


    print_sa_system_info(L,T_initial,T_final,cooling_rate,num_steps)

    # Lists to store observables over time
    magnetization = []
    energy = []
    temperature_history = []

    # Performance comparison
    start = time.time()

    # System evolution with simulated annealing
    for step in range(num_steps):
        # Update the system at the current temperature
        lattice = simulated_annealing_step(lattice, J, T_current)

        # Cool the system using the exponential cooling schedule
        # T_current = sa_exponential_cooling_schedule(T_current, cooling_rate)

        # Alternative: use linear cooling schedule
        T_current = sa_linear_cooling_schedule(step, num_steps, T_initial, T_final)

        # After a certain number of equilibrium steps, record the observables
        if step >= equilibration:
            mag = compute_average_magnetization(lattice)
            magnetization.append(mag)
            E_avg = calculate_total_energy(lattice, J) / lattice.size
            energy.append(E_avg)
            temperature_history.append(T_current)

        if step % 100 == 0:
            print(f"Step: {step}, Temperature: {T_current:.4f}")

    print(f"Step: {num_steps}, Final Temperature: {T_current:.4f}")
    time1 = time.time() - start
    print(f"Execution time: {time1:.2f} seconds")

    # Plot of the observables
    plot_magnetization(magnetization, f"SA (T: {T_initial}→{T_final})")
    plot_energy(energy, f"SA (T: {T_initial}→{T_final})")

    plot_sa_temperature_step(equilibration,num_steps,temperature_history)

    # Set plot title
    plot_title = f"2D Ising Model {L}×{L} — Simulated Annealing (T: {T_initial}→{T_final:.2f})"
    # Visualize final configuration
    plot_spin_configuration(lattice, title=plot_title)

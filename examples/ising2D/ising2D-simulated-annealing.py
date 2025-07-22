#!/usr/bin/env python3
#
import time
import numpy as np
import matplotlib.pyplot as plt
from IsingModel2D import *

# Parametri principali
L = 50               # Dimensione del lattice (50x50)
T_initial = 5.0      # Temperatura iniziale (alta)
T_final = 0.5        # Temperatura finale (bassa)
num_steps = 2000     # Numero di passi di evoluzione
equilibration = 200  # Numero di passi di equilibrio termico
p = 0.5              # probability of spin −1
J = 1.0              # coupling constant
seed = 42            # fixed rng seed
cooling_rate = 0.999 # Tasso di raffreddamento (α)

if __name__ == "__main__":

    # Initialize lattice and analyze
    lattice = initialize_lattice(L, p)

    # Temperatura iniziale per simulated annealing
    T_current = T_initial


    print_sa_system_info(L,T_initial,T_final,cooling_rate,num_steps)

    # Liste per memorizzare osservabili nel tempo
    magnetization = []
    energy = []
    temperature_history = []

    # Performance comparison
    start = time.time()

    # Evoluzione del sistema con simulated annealing
    for step in range(num_steps):
        # Aggiorna il sistema alla temperatura corrente
        lattice = simulated_annealing_step(lattice, J, T_current)

        # Raffredda il sistema usando il cooling schedule esponenziale
        T_current = sa_exponential_cooling_schedule(T_current, cooling_rate)

        # Alternative: usa cooling schedule lineare
        # T_current = sa_linear_cooling_schedule(step, num_steps, T_initial, T_final)

        # Dopo un certo numero di passi di equilibrio, registra le osservabili
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

    # Plot delle osservabili
    plot_magnetization(magnetization, f"SA (T: {T_initial}→{T_final})")
    plot_energy(energy, f"SA (T: {T_initial}→{T_final})")

    plot_sa_temperature_step(equilibration,num_steps,temperature_history)

    # Set plot title
    plot_title = f"2D Ising Model {L}×{L} — Simulated Annealing (T: {T_initial}→{T_final:.2f})"
    # Visualize final configuration
    plot_spin_configuration(lattice, title=plot_title)


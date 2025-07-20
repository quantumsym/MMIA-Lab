#!/usr/bin/env python3
"""
2D Ising Model - Modular version with local bond energy function
Author: S.Magrì <s.magri@quantumsym.com>  luglio 2025
"""
### Load NumPy
import numpy as np
###  Load MatPlotLib
import matplotlib.pyplot as plt
###  Uncomment to display graphs in a Jupyter Notebook
#%matplotlib inline
###  Uncomment to import/process/export data with Pandas
#import pandas as pd
###  Uncomment to optimize CPU execution with Numba
from numba import njit

#---------------------------------------------------
# Test parameters
#L = 100      # lattice side length
#p = 0.1     # probability of spin −1
#J = 1.0     # coupling constant
#seed = 123  # fixed rng seed

# ------------------------------------------------------------------
# 0. Global Random Number Generator with reproducible seed
# ------------------------------------------------------------------
#rng = np.random.default_rng(seed)      # Generator instance with fixed seed


# ------------------------------------------------------------------
# 1. Lattice initialization
# ------------------------------------------------------------------
def initialize_lattice(seed: int = 1234,L: int = 50, p: float = 0.5) -> np.ndarray:
    """
    Generate L×L lattice of ±1 spins with probabilities:
      - p     for spin −1
      - 1−p   for spin +1
    Uses rng.choice for weighted sampling.
    """
    rng = np.random.default_rng(seed)      # Generator instance with fixed seed
    # array of possible states and their probabilities
    states = np.array([-1, 1], dtype=int)
    probs  = np.array([p, 1.0 - p])
    lattice = rng.choice(states, size=(L, L), p=probs)
    return lattice


# ------------------------------------------------------------------
# 2A. Local bond energy right & bottom (periodic boundary conditions)
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def local_bond_energy_rb(spins: np.ndarray, i: int, j: int,
                         J: float = 1.0) -> float:
    """
    Returns the energy contribution of two bonds:
      - (i, j) with right neighbor
      - (i, j) with bottom neighbor
    Implements periodic boundary conditions.
    Formula: E_rb = −J * s_ij * (s_right + s_bottom)
    """
    L  = spins.shape[0]
    s  = spins[i, j]
    s_right  = spins[i, (j + 1) % L]
    s_bottom = spins[(i + 1) % L, j]
    neighbors_rb = s_right + s_bottom   # Sum of right and bottom neighbors

    return -J * s * neighbors_rb

# ------------------------------------------------------------------
# 2B. Local bond energy all neighbors (periodic boundary conditions)
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def local_bond_energy_all(spins: np.ndarray, i: int, j: int,
                          J: float = 1.0) -> float:
    """
    Returns the energy contribution of all four bonds:
      - (i, j) with left neighbor
      - (i, j) with right neighbor
      - (i, j) with top neighbor
      - (i, j) with bottom neighbor
    Implements periodic boundary conditions.
    Formula: E_all = −J * s_ij * (s_left + s_right + s_top + s_bottom)
    """
    L = spins.shape[0]
    s = spins[i, j]

    # All four nearest neighbors with periodic boundary conditions
    s_left   = spins[i, (j - 1) % L]  # Left neighbor
    s_right  = spins[i, (j + 1) % L]  # Right neighbor
    s_top    = spins[(i - 1) % L, j]  # Top neighbor
    s_bottom = spins[(i + 1) % L, j]  # Bottom neighbor
    neighbors_sum = s_left + s_right + s_top + s_bottom   # Sum of neighbors

    return -J * s * neighbors_sum


# ------------------------------------------------------------------
# 3. Total energy calculation
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def calculate_total_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Calculate total energy by summing right+bottom contributions
    for each site (avoids double counting).
    """
    L = spins.shape[0]
    energy = 0.0
    for i in range(L):
        for j in range(L):
            energy += local_bond_energy_rb(spins, i, j, J)
    return energy


# ------------------------------------------------------------------
# 4. Average energy per spin
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def calculate_average_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Average energy per spin (= E_total / N_spins).
    """
    return calculate_total_energy(spins, J) / spins.size


# ------------------------------------------------------------------
# 5A. Total Magnetization calculation
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def compute_total_magnetization(spins: np.ndarray) -> float:
    """
    Returns total_magnetization
    """
    return spins.sum()

# ------------------------------------------------------------------
# 5B. Average Magnetization calculation
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def compute_average_magnetization(spins: np.ndarray) -> float:
    """
    Returns  average_magnetization_per_spin.
    """
    M_tot = spins.sum()
    size = spins.size
    return M_tot / size

# ------------------------------------------------------------------
# 6. Lattice visualization
# ------------------------------------------------------------------
def plot_spin_configuration(spins: np.ndarray, title: str | None = None,
                            cmap: str = "bwr") -> None:
    """
    Display the lattice using imshow.
    Spin +1 → red, spin −1 → blue.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(spins, cmap=cmap, interpolation="nearest",
               vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 1], label="Spin")

    if title:
        plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 7. System information display
# ------------------------------------------------------------------
def print_system_info(spins: np.ndarray, J: float, p: float) -> None:
    """
    Display main statistics of the lattice.
    """
    L = spins.shape[0]
    E_tot = calculate_total_energy(spins, J)
    E_avg = E_tot / spins.size
    M_tot = compute_total_magnetization(spins)
    M_avg = compute_average_magnetization(spins)

    print(f"Lattice size          : {L} × {L}")
    print(f"Probability spin −1   : {p:.3f}")
    print(f"Coupling constant J   : {J}")
    print("-" * 30)
    print(f"Total energy          : {E_tot: .3f}")
    print(f"Average energy per spin: {E_avg: .3f}")
    print(f"Total magnetization   : {M_tot: .3f}")
    print(f"Average magnetization : {M_avg: .3f}")
    print("-" * 30)



# ------------------------------------------------------------------
# 8. Plot average   magnetization over  time
# ------------------------------------------------------------------
def plot_magnetization(magnetizations,temperature):
    """ 
    Plot of average magnetization over time
    """
    plt.ylim(-1.2,+1.2)
    plt.plot(magnetizations)
    plt.xlabel('Numero di passi')
    plt.ylabel('Magnetizzazione assoluta')
    plt.title(f'Simulazione Ising 2D a T={temperature}')
    plt.show()


# ------------------------------------------------------------------
# 9. Metropolis spin update
# ------------------------------------------------------------------
@njit(cache=True,fastmath=True)
def metropolis_spin_update(spins,i,j,J,T):
    # Calcola la variazione di energia se si flippa lo spin
    # -2 for local site energy
    delta_E = -2 * local_bond_energy_all(spins,i,j,J)

    # Accetta il flip se diminuisce energia o con probabilità Boltzmann
    if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
        spins[i, j] *= -1  # Flip dello spin

    return spins



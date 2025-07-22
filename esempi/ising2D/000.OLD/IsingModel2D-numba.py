#!/usr/bin/env python3
"""
2D Ising Model - Numba optimized version
Author: <your name>
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ------------------------------------------------------------------
# 0. Global RNG with reproducible seed
# ------------------------------------------------------------------
rng = np.random.default_rng(123)

# ------------------------------------------------------------------
# 1. Lattice initialization (optimized with Numba)
# ------------------------------------------------------------------
@njit(cache=True)
def _initialize_lattice_core(L: int, random_values: np.ndarray, p: float) -> np.ndarray:
    """
    Core initialization function optimized with Numba.
    Separates random generation from actual computation.
    """
    lattice = np.empty((L, L), dtype=np.int32)
    for i in range(L):
        for j in range(L):
            lattice[i, j] = -1 if random_values[i, j] < p else 1
    return lattice

def initialize_lattice(L: int, p: float = 0.5) -> np.ndarray:
    """
    Generate L×L lattice of ±1 spins with probabilities:
      - p     for spin −1
      - 1−p   for spin +1
    """
    # Generate random values outside of njit function
    random_values = rng.random((L, L))
    return _initialize_lattice_core(L, random_values, p)

# ------------------------------------------------------------------
# 2. Local bond energy right & bottom (optimized with Numba)
# ------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def local_bond_energy_rb(spins: np.ndarray, i: int, j: int, J: float = 1.0) -> float:
    """
    Returns the energy contribution of two bonds:
      - (i, j) with right neighbor
      - (i, j) with bottom neighbor
    Implements periodic boundary conditions.
    """
    L = spins.shape[0]
    s = spins[i, j]
    s_right = spins[i, (j + 1) % L]
    s_bottom = spins[(i + 1) % L, j]
    return -J * s * (s_right + s_bottom)

# ------------------------------------------------------------------
# 3. Total energy calculation (optimized with Numba)
# ------------------------------------------------------------------
@njit(cache=True, fastmath=True)
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
# 4. Average energy per spin (optimized with Numba)
# ------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def calculate_average_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Average energy per spin (= E_total / N_spins).
    """
    return calculate_total_energy(spins, J) / spins.size

# ------------------------------------------------------------------
# 5. Magnetization calculation (optimized with Numba)
# ------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def calculate_magnetization(spins: np.ndarray) -> tuple[float, float]:
    """
    Returns (total_magnetization, average_magnetization_per_spin).
    """
    M_tot = 0.0
    for i in range(spins.shape[0]):
        for j in range(spins.shape[1]):
            M_tot += spins[i, j]
    return M_tot, M_tot / spins.size

# ------------------------------------------------------------------
# 6. Lattice visualization (NOT optimized - uses matplotlib)
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
# 7. System information display (NOT optimized - I/O operations)
# ------------------------------------------------------------------
def print_system_info(spins: np.ndarray, J: float, p: float) -> None:
    """
    Display main statistics of the lattice.
    """
    L = spins.shape[0]
    E_tot = calculate_total_energy(spins, J)
    E_avg = E_tot / spins.size
    M_tot, M_avg = calculate_magnetization(spins)

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
# 8. Alternative vectorized energy calculation (for comparison)
# ------------------------------------------------------------------
#@njit(cache=True, fastmath=True)
@njit(cache=True)
def calculate_total_energy_vectorized(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Vectorized version using numpy operations within Numba.
    Often faster for larger lattices.
    """
    # Horizontal interactions
    energy = -J * np.sum(spins * np.roll(spins, -1, axis=1))
    # Vertical interactions  
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=0))
    return energy

# ------------------------------------------------------------------
# Example execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Test parameters
    L = 50      # larger lattice to see performance benefits
    p = 0.3     # probability of spin −1
    J = 1.0     # coupling constant
    
    print("Initializing lattice...")
    lattice = initialize_lattice(L, p)
    
    print("Calculating system properties...")
    print_system_info(lattice, J, p)
    
    # Performance comparison
    import time
    
    # Test energy calculation methods
    start = time.time()
    E1 = calculate_total_energy(lattice, J)
    time1 = time.time() - start
    
    start = time.time()
    E2 = calculate_total_energy_vectorized(lattice, J)
    time2 = time.time() - start
    
    print(f"\nPerformance comparison:")
    print(f"Loop method:       {time1:.6f}s, Energy: {E1:.3f}")
    print(f"Vectorized method: {time2:.6f}s, Energy: {E2:.3f}")
    print(f"Speedup: {time1/time2:.2f}x")
    
    # Visualize configuration
    plot_spin_configuration(lattice, title=f"2D Ising Model {L}×{L} — p={p}, J={J}")


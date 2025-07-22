#!/usr/bin/env python
#
# https://viadean.notion.site/A-spin-glass-on-a-square-lattice-using-the-Ising-model-1d01ae7b9a328001a02ac279e2674e85
#
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# setup random number generator
rng = np.random.default_rng()


def initialize_lattice(L):
    """Initialize a random spin lattice of size LxL."""
    spins = rng.choice([-1, 1], size=(L, L))
    return spins

def initialize_interactions(L):
    """Initialize random interaction strengths (J) between nearest neighbors."""
    J_h = rng.choice([-1, 1], size=(L, L))  # Horizontal interactions
    J_v = rng.choice([-1, 1], size=(L, L))  # Vertical interactions
    return J_h, J_v

@njit
def calculate_energy(spins, J_h, J_v):
    """Calculate the total energy of the system."""
    L = spins.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            # Add interaction energies with right and down neighbors
            energy -= J_h[i, j] * S * spins[i, (j + 1) % L]  # Right neighbor
            energy -= J_v[i, j] * S * spins[(i + 1) % L, j]  # Down neighbor
    return energy

@njit
def metropolis_step(spins, J_h, J_v, T):

    """Perform a single Metropolis step."""
    L = spins.shape[0]
    for _ in range(L * L):  # Attempt to flip each spin once on average
        i, j = np.random.randint(0, L, size=2)
        S = spins[i, j]
        # Energy change if the spin is flipped
        dE = 2 * S * (
            J_h[i, j] * spins[i, (j + 1) % L] +  # Right neighbor
            J_h[i, (j - 1) % L] * spins[i, (j - 1) % L] +  # Left neighbor
            J_v[i, j] * spins[(i + 1) % L, j] +  # Down neighbor
            J_v[(i - 1) % L, j] * spins[(i - 1) % L, j]  # Up neighbor
        )

        # Metropolis acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1  # Flip the spin

def simulate(L, T, steps):
    """Run the spin glass simulation."""
    spins = initialize_lattice(L)
    J_h, J_v = initialize_interactions(L)
    energies = []

    for step in range(steps):
        metropolis_step(spins, J_h, J_v, T)
        energy = calculate_energy(spins, J_h, J_v)
        energies.append(energy)

        # Visualization every 200 steps
        if step % 200 == 0:
            plot_spins(spins,step)

    return spins, energies


def plot_spins(spins,step):
    plt.imshow(spins, cmap="coolwarm")
    plt.title(f"Step {step}")
    plt.show()


def plot_energies(energies):
    # Plot energy over time
    plt.plot(energies)
    plt.title("Energy vs. Steps")
    plt.xlabel("Steps")
    plt.ylabel("Energy")
    plt.show()


if __name__ == "__main__":
    L = 50  # Lattice size
    T = 1.0  # Temperature
    steps = 2000  # Number of simulation steps

    final_spins, energies = simulate(L, T, steps)

    plot_energies(energies)




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

#---------------------------------------------------
# Test parameters
#L = 100      # lattice side length
#p = 0.1     # probability of spin −1
#J = 1.0     # coupling constant
seed = 123  # fixed rng seed

# ------------------------------------------------------------------
# 0. Global Random Number Generator with reproducible seed
# ------------------------------------------------------------------
rng = np.random.default_rng(seed)      # Generator instance with fixed seed


# ------------------------------------------------------------------
# 1. Lattice initialization
# ------------------------------------------------------------------
def initialize_lattice(L: int = 50, p: float = 0.5) -> np.ndarray:
    """
    Generate L×L lattice of ±1 spins with probabilities:
      - p     for spin −1
      - 1−p   for spin +1
    Uses rng.choice for weighted sampling.
    """
    # array of possible states and their probabilities
    states = np.array([-1, 1], dtype=int)
    probs  = np.array([p, 1.0 - p])
    lattice = rng.choice(states, size=(L, L), p=probs)
    return lattice

# ------------------------------------------------------------------
# 2. Local bond energy right & bottom (periodic boundary conditions)
# ------------------------------------------------------------------
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
# 3. Local bond energy all neighbors (periodic boundary conditions)
# ------------------------------------------------------------------
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
# 4. Local field  (periodic boundary conditions)
# -----------------------------------------------------------------


def calculate_local_field(spins, i, j, J, B):
    """
    Calcola il campo locale per lo spin in posizione (i,j)
    """
    L = spins.shape[0]

    # Condizioni al contorno periodiche
    neighbors = (
        spins[(i-1) % L, j] +
        spins[(i+1) % L, j] +
        spins[i, (j-1) % L] +
        spins[i, (j+1) % L]
    )

    return J * neighbors + B


# ------------------------------------------------------------------
# 5. Local field, right and botton contribute (periodic boundary conditions)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 6. Local field, left and top contribute (periodic boundary conditions)
# -----------------------------------------------------------------


# ------------------------------------------------------------------
# 7. Energy change for spin flip
# ------------------------------------------------------------------
def delta_energy_spin_flip(spins: np.ndarray, i: int , j: int, J: float = 1.0) -> float:
    return - 2 * local_bond_energy_all(spins, i, j,J )

# ------------------------------------------------------------------
# 8. Total energy calculation
# ------------------------------------------------------------------
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
# 9. Average energy per spin
# ------------------------------------------------------------------
def calculate_average_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Average energy per spin (= E_total / N_spins).
    """
    return calculate_total_energy(spins, J) / spins.size


# ------------------------------------------------------------------
#  10. Total Magnetization calculation
# ------------------------------------------------------------------
def compute_total_magnetization(spins: np.ndarray) -> float:
    """
    Returns total_magnetization
    """
    return spins.sum()

# ------------------------------------------------------------------
# 11. Average Magnetization calculation
# ------------------------------------------------------------------
def compute_average_magnetization(spins: np.ndarray) -> float:
    """
    Returns  average_magnetization_per_spin.
    """
    M_tot = spins.sum()
    size = spins.size
    return M_tot / size

# ------------------------------------------------------------------
# 12. Lattice visualization
# ------------------------------------------------------------------
def plot_spin_configuration(spins: np.ndarray, title: str | None = None,
        cmap: str = "Greys") -> None:
    """
    Display the lattice using imshow.
    Spin +1 → red, spin −1 → blue.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(spins, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1 )
    cbar = plt.colorbar(ticks=[-1, 1], label="Spin" )
    cbar.ax.set_yticklabels(["-1", "+1"])

    if title:
        plt.title(title)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('ising2D-spin-configuration.svg')
    plt.savefig('ising2D-spin-configuration.pdf')
    plt.show()

# ------------------------------------------------------------------
# 13. Display simulation parameter
# ------------------------------------------------------------------
def print_parameter( L: int, J: float = 1 , p: float = 0.50, T: float = 0.0, steps: int = 0) -> None:
    """
    Display simulation parameter
    """
    print(f"Lattice size          : {L} × {L}")
    print(f"Probability spin −1   : {p:.3f}")
    print(f"Coupling constant J   : {J}")
    print(f"Temperature           : {T}")
    print(f"Time steps            : {steps}")



# ------------------------------------------------------------------
# 14. System information display
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

    print("-" * 30)
    print(f"Total energy          : {E_tot: .3f}")
    print(f"Average energy per spin: {E_avg: .3f}")
    print(f"Total magnetization   : {M_tot: .3f}")
    print(f"Average magnetization : {M_avg: .3f}")

# ------------------------------------------------------------------
# 15. Plot average   magnetization over  time
# ------------------------------------------------------------------
def plot_magnetization(magnetizations,temperature):
    """ 
    Plot of average magnetization over time
    """
    plt.ylim(-1.2,+1.2)
    plt.plot(magnetizations)
    plt.xlabel('Time Steps')
    plt.ylabel('Magnetization')
    plt.title(f'Ising 2D  T={temperature}')
    plt.savefig('ising2D-magnetization.svg')
    plt.savefig('ising2D-magnetization.pdf')
    plt.show()



# ------------------------------------------------------------------
# 16. Plot average   energy over  time
# ------------------------------------------------------------------
def plot_energy(energy,temperature):
    """ 
    Plot of average energy over time
    """
    # plot energy evolving in time
    plt.plot(energy)
    plt.xlabel("Time step")
    plt.ylabel("Energy", color='red')
    plt.title(f'Ising 2D  T={temperature}')
    plt.savefig('ising2D-average-energy.svg')
    plt.savefig('ising2D-average-energy.pdf')
    plt.show()



# ------------------------------------------------------------------
# 17. Metropolis-Hastings spin update
# ------------------------------------------------------------------
def metropolis_spin_update(spins,i,j,J,T):
    # Calcola la variazione di energia se si flippa lo spin
    # -2 for local site energy
    delta_E = -2 * local_bond_energy_all(spins,i,j,J)

    # Accetta il flip se energia diminuisce 
    # oppure lo accetta con la distribuzione di probabilità di  Boltzmann
    if delta_E < 0:
        spins[i, j] *= -1  # Spin Flip
    else:
        P_Boltzmann = np.exp( - delta_E / T)   # relative probability Boltzmann distribution
        if rng.random() < P_Boltzmann:
            spins[i, j] *= -1  # Spin Flip

    return spins


# ------------------------------------------------------------------
# 18. Glauber Dynamics (Heath Bath)  spin update
# ------------------------------------------------------------------
def glauber_spin_update(spins,i,j,J,T):
    # Calcola la variazione di energia se si flippa lo spin
    # -2 for local site energy
    delta_E = -2 * local_bond_energy_all(spins,i,j,J)

    # Compute Fermi distribution probability
    P_Fermi  =  1.0 / (1.0 + np.exp(delta_E / T))

    if rng.random() < P_Fermi:
            spins[i, j] *= -1  # Spin Flip

    return spins


# ------------------------------------------------------------------
# 19. Single Glauber Update
# ------------------------------------------------------------------
def glauber_step(lattice):
    """Un singolo aggiornamento Glauber (heat-bath) su uno spin casuale."""
    i = rng.integers(L)
    j = rng.integers(L)
    dE = delta_E(lattice, i, j)
    if rng.random() < 1.0 / (1.0 + np.exp(dE / T)):
        lattice[i, j] *= -1




#----------------------------------------------------------------------
# 20.   Global Simulated Annealing Step 
#----------------------------------------------------------------------

def simulated_annealing_step(lattice, J, T):
    """
    Esegue un passo di simulated annealing su tutto il lattice
    La temperatura T viene gradualmente ridotta durante l'evoluzione
    """
    L = len(lattice)
    for _ in range(L**2):
        # Seleziona un sito casuale
        i, j = np.random.randint(0, L, 2)
        lattice = metropolis_spin_update(lattice, i, j, J, T)

    return lattice

#----------------------------------------------------------------------
# 21. Temperature Cooling linear schedule
#----------------------------------------------------------------------

def sa_linear_cooling_schedule(step, num_steps, T_initial, T_final):
    """
    Cooling schedule lineare: T decresce linearmente da T_initial a T_final
    """
    return T_initial - (T_initial - T_final) * step / num_steps

#----------------------------------------------------------------------
# 22. Temperature Cooling exponential schedule
#----------------------------------------------------------------------

def sa_exponential_cooling_schedule(T_current, cooling_rate):
    """
    Cooling schedule esponenziale: T_new = α * T_current
    """
    return T_current * cooling_rate


#----------------------------------------------------------------------
# 23. Display Simulated Annealing Parameters
#----------------------------------------------------------------------

def print_sa_system_info(L,T_initial,T_final,cooling_rate,num_steps):
    print(f"=== SIMULATED ANNEALING ===")
    print(f"Lattice size: {L}x{L}")
    print(f"Initial temperature: {T_initial}")
    print(f"Final temperature: {T_final}")
    print(f"Cooling rate: {cooling_rate}")
    print(f"Total steps: {num_steps}")
    print()


#----------------------------------------------------------------------
# 24.  Plot Simulated Annealing Temperature vs Time
#----------------------------------------------------------------------

def plot_sa_temperature_step(equilibration,num_steps,temperature_history):
    # Plot aggiuntivo: temperatura vs step
    plt.figure(figsize=(8, 6))
    plt.plot(range(equilibration, num_steps), temperature_history, 'r-', linewidth=2)
    plt.xlabel('Monte Carlo Step')
    plt.ylabel('Temperature')
    plt.title('Temperature Schedule - Simulated Annealing')
    plt.grid(True, alpha=0.3)
    plt.savefig('ising2D-annealing-temperature.svg')
    plt.savefig('ising2D-annealing-temperature.pdf')
    plt.show()

#----------------------------------------------------------------------
# 25. Wolff Cluster Spin Update
#----------------------------------------------------------------------

def wolff_cluster_update(spins, J, beta):
    L = spins.shape[0]
    to_flip = np.zeros_like(spins, dtype=bool)
    x, y = np.random.randint(L, size=2)
    seed_spin = spins[x, y]
    stack = [(x, y)]
    to_flip[x, y] = True
    p_add = 1 - np.exp(-2 * beta * J)
    while stack:
        i, j = stack.pop()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = (i+dx)%L, (j+dy)%L
            if not to_flip[ni, nj] and spins[ni, nj] == seed_spin:
                if np.random.rand() < p_add:
                    to_flip[ni, nj] = True
                    stack.append((ni, nj))
    spins[to_flip] *= -1

    return spins



#----------------------------------------------------------------------
# 25. Gibbs  Sampling
#----------------------------------------------------------------------

def gibbs_sampling(spins, i, j, J=1.0, B=0.0, beta= 1.0):
    # Calculate the local field
    h_local = calculate_local_field(spins, i, j, J, B)

    # Conditional probability for spin up
    P_up = 1.0 / (1.0 + np.exp(-2 * beta * h_local))

    # Sampling from the conditional distribution
    if rng.random() < P_up :
        spins[i, j] = 1
    else:
        spins[i, j] = -1

    return spins


#
#-----------------------------------------------------------------

"""
2D Ising Model - Modular version with local bond energy function
Author: S.Magrì <s.magri@quantumsym.com>  luglio 2025
"""
#------------------------------------------------------------------------
#
# Test parameters
L = 50      # lattice side length
p = 0.3      # probability of spin −1
J = 1.0      # coupling constant

# ------------------------------------------------------------------

if __name__ == "__main__":

    # Initialize lattice and analyze
    lattice = initialize_lattice(L, p)
    print_parameter(L, J, p)
    print_system_info(lattice, J, p)

    # Local right+bottom energy of a single site (example)
    i0, j0 = 5, 7
    e_rb = local_bond_energy_rb(lattice, i0, j0, J)
    print(f"Right+bottom energy of spin ({i0}, {j0}): {e_rb: .3f}\n")

    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — p={p}, J={J}"
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)


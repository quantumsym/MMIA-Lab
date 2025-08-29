#!/usr/bin/env python3
"""
SK (Sherrington-Kirkpatrick) Spin Glass Model - Mean Field Gaussian Approximation
==============================================================================

This script implements the SK spin glass model using the mean field approximation.
The SK model is a fully connected spin glass where each spin interacts with all others
through random Gaussian couplings.

Key Features:
- Continuous spins in [-1, 1] range
- Gaussian random coupling matrix J_ij ~ N(0, 1/√N)
- Mean field update rule: s_i = tanh(h_i / T)
- Fast convergence suitable for rapid exploration

Author: sandro@freenetst.it
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
N = 100                    # Number of spins in the system
T = 1.0                    # Temperature (in units where k_B = 1)
num_steps = 1000           # Total number of iteration steps
equilibration = 100        # Number of equilibration steps before data collection
seed = 42                  # Random seed for reproducibility

# Random number generator with fixed seed for reproducible results
rng = np.random.default_rng(seed)

# =============================================================================
# SYSTEM INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_sk_system(N):
    """
    Initialize the SK spin glass system.
    
    This function creates:
    1. Random initial spin configuration
    2. Symmetric Gaussian coupling matrix
    
    Parameters:
    -----------
    N : int
        Number of spins in the system
        
    Returns:
    --------
    spins : numpy.ndarray
        Initial spin configuration, shape (N,)
        Each spin is randomly ±1
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
        J_ij ~ N(0, 1/√N) for thermodynamic limit
        Symmetric: J_ij = J_ji
        Zero diagonal: J_ii = 0 (no self-interaction)
    """
    # Initialize spins randomly as ±1
    # This gives a random starting configuration
    spins = rng.choice([-1, 1], size=N)
    
    # Create Gaussian coupling matrix
    # Scaling by 1/√N ensures extensive energy in thermodynamic limit
    J = rng.normal(0, 1/np.sqrt(N), size=(N, N))
    
    # Make the coupling matrix symmetric: J_ij = J_ji
    # This ensures the Hamiltonian is real and symmetric
    J = (J + J.T) / 2
    
    # Set diagonal elements to zero (no self-interaction)
    # Spins don't interact with themselves
    np.fill_diagonal(J, 0)
    
    return spins, J

# =============================================================================
# ENERGY CALCULATION FUNCTIONS
# =============================================================================

def calculate_energy(spins, J):
    """
    Calculate total energy of the SK system.
    
    The SK Hamiltonian is: H = -∑_{i<j} J_ij * s_i * s_j
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
        
    Returns:
    --------
    energy : float
        Total energy of the system
        
    Notes:
    ------
    - We sum over i<j to avoid double counting
    - Negative sign in Hamiltonian: ferromagnetic couplings (J>0) favor alignment
    - Energy is extensive: scales with N
    """
    energy = 0.0
    N = len(spins)
    
    # Sum over all unique pairs (i,j) with i < j
    for i in range(N):
        for j in range(i+1, N):
            # Add contribution from this pair
            energy -= J[i, j] * spins[i] * spins[j]
    
    return energy

def calculate_local_field(spins, J, i):
    """
    Calculate local magnetic field acting on spin i.
    
    The local field is: h_i = ∑_j J_ij * s_j
    This represents the effective field that spin i experiences
    due to all other spins in the system.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
    i : int
        Index of the spin for which to calculate local field
        
    Returns:
    --------
    h_i : float
        Local magnetic field at site i
        
    Notes:
    ------
    - Includes all interactions with other spins
    - Since J_ii = 0, self-interaction is automatically excluded
    - Sign convention: positive field favors spin up (+1)
    """
    # Sum over all couplings J_ij * s_j
    # Since J_ii = 0, the self-interaction term vanishes
    return np.sum(J[i, :] * spins)

# =============================================================================
# MEAN FIELD UPDATE FUNCTIONS
# =============================================================================

def mean_field_update(spins, J, T):
    """
    Perform one mean field update step.
    
    In the mean field approximation, each spin is updated according to:
    s_i^{new} = tanh(h_i / T)
    
    where h_i is the local field and T is the temperature.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
    T : float
        Temperature (in units where k_B = 1)
        
    Returns:
    --------
    new_spins : numpy.ndarray
        Updated spin configuration, shape (N,)
        
    Notes:
    ------
    - tanh function ensures spins stay in [-1, 1] range
    - At T → 0: spins become ±1 (classical limit)
    - At T → ∞: spins approach 0 (paramagnetic limit)
    - This is a deterministic update (no randomness)
    """
    N = len(spins)
    new_spins = np.zeros(N)
    
    # Update each spin according to mean field rule
    for i in range(N):
        # Calculate local field acting on spin i
        h_i = calculate_local_field(spins, J, i)
        
        # Mean field update: s_i = tanh(h_i / T)
        # This minimizes the local free energy
        new_spins[i] = np.tanh(h_i / T)
    
    return new_spins

# =============================================================================
# PHYSICAL OBSERVABLES
# =============================================================================

def calculate_magnetization(spins):
    """
    Calculate average magnetization per spin.
    
    Magnetization M = (1/N) * ∑_i s_i
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
        
    Returns:
    --------
    magnetization : float
        Average magnetization per spin
        Range: [-1, 1]
        
    Notes:
    ------
    - M = +1: all spins aligned up (ferromagnetic)
    - M = -1: all spins aligned down (ferromagnetic)
    - M = 0: no net magnetization (paramagnetic or frustrated)
    """
    return np.mean(spins)

def calculate_overlap(spins1, spins2):
    """
    Calculate overlap between two spin configurations.
    
    Overlap q = (1/N) * ∑_i s1_i * s2_i
    
    Parameters:
    -----------
    spins1, spins2 : numpy.ndarray
        Two spin configurations to compare, shape (N,)
        
    Returns:
    --------
    overlap : float
        Overlap between configurations
        Range: [-1, 1]
        
    Notes:
    ------
    - q = +1: configurations are identical
    - q = -1: configurations are opposite
    - q = 0: configurations are uncorrelated
    - Important order parameter in spin glass theory
    """
    return np.mean(spins1 * spins2)

# =============================================================================
# SYSTEM INFORMATION AND DIAGNOSTICS
# =============================================================================

def print_system_info(spins, J, step=None):
    """
    Print comprehensive information about the current system state.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    step : int, optional
        Current iteration step number
    """
    energy = calculate_energy(spins, J)
    magnetization = calculate_magnetization(spins)
    
    if step is not None:
        print(f"Step {step}:")
    print(f"  Energy: {energy:.4f}")
    print(f"  Energy per spin: {energy/len(spins):.4f}")
    print(f"  Magnetization: {magnetization:.4f}")
    print(f"  |Magnetization|: {abs(magnetization):.4f}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_evolution(magnetizations, energies, overlaps, T):
    """
    Plot the time evolution of key observables.
    
    Creates a three-panel plot showing:
    1. Magnetization vs time
    2. Energy vs time  
    3. Overlap with previous configuration vs time
    
    Parameters:
    -----------
    magnetizations : list
        Time series of magnetization values
    energies : list
        Time series of energy values
    overlaps : list
        Time series of overlap values with previous step
    T : float
        Temperature (for plot title)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot magnetization evolution
    ax1.plot(magnetizations, 'b-', linewidth=2, label='Magnetization')
    ax1.set_ylabel('Magnetization')
    ax1.set_title(f'SK Model - Mean Field Gaussian Approximation (T={T})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot energy evolution
    ax2.plot(energies, 'r-', linewidth=2, label='Energy')
    ax2.set_ylabel('Energy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot overlap evolution (measure of configuration changes)
    ax3.plot(overlaps, 'g-', linewidth=2, label='Overlap with previous step')
    ax3.set_ylabel('Overlap')
    ax3.set_xlabel('Iterations')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def plot_spin_distribution(spins, T):
    """
    Plot histogram of spin values.
    
    Shows the distribution of spin values in the final configuration.
    In mean field approximation, spins can take continuous values.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Final spin configuration
    T : float
        Temperature (for plot title)
    """
    plt.figure(figsize=(8, 6))
    plt.hist(spins, bins=50, alpha=0.7, density=True, edgecolor='black', 
             color='skyblue', label=f'N={len(spins)} spins')
    plt.xlabel('Spin Value')
    plt.ylabel('Probability Density')
    plt.title(f'Spin Distribution - Mean Field Gaussian (T={T})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add vertical lines at ±1 for reference
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='±1 bounds')
    plt.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

# =============================================================================
# MAIN SIMULATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SK SPIN GLASS MODEL - MEAN FIELD GAUSSIAN APPROXIMATION")
    print("=" * 60)
    print(f"Number of spins: {N}")
    print(f"Temperature: {T}")
    print(f"Total steps: {num_steps}")
    print(f"Equilibration steps: {equilibration}")
    print(f"Random seed: {seed}")
    print()
    
    # Initialize the system
    print("Initializing system...")
    spins, J = initialize_sk_system(N)
    
    print("Initial configuration:")
    print_system_info(spins, J)
    print()
    
    # Arrays to store time evolution data
    magnetizations = []  # Time series of magnetization
    energies = []        # Time series of energy
    overlaps = []        # Time series of overlap with previous step
    
    print("Starting mean field evolution...")
    start_time = time.time()
    
    # Main evolution loop
    for step in range(num_steps):
        # Store previous configuration for overlap calculation
        old_spins = spins.copy()
        
        # Perform mean field update
        spins = mean_field_update(spins, J, T)
        
        # Collect data after equilibration period
        if step >= equilibration:
            magnetizations.append(calculate_magnetization(spins))
            energies.append(calculate_energy(spins, J))
            overlaps.append(calculate_overlap(spins, old_spins))
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: M = {calculate_magnetization(spins):.4f}, "
                  f"E = {calculate_energy(spins, J):.4f}")
    
    execution_time = time.time() - start_time
    print(f"\nSimulation completed in {execution_time:.2f} seconds")
    
    print("\nFinal configuration:")
    print_system_info(spins, J)
    
    # =============================================================================
    # DATA ANALYSIS AND VISUALIZATION
    # =============================================================================
    
    print("\n" + "=" * 40)
    print("FINAL STATISTICS")
    print("=" * 40)
    
    # Calculate final statistics
    final_magnetization = np.mean(magnetizations)
    magnetization_std = np.std(magnetizations)
    final_energy = np.mean(energies)
    energy_std = np.std(energies)
    final_overlap = np.mean(overlaps)
    overlap_std = np.std(overlaps)
    
    print(f"Average magnetization: {final_magnetization:.4f} ± {magnetization_std:.4f}")
    print(f"Average energy: {final_energy:.4f} ± {energy_std:.4f}")
    print(f"Average energy per spin: {final_energy/N:.4f} ± {energy_std/N:.4f}")
    print(f"Average overlap: {final_overlap:.4f} ± {overlap_std:.4f}")
    
    # Check for convergence
    if len(overlaps) > 10:
        recent_overlap = np.mean(overlaps[-10:])
        if recent_overlap > 0.99:
            print("✓ System appears to have converged (high overlap)")
        else:
            print("⚠ System may not have fully converged (low overlap)")
    
    # Generate visualizations
    print("\nGenerating plots...")
    plot_evolution(magnetizations, energies, overlaps, T)
    plot_spin_distribution(spins, T)
    
    print("\nSimulation complete!")


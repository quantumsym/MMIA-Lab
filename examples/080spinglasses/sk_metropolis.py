#!/usr/bin/env python3
"""
SK (Sherrington-Kirkpatrick) Spin Glass Model - Metropolis Monte Carlo
=====================================================================

This script implements the SK spin glass model using the Metropolis-Hastings
Monte Carlo algorithm. This is the most accurate method for studying
thermodynamic properties of spin glasses.

Key Features:
- Discrete spins ±1 with Monte Carlo dynamics
- Metropolis acceptance criterion with Boltzmann probability
- Calculation of thermodynamic quantities (susceptibility, specific heat)
- Autocorrelation analysis for equilibration assessment
- Ergodicity tests and glass phase analysis

Author: Based on numpy and matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
N = 100                    # Number of spins in the system
T = 1.0                    # Temperature (in units where k_B = 1)
num_steps = 10000          # Total number of Monte Carlo sweeps
equilibration = 1000       # Number of equilibration sweeps before data collection
seed = 42                  # Random seed for reproducibility

# Random number generator with fixed seed
rng = np.random.default_rng(seed)

# =============================================================================
# SYSTEM INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_sk_system(N):
    """
    Initialize the SK spin glass system for Monte Carlo simulation.
    
    Parameters:
    -----------
    N : int
        Number of spins in the system
        
    Returns:
    --------
    spins : numpy.ndarray
        Initial discrete spin configuration (±1), shape (N,)
    J : numpy.ndarray
        Symmetric Gaussian coupling matrix, shape (N, N)
        
    Notes:
    ------
    - Spins are discrete variables: s_i ∈ {-1, +1}
    - Random initial configuration provides unbiased starting point
    - Coupling matrix J_ij ~ N(0, 1/√N) with proper thermodynamic scaling
    - Symmetric matrix ensures real Hamiltonian: J_ij = J_ji
    - Zero diagonal prevents self-interactions: J_ii = 0
    """
    # Initialize spins as discrete ±1 variables
    # Random initial state avoids bias toward any particular configuration
    spins = rng.choice([-1, 1], size=N)
    
    # Create Gaussian coupling matrix with thermodynamic scaling
    # Scaling by 1/√N ensures energy remains extensive as N → ∞
    J = rng.normal(0, 1/np.sqrt(N), size=(N, N))
    
    # Enforce symmetry: J_ij = J_ji
    # This ensures the Hamiltonian is real and symmetric
    J = (J + J.T) / 2
    
    # Remove self-interactions: J_ii = 0
    # Spins don't interact with themselves
    np.fill_diagonal(J, 0)
    
    return spins, J

# =============================================================================
# ENERGY CALCULATION FUNCTIONS
# =============================================================================

def calculate_energy(spins, J):
    """
    Calculate total energy of the SK system.
    
    SK Hamiltonian: H = -∑_{i<j} J_ij * s_i * s_j
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration (±1), shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
        
    Returns:
    --------
    energy : float
        Total energy of the system
        
    Notes:
    ------
    - Sum over i<j avoids double counting of interactions
    - Negative sign: positive J_ij favors parallel spins (ferromagnetic)
    - Energy is extensive: E ~ O(N) for proper thermodynamic limit
    - Each pair interaction contributes -J_ij * s_i * s_j to total energy
    """
    energy = 0.0
    N = len(spins)
    
    # Sum over all unique pairs (i,j) with i < j
    # This avoids double counting since J_ij = J_ji
    for i in range(N):
        for j in range(i+1, N):
            # Add contribution from pair (i,j)
            energy -= J[i, j] * spins[i] * spins[j]
    
    return energy

def calculate_local_field(spins, J, i):
    """
    Calculate local magnetic field acting on spin i.
    
    Local field: h_i = ∑_j J_ij * s_j (j ≠ i)
    
    This represents the effective magnetic field that spin i experiences
    due to all other spins in the system.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
    i : int
        Index of spin for which to calculate local field
        
    Returns:
    --------
    h_i : float
        Local magnetic field at site i
        
    Notes:
    ------
    - Positive field favors spin up (s_i = +1)
    - Negative field favors spin down (s_i = -1)
    - Since J_ii = 0, self-interaction is automatically excluded
    - This is the key quantity for Metropolis acceptance probability
    """
    # Sum all interactions: h_i = ∑_j J_ij * s_j
    # Since J_ii = 0, the self-interaction term vanishes automatically
    return np.sum(J[i, :] * spins) - J[i, i] * spins[i]

def delta_energy_spin_flip(spins, J, i):
    """
    Calculate energy change for flipping spin i.
    
    Energy change: ΔE = E_new - E_old = 2 * s_i * h_i
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    i : int
        Index of spin to flip
        
    Returns:
    --------
    delta_E : float
        Energy change for flipping spin i
        
    Notes:
    ------
    - ΔE < 0: flip decreases energy (favorable)
    - ΔE > 0: flip increases energy (unfavorable, accepted with probability)
    - Factor of 2 comes from: s_i → -s_i changes interaction by 2*s_i*h_i
    - This is the key quantity for Metropolis criterion
    """
    h_i = calculate_local_field(spins, J, i)
    return 2 * spins[i] * h_i

# =============================================================================
# METROPOLIS MONTE CARLO ALGORITHM
# =============================================================================

def metropolis_spin_update(spins, J, i, T):
    """
    Perform Metropolis update for spin i.
    
    Metropolis-Hastings algorithm:
    1. Calculate energy change ΔE for proposed flip
    2. If ΔE ≤ 0: accept flip (energy decreases)
    3. If ΔE > 0: accept with probability P = exp(-ΔE/T)
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration (modified in-place)
    J : numpy.ndarray
        Coupling matrix
    i : int
        Index of spin to attempt flipping
    T : float
        Temperature
        
    Returns:
    --------
    spins : numpy.ndarray
        Updated spin configuration
        
    Notes:
    ------
    - Satisfies detailed balance: ensures correct equilibrium distribution
    - At T → 0: only energy-lowering moves accepted (T=0 dynamics)
    - At T → ∞: all moves accepted with equal probability (random walk)
    - Boltzmann factor exp(-ΔE/T) gives correct thermal equilibrium
    """
    # Calculate energy change for proposed spin flip
    delta_E = delta_energy_spin_flip(spins, J, i)
    
    # Metropolis acceptance criterion
    if delta_E <= 0:
        # Energy decreases or stays same: always accept
        spins[i] *= -1  # Flip the spin: +1 → -1 or -1 → +1
    else:
        # Energy increases: accept with Boltzmann probability
        P_Boltzmann = np.exp(-delta_E / T)
        if rng.random() < P_Boltzmann:
            spins[i] *= -1  # Accept the flip
        # Otherwise: reject flip (spin remains unchanged)
    
    return spins

def metropolis_sweep(spins, J, T):
    """
    Perform one complete Metropolis sweep.
    
    A sweep consists of attempting to flip each spin exactly once.
    This ensures each spin has equal opportunity for updates.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    T : float
        Temperature
        
    Returns:
    --------
    spins : numpy.ndarray
        Updated spin configuration after full sweep
        
    Notes:
    ------
    - Sequential updates: spins updated in order 0, 1, 2, ..., N-1
    - Alternative: random order updates (see metropolis_random_sweep)
    - One sweep ≈ one unit of Monte Carlo time
    - After many sweeps, system reaches thermal equilibrium
    """
    N = len(spins)
    for i in range(N):
        spins = metropolis_spin_update(spins, J, i, T)
    return spins

def metropolis_random_sweep(spins, J, T):
    """
    Perform one Metropolis sweep with random spin selection.
    
    Instead of sequential updates, randomly select N spins for update attempts.
    This can improve ergodicity and reduce correlations.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    T : float
        Temperature
        
    Returns:
    --------
    spins : numpy.ndarray
        Updated configuration after N random update attempts
        
    Notes:
    ------
    - Each sweep: exactly N update attempts with random spin selection
    - Some spins may be selected multiple times, others not at all
    - Generally better mixing than sequential updates
    - Preferred method for most Monte Carlo simulations
    """
    N = len(spins)
    for _ in range(N):
        # Randomly select a spin to attempt flipping
        i = rng.integers(0, N)
        spins = metropolis_spin_update(spins, J, i, T)
    return spins

# =============================================================================
# PHYSICAL OBSERVABLES AND THERMODYNAMIC QUANTITIES
# =============================================================================

def calculate_magnetization(spins):
    """
    Calculate average magnetization per spin.
    
    Magnetization: M = (1/N) * ∑_i s_i
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Spin configuration
        
    Returns:
    --------
    magnetization : float
        Average magnetization in [-1, 1]
        
    Notes:
    ------
    - M = +1: all spins up (ferromagnetic order)
    - M = -1: all spins down (ferromagnetic order)  
    - M ≈ 0: no net magnetization (paramagnetic or spin glass)
    - In SK model: <M> = 0 due to symmetry, but fluctuations important
    """
    return np.mean(spins)

def calculate_susceptibility(magnetizations):
    """
    Calculate magnetic susceptibility from magnetization fluctuations.
    
    Susceptibility: χ = N * <(M - <M>)²> = N * Var(M)
    
    Parameters:
    -----------
    magnetizations : list or numpy.ndarray
        Time series of magnetization measurements
        
    Returns:
    --------
    susceptibility : float
        Magnetic susceptibility
        
    Notes:
    ------
    - Measures response to external magnetic field
    - χ = ∂M/∂H at H=0 (fluctuation-dissipation theorem)
    - Large χ indicates strong magnetic fluctuations
    - Diverges at magnetic phase transitions
    """
    N = len(magnetizations)
    return N * np.var(magnetizations)

def calculate_specific_heat(energies, T):
    """
    Calculate specific heat from energy fluctuations.
    
    Specific heat: C = <(E - <E>)²> / T² = Var(E) / T²
    
    Parameters:
    -----------
    energies : list or numpy.ndarray
        Time series of energy measurements
    T : float
        Temperature
        
    Returns:
    --------
    specific_heat : float
        Specific heat per spin
        
    Notes:
    ------
    - Measures heat capacity: C = ∂<E>/∂T
    - Related to energy fluctuations by fluctuation-dissipation theorem
    - Peaks at phase transitions due to large energy fluctuations
    - Important thermodynamic quantity for phase identification
    """
    return np.var(energies) / (T**2)

def calculate_overlap(spins1, spins2):
    """
    Calculate overlap between two spin configurations.
    
    Overlap: q = (1/N) * ∑_i s1_i * s2_i
    
    Parameters:
    -----------
    spins1, spins2 : numpy.ndarray
        Two spin configurations to compare
        
    Returns:
    --------
    overlap : float
        Overlap in [-1, 1]
        
    Notes:
    ------
    - q = +1: configurations identical
    - q = -1: configurations opposite
    - q ≈ 0: configurations uncorrelated
    - Key order parameter in spin glass theory
    - Used to detect replica symmetry breaking
    """
    return np.mean(spins1 * spins2)

# =============================================================================
# SYSTEM DIAGNOSTICS AND ANALYSIS
# =============================================================================

def print_system_info(spins, J, T, step=None):
    """
    Print comprehensive information about current system state.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    T : float
        Temperature
    step : int, optional
        Current Monte Carlo step number
    """
    energy = calculate_energy(spins, J)
    magnetization = calculate_magnetization(spins)
    
    if step is not None:
        print(f"Monte Carlo Step {step}:")
    print(f"  Energy: {energy:.6f}")
    print(f"  Energy per spin: {energy/len(spins):.6f}")
    print(f"  Magnetization: {magnetization:.6f}")
    print(f"  |Magnetization|: {abs(magnetization):.6f}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_evolution(magnetizations, energies, T):
    """
    Plot time evolution of magnetization and energy.
    
    Creates a two-panel plot showing the Monte Carlo time series
    of key observables.
    
    Parameters:
    -----------
    magnetizations : list
        Time series of magnetization values
    energies : list
        Time series of energy values
    T : float
        Temperature (for plot title)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Magnetization time series
    ax1.plot(magnetizations, 'b-', linewidth=1, alpha=0.7, label='Magnetization')
    ax1.set_ylabel('Magnetization')
    ax1.set_title(f'SK Model - Metropolis Monte Carlo (T={T})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add running average for magnetization
    if len(magnetizations) > 50:
        window = min(50, len(magnetizations)//10)
        running_avg = np.convolve(magnetizations, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(magnetizations)), running_avg, 'r-', 
                linewidth=2, label=f'Running average (window={window})')
        ax1.legend()
    
    # Energy time series
    ax2.plot(energies, 'r-', linewidth=1, alpha=0.7, label='Energy')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Monte Carlo Sweeps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add running average for energy
    if len(energies) > 50:
        window = min(50, len(energies)//10)
        running_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(energies)), running_avg, 'g-', 
                linewidth=2, label=f'Running average (window={window})')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig("sk_metropolis_energies_magnetization.svg")
    plt.show()

def plot_distributions(magnetizations, energies, T):
    """
    Plot probability distributions of magnetization and energy.
    
    Shows histograms of the equilibrium distributions, which reveal
    the thermodynamic properties of the system.
    
    Parameters:
    -----------
    magnetizations : list
        Equilibrium magnetization measurements
    energies : list
        Equilibrium energy measurements
    T : float
        Temperature (for plot title)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnetization distribution
    ax1.hist(magnetizations, bins=30, alpha=0.7, density=True, 
             edgecolor='black', color='skyblue', label='Magnetization')
    ax1.set_xlabel('Magnetization')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Magnetization Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add Gaussian fit for comparison
    mag_mean = np.mean(magnetizations)
    mag_std = np.std(magnetizations)
    x_mag = np.linspace(min(magnetizations), max(magnetizations), 100)
    gaussian_mag = (1/(mag_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_mag-mag_mean)/mag_std)**2)
    ax1.plot(x_mag, gaussian_mag, 'r-', linewidth=2, 
             label=f'Gaussian fit\nμ={mag_mean:.3f}, σ={mag_std:.3f}')
    ax1.legend()
    
    # Energy distribution
    ax2.hist(energies, bins=30, alpha=0.7, density=True, 
             edgecolor='black', color='lightcoral', label='Energy')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Energy Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add Gaussian fit for comparison
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    x_energy = np.linspace(min(energies), max(energies), 100)
    gaussian_energy = (1/(energy_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_energy-energy_mean)/energy_std)**2)
    ax2.plot(x_energy, gaussian_energy, 'g-', linewidth=2, 
             label=f'Gaussian fit\nμ={energy_mean:.3f}, σ={energy_std:.3f}')
    ax2.legend()
    
    plt.suptitle(f'Equilibrium Distributions - Metropolis SK (T={T})')
    plt.tight_layout()
    plt.savefig("sk_metropolis_histograms.svg")
    plt.show()

def plot_autocorrelation(data, max_lag=100, observable_name="Observable"):
    """
    Calculate and plot autocorrelation function.
    
    Autocorrelation measures how long the system "remembers" its past state.
    Important for determining equilibration time and statistical independence.
    
    Parameters:
    -----------
    data : list or numpy.ndarray
        Time series data
    max_lag : int
        Maximum lag time to calculate
    observable_name : str
        Name of observable for plot labels
        
    Notes:
    ------
    - C(t) = <O(0) * O(t)> / <O²> - <O>²/<O²>
    - Exponential decay: C(t) ≈ exp(-t/τ) where τ is correlation time
    - τ determines how many sweeps needed for independent samples
    """
    def autocorr(x, max_lag):
        """Calculate normalized autocorrelation function."""
        n = len(x)
        # Center the data
        x = x - np.mean(x)
        # Full correlation
        autocorr_full = np.correlate(x, x, mode='full')
        # Take positive lags only
        autocorr_full = autocorr_full[n-1:]
        # Normalize by zero-lag value
        autocorr_full = autocorr_full / autocorr_full[0]
        return autocorr_full[:max_lag+1]
    
    lags = np.arange(max_lag + 1)
    autocorr_values = autocorr(data, max_lag)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lags, autocorr_values, 'b-', linewidth=2, label='Autocorrelation')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1/np.e, color='r', linestyle='--', alpha=0.7, 
                label=f'1/e ≈ {1/np.e:.3f}')
    
    plt.xlabel('Lag (Monte Carlo sweeps)')
    plt.ylabel('Autocorrelation Function')
    plt.title(f'Autocorrelation Analysis - {observable_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sk_metropolis_autocorrelation.svg")
    plt.show()
    
    # Estimate correlation time
    try:
        # Find first crossing of 1/e level
        tau_idx = np.where(autocorr_values <= 1/np.e)[0][0]
        print(f"Correlation time τ ≈ {tau_idx} sweeps")
        print(f"Effective sample size ≈ {len(data)//tau_idx} independent samples")
    except IndexError:
        print(f"Correlation time τ > {max_lag} sweeps (increase max_lag)")
    
    return autocorr_values

# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def simulated_annealing_schedule(step, total_steps, T_initial, T_final):
    """
    Temperature schedule for simulated annealing.
    
    Exponential cooling: T(t) = T_initial * (T_final/T_initial)^(t/total_steps)
    
    Parameters:
    -----------
    step : int
        Current step number
    total_steps : int
        Total number of annealing steps
    T_initial : float
        Starting temperature
    T_final : float
        Final temperature
        
    Returns:
    --------
    temperature : float
        Temperature at current step
        
    Notes:
    ------
    - Slow cooling allows system to find low-energy states
    - Too fast: system gets trapped in metastable states
    - Too slow: inefficient computation
    """
    return T_initial * (T_final / T_initial) ** (step / total_steps)

def analyze_glass_phase(magnetizations, energies, T, T_glass=1.0):
    """
    Analyze signatures of spin glass phase.
    
    Parameters:
    -----------
    magnetizations : list
        Magnetization time series
    energies : list
        Energy time series
    T : float
        Current temperature
    T_glass : float
        Estimated glass transition temperature
        
    Notes:
    ------
    - Glass phase (T < T_glass): non-ergodic, slow dynamics, many metastable states
    - Paramagnetic phase (T > T_glass): ergodic, fast equilibration
    """
    print(f"\n" + "=" * 40)
    print("SPIN GLASS PHASE ANALYSIS")
    print("=" * 40)
    
    mag_std = np.std(magnetizations)
    energy_per_spin = np.mean(energies) / len(magnetizations)
    
    print(f"Current temperature: T = {T:.3f}")
    print(f"Estimated T_glass: {T_glass:.3f}")
    
    if T < T_glass:
        print("Expected phase: SPIN GLASS")
        print("Characteristics to look for:")
        print("- Low energy per spin")
        print("- Large magnetization fluctuations")
        print("- Slow relaxation (long correlation times)")
        print("- Non-Gaussian distributions")
    else:
        print("Expected phase: PARAMAGNETIC")
        print("Characteristics to look for:")
        print("- Higher energy per spin")
        print("- Small magnetization fluctuations")
        print("- Fast relaxation (short correlation times)")
        print("- Gaussian-like distributions")
    
    print(f"\nObserved properties:")
    print(f"Energy per spin: {energy_per_spin:.4f}")
    print(f"Magnetization std: {mag_std:.4f}")
    
    # Simple heuristics for phase identification
    if T < T_glass:
        if energy_per_spin < -0.3 and mag_std > 0.1:
            print("✓ Consistent with spin glass phase")
        else:
            print("⚠ May need longer equilibration or different T")
    else:
        if mag_std < 0.2:
            print("✓ Consistent with paramagnetic phase")
        else:
            print("⚠ Unexpected large fluctuations")

# =============================================================================
# MAIN SIMULATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SK SPIN GLASS MODEL - METROPOLIS MONTE CARLO SIMULATION")
    print("=" * 70)
    print(f"Number of spins: {N}")
    print(f"Temperature: {T}")
    print(f"Total Monte Carlo sweeps: {num_steps}")
    print(f"Equilibration sweeps: {equilibration}")
    print(f"Random seed: {seed}")
    print()
    
    # Initialize system
    print("Initializing SK spin glass system...")
    spins, J = initialize_sk_system(N)
    
    print("Initial configuration:")
    print_system_info(spins, J, T)
    print()
    
    # Data storage arrays
    magnetizations = []  # Magnetization time series
    energies = []        # Energy time series
    
    print("Starting Metropolis Monte Carlo simulation...")
    print("(Progress will be printed every 1000 sweeps)")
    start_time = time.time()
    
    # Main Monte Carlo loop
    for step in range(num_steps):
        # Perform one Monte Carlo sweep
        spins = metropolis_random_sweep(spins, J, T)
        
        # Collect data after equilibration period
        if step >= equilibration:
            magnetizations.append(calculate_magnetization(spins))
            energies.append(calculate_energy(spins, J))
        
        # Print progress
        if step % 1000 == 0:
            current_energy = calculate_energy(spins, J)
            current_mag = calculate_magnetization(spins)
            print(f"Sweep {step}: E = {current_energy:.4f}, M = {current_mag:.4f}")
    
    execution_time = time.time() - start_time
    print(f"\nSimulation completed in {execution_time:.2f} seconds")
    print(f"Average time per sweep: {execution_time/num_steps*1000:.2f} ms")
    
    print("\nFinal configuration:")
    print_system_info(spins, J, T)
    
    # =============================================================================
    # THERMODYNAMIC ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("THERMODYNAMIC QUANTITIES")
    print("=" * 50)
    
    # Calculate equilibrium averages and fluctuations
    mag_mean = np.mean(magnetizations)
    mag_std = np.std(magnetizations)
    mag_abs_mean = np.mean(np.abs(magnetizations))
    
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    energy_per_spin_mean = energy_mean / N
    energy_per_spin_std = energy_std / N
    
    # Calculate thermodynamic response functions
    susceptibility = calculate_susceptibility(magnetizations)
    specific_heat = calculate_specific_heat(energies, T)
    
    print(f"Magnetization:")
    print(f"  <M> = {mag_mean:.6f} ± {mag_std:.6f}")
    print(f"  <|M|> = {mag_abs_mean:.6f}")
    print(f"  Fluctuations: σ_M = {mag_std:.6f}")
    
    print(f"\nEnergy:")
    print(f"  <E> = {energy_mean:.6f} ± {energy_std:.6f}")
    print(f"  <E>/N = {energy_per_spin_mean:.6f} ± {energy_per_spin_std:.6f}")
    print(f"  Fluctuations: σ_E = {energy_std:.6f}")
    
    print(f"\nResponse functions:")
    print(f"  Magnetic susceptibility χ = {susceptibility:.6f}")
    print(f"  Specific heat C = {specific_heat:.6f}")
    
    # =============================================================================
    # EQUILIBRATION AND CORRELATION ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("EQUILIBRATION AND CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Test for equilibration: compare first and second half
    if len(magnetizations) > 100:
        mid_point = len(magnetizations) // 2
        first_half_mag = np.mean(magnetizations[:mid_point])
        second_half_mag = np.mean(magnetizations[mid_point:])
        mag_drift = abs(first_half_mag - second_half_mag)
        
        first_half_energy = np.mean(energies[:mid_point])
        second_half_energy = np.mean(energies[mid_point:])
        energy_drift = abs(first_half_energy - second_half_energy)
        
        print(f"Equilibration test (first vs second half):")
        print(f"  Magnetization drift: {mag_drift:.6f}")
        print(f"  Energy drift: {energy_drift:.6f}")
        
        if mag_drift < mag_std/2 and energy_drift < energy_std/2:
            print("✓ System appears well equilibrated")
        else:
            print("⚠ System may need longer equilibration")
    
    # Autocorrelation analysis
    print(f"\nAutocorrelation analysis:")
    autocorr_mag = plot_autocorrelation(magnetizations, max_lag=min(200, len(magnetizations)//5), 
                                       observable_name="Magnetization")
    
    # =============================================================================
    # ERGODICITY TESTS
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("ERGODICITY TESTS")
    print("=" * 50)
    
    print(f"Sample statistics:")
    print(f"  Number of measurements: {len(magnetizations)}")
    print(f"  Measurement interval: 1 sweep")
    print(f"  Total measurement time: {len(magnetizations)} sweeps")
    
    # Test ergodicity: time average vs ensemble properties
    print(f"\nTime averages:")
    print(f"  Time average |M|: {np.mean(np.abs(magnetizations)):.6f}")
    print(f"  Time average M²: {np.mean(np.array(magnetizations)**2):.6f}")
    print(f"  Standard deviation M: {np.std(magnetizations):.6f}")
    
    # =============================================================================
    # VISUALIZATION AND FINAL ANALYSIS
    # =============================================================================
    
    print("\nGenerating visualizations...")
    
    # Plot time evolution
    plot_evolution(magnetizations, energies, T)
    
    # Plot equilibrium distributions
    plot_distributions(magnetizations, energies, T)
    
    # Analyze glass phase properties
    analyze_glass_phase(magnetizations, energies, T)
    
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"✓ Completed {num_steps} Monte Carlo sweeps")
    print(f"✓ Collected {len(magnetizations)} equilibrium measurements")
    print(f"✓ Final energy per spin: {energy_per_spin_mean:.6f}")
    print(f"✓ Final magnetization: {mag_mean:.6f} ± {mag_std:.6f}")
    print(f"✓ Execution time: {execution_time:.2f} seconds")
    print("\nMonte Carlo simulation complete!")


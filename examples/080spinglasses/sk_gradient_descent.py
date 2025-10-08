#!/usr/bin/env python3
"""
SK (Sherrington-Kirkpatrick) Spin Glass Model - Gradient Descent Optimization
============================================================================

This script implements the SK spin glass model using gradient descent optimization.
The energy landscape is explored by following the negative gradient, with momentum
and adaptive learning rate for improved convergence.

Key Features:
- Continuous spins with gradient-based optimization
- Momentum acceleration for faster convergence
- Adaptive learning rate based on gradient norm
- Energy landscape visualization
- Convergence monitoring

Author: Based on numpy and matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
N = 100                    # Number of spins in the system
learning_rate = 0.01       # Initial learning rate for gradient descent
num_steps = 2000           # Total number of optimization steps
equilibration = 200        # Number of steps before data collection
seed = 42                  # Random seed for reproducibility
momentum = 0.9             # Momentum parameter (0 = no momentum, 1 = full momentum)

# Random number generator with fixed seed
rng = np.random.default_rng(seed)

# =============================================================================
# SYSTEM INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_sk_system(N):
    """
    Initialize the SK spin glass system for gradient descent optimization.
    
    Parameters:
    -----------
    N : int
        Number of spins in the system
        
    Returns:
    --------
    spins : numpy.ndarray
        Initial continuous spin configuration in [-1, 1], shape (N,)
    J : numpy.ndarray
        Symmetric Gaussian coupling matrix, shape (N, N)
        
    Notes:
    ------
    - Spins are initialized as continuous variables in [-1, 1]
    - Coupling matrix J_ij ~ N(0, 1/√N) ensures proper thermodynamic scaling
    - Matrix is symmetric (J_ij = J_ji) and has zero diagonal
    """
    # Initialize spins as continuous random variables in [-1, 1]
    # This provides a good starting point for optimization
    spins = rng.uniform(-1, 1, size=N)
    
    # Create Gaussian coupling matrix with proper scaling
    J = rng.normal(0, 1/np.sqrt(N), size=(N, N))
    
    # Ensure symmetry: J_ij = J_ji
    J = (J + J.T) / 2
    
    # Remove self-interactions: J_ii = 0
    np.fill_diagonal(J, 0)
    
    return spins, J

# =============================================================================
# ENERGY AND GRADIENT CALCULATIONS
# =============================================================================

def calculate_energy(spins, J):
    """
    Calculate total energy of the SK system.
    
    Energy function: E = -0.5 * ∑_{i,j} J_ij * s_i * s_j
    
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
    - Factor of 0.5 avoids double counting since J is symmetric
    - Equivalent to -∑_{i<j} J_ij * s_i * s_j but more efficient
    - Energy is extensive (scales with N)
    """
    # Use matrix multiplication for efficient calculation
    # E = -0.5 * s^T * J * s where s is the spin vector
    return -0.5 * np.sum(J * np.outer(spins, spins))

def calculate_gradient(spins, J):
    """
    Calculate gradient of energy with respect to spins.
    
    Gradient: ∂E/∂s_i = -∑_j J_ij * s_j
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
        
    Returns:
    --------
    gradient : numpy.ndarray
        Gradient vector, shape (N,)
        
    Notes:
    ------
    - Gradient points in direction of steepest energy increase
    - We follow negative gradient to minimize energy
    - Each component is the local field acting on that spin
    """
    # Matrix-vector multiplication: grad_i = -∑_j J_ij * s_j
    return -np.dot(J, spins)

def calculate_gradient_norm(spins, J):
    """
    Calculate L2 norm of the energy gradient.
    
    This serves as a convergence criterion: small gradient norm
    indicates we're near a local minimum.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
        
    Returns:
    --------
    norm : float
        L2 norm of gradient vector
        
    Notes:
    ------
    - ||∇E|| = 0 at exact local minima
    - Practical convergence: ||∇E|| < tolerance (e.g., 1e-4)
    - Used for adaptive learning rate and stopping criteria
    """
    gradient = calculate_gradient(spins, J)
    return np.linalg.norm(gradient)

# =============================================================================
# OPTIMIZATION ALGORITHMS
# =============================================================================

def gradient_descent_step(spins, J, learning_rate, velocity=None, momentum=0.0):
    """
    Perform one gradient descent step with momentum.
    
    Update rules:
    v_new = momentum * v_old - learning_rate * gradient
    s_new = s_old + v_new
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration, shape (N,)
    J : numpy.ndarray
        Coupling matrix, shape (N, N)
    learning_rate : float
        Step size for gradient descent
    velocity : numpy.ndarray, optional
        Previous velocity vector for momentum, shape (N,)
    momentum : float
        Momentum parameter in [0, 1]
        
    Returns:
    --------
    new_spins : numpy.ndarray
        Updated spin configuration, shape (N,)
    new_velocity : numpy.ndarray
        Updated velocity vector, shape (N,)
        
    Notes:
    ------
    - Momentum helps accelerate convergence and escape shallow minima
    - Higher momentum (closer to 1) retains more previous direction
    - Spins are clipped to [-1, 1] to maintain physical constraints
    """
    # Calculate current gradient
    gradient = calculate_gradient(spins, J)
    
    # Initialize velocity if not provided
    if velocity is None:
        velocity = np.zeros_like(spins)
    
    # Update velocity with momentum term
    # This combines current gradient with previous direction
    velocity = momentum * velocity - learning_rate * gradient
    
    # Update spin positions
    new_spins = spins + velocity
    
    # Apply constraints: keep spins in [-1, 1] range
    # This maintains physical interpretation of spins
    new_spins = np.clip(new_spins, -1, 1)
    
    return new_spins, velocity

def steepest_descent_step(spins, J, learning_rate):
    """
    Perform simple steepest descent step without momentum.
    
    Update rule: s_new = s_old - learning_rate * gradient
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    learning_rate : float
        Step size
        
    Returns:
    --------
    new_spins : numpy.ndarray
        Updated spin configuration
        
    Notes:
    ------
    - Simpler than momentum version but may converge slower
    - Good for comparison with momentum-based optimization
    - More stable but less efficient than momentum methods
    """
    gradient = calculate_gradient(spins, J)
    new_spins = spins - learning_rate * gradient
    
    # Apply spin constraints
    new_spins = np.clip(new_spins, -1, 1)
    
    return new_spins

# =============================================================================
# ADAPTIVE LEARNING RATE
# =============================================================================

def adaptive_learning_rate(gradient_norm, initial_lr, decay_factor=0.99, 
                          threshold=1.0):
    """
    Adapt learning rate based on gradient norm.
    
    Strategy: Reduce learning rate when gradient is small to improve
    convergence near minima.
    
    Parameters:
    -----------
    gradient_norm : float
        Current L2 norm of gradient
    initial_lr : float
        Base learning rate
    decay_factor : float
        Factor to reduce learning rate (< 1)
    threshold : float
        Gradient norm threshold for adaptation
        
    Returns:
    --------
    adapted_lr : float
        Adapted learning rate
        
    Notes:
    ------
    - Reduces learning rate when close to minimum (small gradient)
    - Helps achieve better convergence precision
    - Prevents overshooting near local minima
    """
    if gradient_norm < threshold:
        return initial_lr * decay_factor
    else:
        return initial_lr

# =============================================================================
# PHYSICAL OBSERVABLES
# =============================================================================

def calculate_magnetization(spins):
    """
    Calculate average magnetization per spin.
    
    M = (1/N) * ∑_i s_i
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Spin configuration
        
    Returns:
    --------
    magnetization : float
        Average magnetization in [-1, 1]
    """
    return np.mean(spins)

# =============================================================================
# SYSTEM DIAGNOSTICS
# =============================================================================

def print_system_info(spins, J, step=None):
    """
    Print comprehensive system information.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    step : int, optional
        Current optimization step
    """
    energy = calculate_energy(spins, J)
    magnetization = calculate_magnetization(spins)
    gradient_norm = calculate_gradient_norm(spins, J)
    
    if step is not None:
        print(f"Step {step}:")
    print(f"  Energy: {energy:.6f}")
    print(f"  Energy per spin: {energy/len(spins):.6f}")
    print(f"  Magnetization: {magnetization:.6f}")
    print(f"  Gradient norm: {gradient_norm:.6f}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_evolution(energies, magnetizations, gradient_norms, learning_rates):
    """
    Plot evolution of key quantities during optimization.
    
    Creates a 2x2 subplot showing:
    1. Energy vs iteration
    2. Magnetization vs iteration
    3. Gradient norm vs iteration (log scale)
    4. Learning rate vs iteration
    
    Parameters:
    -----------
    energies : list
        Time series of energy values
    magnetizations : list
        Time series of magnetization values
    gradient_norms : list
        Time series of gradient norms
    learning_rates : list
        Time series of learning rates
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy evolution
    ax1.plot(energies, 'r-', linewidth=2, label='Energy')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Magnetization evolution
    ax2.plot(magnetizations, 'b-', linewidth=2, label='Magnetization')
    ax2.set_ylabel('Magnetization')
    ax2.set_title('Magnetization Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gradient norm evolution (log scale for better visualization)
    ax3.semilogy(gradient_norms, 'g-', linewidth=2, label='Gradient Norm')
    ax3.set_ylabel('Gradient Norm (log scale)')
    ax3.set_xlabel('Iterations')
    ax3.set_title('Convergence Monitor')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Learning rate adaptation
    ax4.plot(learning_rates, 'm-', linewidth=2, label='Learning Rate')
    ax4.set_ylabel('Learning Rate')
    ax4.set_xlabel('Iterations')
    ax4.set_title('Learning Rate Adaptation')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('SK Model - Gradient Descent Optimization', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_spin_distribution(spins):
    """
    Plot histogram of final spin values.
    
    Shows the distribution of optimized spin values.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Final spin configuration
    """
    plt.figure(figsize=(8, 6))
    plt.hist(spins, bins=50, alpha=0.7, density=True, edgecolor='black',
             color='lightcoral', label=f'N={len(spins)} spins')
    plt.xlabel('Spin Value')
    plt.ylabel('Probability Density')
    plt.title('Final Spin Distribution - Gradient Descent')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Zero')
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='±1 bounds')
    plt.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

def plot_energy_landscape_2d(spins, J, indices=[0, 1], resolution=50):
    """
    Visualize 2D energy landscape for two selected spins.
    
    Creates a contour plot showing how energy varies when two spins
    are changed while others remain fixed.
    
    Parameters:
    -----------
    spins : numpy.ndarray
        Current spin configuration
    J : numpy.ndarray
        Coupling matrix
    indices : list
        Indices of two spins to vary [i, j]
    resolution : int
        Number of grid points along each axis
        
    Notes:
    ------
    - Provides insight into local energy landscape structure
    - Shows location of current configuration
    - Helps understand optimization challenges
    """
    if len(spins) < 2:
        print("Need at least 2 spins for 2D landscape plot")
        return
    
    # Create grid for the two selected spins
    s1_range = np.linspace(-1, 1, resolution)
    s2_range = np.linspace(-1, 1, resolution)
    S1, S2 = np.meshgrid(s1_range, s2_range)
    
    # Calculate energy for each point in the grid
    energies_2d = np.zeros_like(S1)
    spins_temp = spins.copy()
    
    for i, s1 in enumerate(s1_range):
        for j, s2 in enumerate(s2_range):
            # Set the two spins to grid values
            spins_temp[indices[0]] = s1
            spins_temp[indices[1]] = s2
            # Calculate energy with other spins fixed
            energies_2d[j, i] = calculate_energy(spins_temp, J)
    
    # Create contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(S1, S2, energies_2d, levels=20, alpha=0.7)
    plt.colorbar(contour, label='Energy')
    
    # Mark current position
    plt.plot(spins[indices[0]], spins[indices[1]], 'ro', markersize=10, 
             label=f'Current position', markeredgecolor='black', markeredgewidth=2)
    
    # Mark global minimum in this 2D slice
    min_idx = np.unravel_index(np.argmin(energies_2d), energies_2d.shape)
    min_s1 = s1_range[min_idx[1]]
    min_s2 = s2_range[min_idx[0]]
    plt.plot(min_s1, min_s2, 'g*', markersize=15, 
             label=f'2D minimum', markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel(f'Spin {indices[0]}')
    plt.ylabel(f'Spin {indices[1]}')
    plt.title(f'Energy Landscape - Spins {indices[0]} and {indices[1]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# MAIN SIMULATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SK SPIN GLASS MODEL - GRADIENT DESCENT OPTIMIZATION")
    print("=" * 70)
    print(f"Number of spins: {N}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Momentum parameter: {momentum}")
    print(f"Total optimization steps: {num_steps}")
    print(f"Equilibration steps: {equilibration}")
    print(f"Random seed: {seed}")
    print()
    
    # Initialize system
    print("Initializing system...")
    spins, J = initialize_sk_system(N)
    velocity = np.zeros_like(spins)  # Initialize velocity for momentum
    
    print("Initial configuration:")
    print_system_info(spins, J)
    print()
    
    # Data storage arrays
    energies = []           # Energy evolution
    magnetizations = []     # Magnetization evolution
    gradient_norms = []     # Gradient norm evolution
    learning_rates = []     # Learning rate evolution
    
    current_lr = learning_rate  # Current adaptive learning rate
    
    print("Starting gradient descent optimization...")
    start_time = time.time()
    
    # Main optimization loop
    for step in range(num_steps):
        # Calculate gradient norm for diagnostics and adaptation
        grad_norm = calculate_gradient_norm(spins, J)
        
        # Adapt learning rate based on gradient norm
        current_lr = adaptive_learning_rate(grad_norm, learning_rate)
        
        # Perform gradient descent step with momentum
        spins, velocity = gradient_descent_step(spins, J, current_lr, 
                                              velocity, momentum)
        
        # Collect data after equilibration
        if step >= equilibration:
            energies.append(calculate_energy(spins, J))
            magnetizations.append(calculate_magnetization(spins))
            gradient_norms.append(grad_norm)
            learning_rates.append(current_lr)
        
        # Print progress
        if step % 200 == 0:
            energy_current = calculate_energy(spins, J)
            print(f"Step {step}: Energy = {energy_current:.6f}, "
                  f"Grad norm = {grad_norm:.6f}, LR = {current_lr:.6f}")
    
    execution_time = time.time() - start_time
    print(f"\nOptimization completed in {execution_time:.2f} seconds")
    
    print("\nFinal configuration:")
    print_system_info(spins, J)
    
    # =============================================================================
    # CONVERGENCE ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("CONVERGENCE ANALYSIS")
    print("=" * 50)
    
    final_energy = energies[-1]
    final_magnetization = magnetizations[-1]
    final_gradient_norm = gradient_norms[-1]
    
    print(f"Final energy: {final_energy:.6f}")
    print(f"Final energy per spin: {final_energy/N:.6f}")
    print(f"Final magnetization: {final_magnetization:.6f}")
    print(f"Final gradient norm: {final_gradient_norm:.6f}")
    
    # Check convergence criteria
    convergence_threshold = 1e-4
    if final_gradient_norm < convergence_threshold:
        print(f"✓ Converged: gradient norm < {convergence_threshold}")
    else:
        print(f"⚠ Not fully converged: gradient norm = {final_gradient_norm:.2e}")
    
    # Analyze energy decrease
    if len(energies) > 10:
        energy_change = energies[-1] - energies[0]
        print(f"Total energy change: {energy_change:.6f}")
        
        # Check for monotonic decrease (good optimization)
        energy_increases = sum(1 for i in range(1, len(energies)) 
                             if energies[i] > energies[i-1])
        print(f"Energy increases: {energy_increases}/{len(energies)-1} steps")
    
    # Generate visualizations
    print("\nGenerating plots...")
    plot_evolution(energies, magnetizations, gradient_norms, learning_rates)
    plot_spin_distribution(spins)
    plot_energy_landscape_2d(spins, J)
    
    print("\nOptimization complete!")


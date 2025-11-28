#!/usr/bin/env python3
#
'''
 implementation of Mean Field Theory for Hopfield networks.

This script provides functions to analyze a Hopfield network using the mean-field
approximation. Instead of simulating discrete neuron states, this approach tracks
the continuous average magnetizations, transforming a discrete optimization problem
into a continuous one. This allows for faster, deterministic analysis of the
network's equilibrium states, free energy, and phase transitions.
'''

import numpy as np
import time

# ============================================================================
#  MEAN FIELD HOPFIELD NETWORK
# ============================================================================

def initialize_mean_field_network(num_neurons, temperature=1.0):
    """
    Initializes a Mean Field Hopfield network model.

    Args:
        num_neurons (int): The number of neurons in the network.
        temperature (float): The temperature T for thermal fluctuations.

    Returns:
        dict: A dictionary representing the mean-field model's state.
    """
    network = {
        'num_neurons': num_neurons,
        'temperature': temperature,
        'weights': np.zeros((num_neurons, num_neurons)),
        'external_fields': np.zeros(num_neurons),
        'magnetizations': np.random.uniform(-0.1, 0.1, num_neurons),
        'stored_patterns': []
    }
    print(f"Initialized Mean Field Hopfield Network with {num_neurons} neurons at T={temperature:.2f}")
    return network

def store_patterns_mf(network, patterns):
    """
    Stores patterns in the network for mean-field analysis.

    Args:
        network (dict): The network dictionary.
        patterns (list): A list of patterns to store (NumPy arrays of {-1, +1}).
    """
    if not patterns:
        raise ValueError("No patterns provided")

    num_patterns = len(patterns)
    network['stored_patterns'] = [p.copy() for p in patterns]
    network['weights'] = np.zeros((network['num_neurons'], network['num_neurons']))

    for pattern in patterns:
        network['weights'] += np.outer(pattern, pattern)
    
    network['weights'] /= num_patterns
    np.fill_diagonal(network['weights'], 0)
    print(f"Stored {num_patterns} patterns for mean-field analysis.")

def compute_effective_field(network, magnetizations):
    """
    Computes the effective field h_i = Σ_j(w_ij * m_j) + θ_i for all neurons.

    Args:
        network (dict): The network dictionary.
        magnetizations (np.ndarray): The current vector of magnetizations.

    Returns:
        np.ndarray: The vector of effective fields.
    """
    return np.dot(network['weights'], magnetizations) + network['external_fields']

def solve_mean_field_equations(network, max_iterations=1000, tolerance=1e-6, learning_rate=0.5, verbose=True):
    """
    Solves the mean-field equations iteratively until convergence.

    The core equation is m_i = tanh(h_i / T).

    Args:
        network (dict): The network dictionary. Its 'magnetizations' will be updated.
        max_iterations (int): Maximum number of iterations to perform.
        tolerance (float): The convergence criterion for the change in magnetizations.
        learning_rate (float): The step size for each update (0 < lr <= 1).
        verbose (bool): If True, prints convergence progress.

    Returns:
        dict: A dictionary containing the results of the iterative solution.
    """
    if verbose:
        print(f"Solving mean-field equations (max_iter={max_iterations}, tol={tolerance:.1e})...")

    start_time = time.time()
    convergence_history = []

    for iteration in range(max_iterations):
        old_magnetizations = network['magnetizations'].copy()

        # 1. Compute effective fields based on current magnetizations
        effective_fields = compute_effective_field(network, old_magnetizations)

        # 2. Compute the new equilibrium magnetizations based on the fields
        if network['temperature'] > 0:
            new_magnetizations = np.tanh(effective_fields / network['temperature'])
        else:
            new_magnetizations = np.sign(effective_fields)

        # 3. Apply the update with a learning rate for stability
        network['magnetizations'] = (1 - learning_rate) * old_magnetizations + learning_rate * new_magnetizations

        # 4. Check for convergence
        max_change = np.max(np.abs(network['magnetizations'] - old_magnetizations))
        convergence_history.append(max_change)

        if max_change < tolerance:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations.")
            break
    else: # This else belongs to the for loop, executed if the loop finishes without break
        if verbose:
            print(f"  Did not converge after {max_iterations} iterations.")

    solve_time = time.time() - start_time

    results = {
        'converged': max_change < tolerance,
        'iterations': iteration + 1,
        'final_magnetizations': network['magnetizations'].copy(),
        'convergence_history': np.array(convergence_history),
        'solve_time': solve_time
    }
    return results

def compute_free_energy_mf(network, magnetizations):
    """
    Computes the mean-field free energy of the system.

    F = -0.5 * Σ_ij(w_ij*m_i*m_j) - Σ_i(θ_i*m_i) - T*S(m)

    Args:
        network (dict): The network dictionary.
        magnetizations (np.ndarray): The magnetizations vector.

    Returns:
        float: The mean-field free energy.
    """
    # Interaction energy term
    interaction_energy = -0.5 * np.dot(magnetizations, np.dot(network['weights'], magnetizations))

    # External field energy term
    field_energy = -np.dot(network['external_fields'], magnetizations)

    # Entropy term: -T * Σ_i S(m_i)
    entropy_contribution = 0.0
    if network['temperature'] > 0:
        # Clip magnetizations to avoid log(0) for m = +/-1
        m_clipped = np.clip(magnetizations, -0.9999, 0.9999)
        term1 = (1 + m_clipped) * np.log(1 + m_clipped)
        term2 = (1 - m_clipped) * np.log(1 - m_clipped)
        entropy = -0.5 * np.sum(term1 + term2)
        entropy_contribution = network['temperature'] * entropy

    free_energy = interaction_energy + field_energy - entropy_contribution
    return free_energy

def compute_pattern_overlaps(network, magnetizations):
    """
    Computes the overlap between magnetizations and stored patterns.

    Overlap m^μ = (1/N) * Σ_i(ξ^μ_i * m_i)

    Args:
        network (dict): The network dictionary.
        magnetizations (np.ndarray): The magnetizations vector.

    Returns:
        list: A list of overlap values, one for each stored pattern.
    """
    if not network['stored_patterns']:
        return []
    
    overlaps = [np.dot(pattern, magnetizations) / network['num_neurons'] for pattern in network['stored_patterns']]
    return overlaps


# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_mean_field_analysis():
    """
    Main demonstration function showing Mean Field Theory analysis capabilities.
    """
    print("=" * 80)
    print("MEAN FIELD HOPFIELD NETWORK ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. BASIC MEAN FIELD ANALYSIS
    print(f"\n{'='*60}")
    print("BASIC MEAN FIELD ANALYSIS")
    print(f"{'='*60}")
    
    # Create test patterns
    num_neurons = 15
    patterns = [
        np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]),  # Alternating
        np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]),    # Block
        np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])   # Structured
    ]
    
    print(f"Created {len(patterns)} test patterns with {num_neurons} neurons each")
    
    # Test at different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for temp in temperatures:
        print(f"\n--- Analysis at Temperature T = {temp} ---")
        
        # Initialize mean field network
        network = initialize_mean_field_network(num_neurons, temperature=temp)
        store_patterns_mf(network, patterns)
        
        # Solve mean field equations
        solution = solve_mean_field_equations(
            network, max_iterations=1000, tolerance=1e-6, learning_rate=0.5, verbose=False
        )
        
        # Compute properties
        free_energy = compute_free_energy_mf(network, network['magnetizations'])
        overlaps = compute_pattern_overlaps(network, network['magnetizations'])
        
        print(f"  Converged: {solution['converged']} (in {solution['iterations']} iterations)")
        print(f"  Free energy: {free_energy:.3f}")
        print(f"  Magnetization norm: {np.linalg.norm(network['magnetizations']):.3f}")
        print(f"  Pattern overlaps: {[f'{o:.3f}' for o in overlaps]}")
        print(f"  Max overlap: {max(np.abs(overlaps)):.3f}")
        print(f"  Solve time: {solution['solve_time']:.3f}s")
    
    # 2. FIXED POINT ANALYSIS
    print(f"\n{'='*60}")
    print("FIXED POINT ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze fixed points at moderate temperature
    network = initialize_mean_field_network(num_neurons, temperature=1.0)
    store_patterns_mf(network, patterns)
    
    print("Searching for multiple fixed points from different initial conditions...")
    
    fixed_points = []
    num_trials = 20
    
    for trial in range(num_trials):
        # Random initial magnetizations
        network['magnetizations'] = np.random.uniform(-0.5, 0.5, num_neurons)
        
        # Solve from this initial condition
        solution = solve_mean_field_equations(network, verbose=False)
        
        if solution['converged']:
            # Check if this is a new fixed point
            is_new = True
            for existing_fp in fixed_points:
                if np.allclose(network['magnetizations'], existing_fp['magnetizations'], atol=1e-3):
                    existing_fp['count'] += 1
                    is_new = False
                    break
            
            if is_new:
                free_energy = compute_free_energy_mf(network, network['magnetizations'])
                overlaps = compute_pattern_overlaps(network, network['magnetizations'])
                
                fixed_points.append({
                    'magnetizations': network['magnetizations'].copy(),
                    'free_energy': free_energy,
                    'overlaps': overlaps,
                    'count': 1
                })
    
    # Sort by free energy (most stable first)
    fixed_points.sort(key=lambda fp: fp['free_energy'])
    
    print(f"\nFound {len(fixed_points)} distinct fixed points:")
    for i, fp in enumerate(fixed_points):
        print(f"  Fixed Point {i+1}:")
        print(f"    Free energy: {fp['free_energy']:.3f}")
        print(f"    Found {fp['count']} times")
        print(f"    Magnetization norm: {np.linalg.norm(fp['magnetizations']):.3f}")
        if fp['overlaps']:
            max_overlap_idx = np.argmax(np.abs(fp['overlaps']))
            print(f"    Best pattern match: Pattern {max_overlap_idx+1} (overlap: {fp['overlaps'][max_overlap_idx]:.3f})")
    
    # 3. TEMPERATURE SWEEP ANALYSIS
    print(f"\n{'='*60}")
    print("TEMPERATURE SWEEP ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze behavior across temperature range
    temp_min, temp_max = 0.1, 5.0
    num_temps = 20
    temperatures = np.linspace(temp_min, temp_max, num_temps)
    
    print(f"Analyzing system behavior from T={temp_min} to T={temp_max}...")
    
    temp_results = {
        'temperatures': temperatures,
        'free_energies': [],
        'magnetization_norms': [],
        'max_overlaps': [],
        'convergence_iterations': []
    }
    
    for temp in temperatures:
        network = initialize_mean_field_network(num_neurons, temperature=temp)
        store_patterns_mf(network, patterns)
        
        # Start from small random magnetizations
        network['magnetizations'] = np.random.uniform(-0.1, 0.1, num_neurons)
        
        # Solve mean field equations
        solution = solve_mean_field_equations(network, verbose=False)
        
        # Compute properties
        free_energy = compute_free_energy_mf(network, network['magnetizations'])
        mag_norm = np.linalg.norm(network['magnetizations'])
        overlaps = compute_pattern_overlaps(network, network['magnetizations'])
        max_overlap = max(np.abs(overlaps)) if overlaps else 0
        
        temp_results['free_energies'].append(free_energy)
        temp_results['magnetization_norms'].append(mag_norm)
        temp_results['max_overlaps'].append(max_overlap)
        temp_results['convergence_iterations'].append(solution['iterations'])
    
    # Detect phase transitions
    mag_norms = np.array(temp_results['magnetization_norms'])
    mag_changes = np.abs(np.diff(mag_norms))
    
    if len(mag_changes) > 0:
        max_change_idx = np.argmax(mag_changes)
        critical_temp = temperatures[max_change_idx]
        
        print(f"\nTemperature sweep results:")
        print(f"  Temperature range: {temp_min:.1f} - {temp_max:.1f}")
        print(f"  Free energy range: {min(temp_results['free_energies']):.3f} - {max(temp_results['free_energies']):.3f}")
        print(f"  Magnetization norm range: {min(mag_norms):.3f} - {max(mag_norms):.3f}")
        print(f"  Max overlap range: {min(temp_results['max_overlaps']):.3f} - {max(temp_results['max_overlaps']):.3f}")
        
        if max(mag_changes) > 0.3:
            print(f"  Possible phase transition detected at T ≈ {critical_temp:.2f}")
        else:
            print(f"  No clear phase transition detected")
    
    # 4. COMPARISON WITH STORED PATTERNS
    print(f"\n{'='*60}")
    print("PATTERN RETRIEVAL ANALYSIS")
    print(f"{'='*60}")
    
    # Test pattern retrieval at low temperature
    network = initialize_mean_field_network(num_neurons, temperature=0.5)
    store_patterns_mf(network, patterns)
    
    print("Testing pattern retrieval from noisy initial conditions...")
    
    for i, pattern in enumerate(patterns):
        print(f"\n--- Pattern {i+1} Retrieval Test ---")
        
        # Start from noisy version of the pattern
        noise_level = 0.3
        noisy_magnetizations = pattern.astype(float) + np.random.normal(0, noise_level, num_neurons)
        noisy_magnetizations = np.clip(noisy_magnetizations, -1, 1)  # Keep in valid range
        
        network['magnetizations'] = noisy_magnetizations
        
        print(f"Original pattern:     {pattern}")
        print(f"Noisy initial state:  {[f'{m:.2f}' for m in noisy_magnetizations]}")
        
        # Solve mean field equations
        solution = solve_mean_field_equations(network, verbose=False)
        
        # Convert final magnetizations to binary for comparison
        final_binary = np.sign(network['magnetizations'])
        
        # Compute overlap and success
        final_overlap = np.dot(pattern, network['magnetizations']) / num_neurons
        retrieval_success = np.array_equal(pattern, final_binary)
        hamming_distance = np.sum(pattern != final_binary)
        
        print(f"Final magnetizations: {[f'{m:.2f}' for m in network['magnetizations']]}")
        print(f"Final binary state:   {final_binary.astype(int)}")
        print(f"Pattern overlap:      {final_overlap:.3f}")
        print(f"Retrieval success:    {retrieval_success}")
        print(f"Hamming distance:     {hamming_distance} bits")
        print(f"Convergence time:     {solution['solve_time']:.3f}s")
    
    # 5. CAPACITY ANALYSIS
    print(f"\n{'='*60}")
    print("STORAGE CAPACITY ANALYSIS")
    print(f"{'='*60}")
    
    # Test storage capacity with increasing number of patterns
    max_patterns = 8
    capacity_results = []
    
    for num_patterns in range(1, max_patterns + 1):
        print(f"\nTesting with {num_patterns} patterns...")
        
        # Generate random patterns
        test_patterns = [np.random.choice([-1, 1], size=num_neurons) for _ in range(num_patterns)]
        
        # Initialize network
        network = initialize_mean_field_network(num_neurons, temperature=0.5)
        store_patterns_mf(network, test_patterns)
        
        # Test retrieval for each pattern
        successful_retrievals = 0
        total_overlap = 0
        
        for pattern in test_patterns:
            # Start from slightly noisy version
            network['magnetizations'] = pattern.astype(float) + np.random.normal(0, 0.1, num_neurons)
            network['magnetizations'] = np.clip(network['magnetizations'], -0.9, 0.9)
            
            # Solve
            solution = solve_mean_field_equations(network, verbose=False)
            
            # Check retrieval
            final_binary = np.sign(network['magnetizations'])
            if np.array_equal(pattern, final_binary):
                successful_retrievals += 1
            
            overlap = np.dot(pattern, network['magnetizations']) / num_neurons
            total_overlap += abs(overlap)
        
        success_rate = successful_retrievals / num_patterns
        avg_overlap = total_overlap / num_patterns
        
        capacity_results.append({
            'num_patterns': num_patterns,
            'success_rate': success_rate,
            'avg_overlap': avg_overlap
        })
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average overlap: {avg_overlap:.3f}")
    
    print(f"\nStorage capacity summary:")
    print("Patterns | Success Rate | Avg Overlap")
    print("---------|--------------|------------")
    for result in capacity_results:
        print(f"{result['num_patterns']:8d} | {result['success_rate']:11.1%} | {result['avg_overlap']:10.3f}")
    
    # Theoretical capacity
    theoretical_capacity = 0.15 * num_neurons
    print(f"\nTheoretical capacity (0.15 * N): {theoretical_capacity:.1f} patterns")
    
    print(f"\n{'='*80}")
    print("MEAN FIELD ANALYSIS DEMONSTRATION COMPLETED")
    print(f"{'='*80}")
    print("Key insights:")
    print("• Mean field theory provides fast, deterministic analysis")
    print("• Multiple fixed points correspond to different memory states")
    print("• Temperature controls the sharpness of pattern retrieval")
    print("• Phase transitions may occur at critical temperatures")
    print("• Storage capacity is limited by pattern interference")
    print("• Continuous magnetizations allow analytical treatment")

if __name__ == "__main__":
    demonstrate_mean_field_analysis()

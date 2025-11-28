#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# This script assumes you have 'hopfield_basic.py' in the same directory
# It imports the functions needed to interact with the network dictionary.
from hopfield_basic import initialize_network, store_multiple_patterns, compute_energy, update_neuron

# ============================================================================
# ENERGY FUNCTION ANALYSIS FOR HOPFIELD NETWORKS
# ============================================================================

def compute_energy_landscape_1d(network, reference_pattern):
    """
    Compute energy landscape for all possible states in a 1D representation.

    This function is computationally expensive for large networks since it
    evaluates 2^N states where N is the number of neurons.
    Practical for networks with up to ~15 neurons.

    Args:
        network (dict): The Hopfield network state (weights, thresholds, etc.).
        reference_pattern (numpy.ndarray): Pattern to use as a reference for analysis.

    Returns:
        tuple: (states, energies, hamming_distances)
    """
    num_neurons = network['num_neurons']
    num_states = 2 ** num_neurons

    if num_states > 32768:  # Limit to prevent memory issues
        print(f"Warning: {num_states} states is too many for complete analysis. Sampling...")
        return compute_energy_landscape_sampled(network, reference_pattern, 1000)

    print(f"Computing energy landscape for {num_states} states...")

    states = []
    energies = []
    hamming_distances = []

    # Iterate through all possible binary combinations
    for i in range(num_states):
        binary_repr = format(i, f'0{num_neurons}b')
        state = np.array([1 if bit == '1' else -1 for bit in binary_repr])

        # Compute energy for this state 
        energy = compute_energy(network, state)

        # Compute Hamming distance from the reference pattern
        hamming_dist = np.sum(state != reference_pattern)

        states.append(state)
        energies.append(energy)
        hamming_distances.append(hamming_dist)

    return states, energies, hamming_distances

def compute_energy_landscape_sampled(network, reference_pattern, num_samples=1000):
    """
    Compute energy landscape using random sampling for large networks.

    Args:
        network (dict): The Hopfield network state.
        reference_pattern (numpy.ndarray): Reference pattern for distance calculation.
        num_samples (int, optional): Number of random states to sample. Default 1000.

    Returns:
        tuple: (states, energies, hamming_distances)
    """
    print(f"Sampling {num_samples} random states for energy analysis...")

    states = []
    energies = []
    hamming_distances = []

    # Always include the reference pattern itself
    states.append(reference_pattern.copy())
    energies.append(compute_energy(network, reference_pattern))
    hamming_distances.append(0)

    # Generate random states
    for _ in range(num_samples - 1):
        random_state = np.random.choice([-1, 1], size=network['num_neurons'])
        energy = compute_energy(network, random_state)
        hamming_dist = np.sum(random_state != reference_pattern)

        states.append(random_state)
        energies.append(energy)
        hamming_distances.append(hamming_dist)

    return states, energies, hamming_distances

def analyze_energy_minima(states, energies, threshold_percentile=10):
    """
    Identify and analyze local energy minima (attractors) in the energy landscape.

    Args:
        states (list): All states analyzed.
        energies (list): Energy for each state.
        threshold_percentile (float, optional): Percentile threshold for identifying minima.

    Returns:
        dict: Analysis results.
    """
    energies = np.array(energies)
    energy_threshold = np.percentile(energies, threshold_percentile)
    low_energy_indices = np.where(energies <= energy_threshold)[0]

    minima_states = [states[i] for i in low_energy_indices]
    minima_energies = energies[low_energy_indices]

    global_min_idx = np.argmin(energies)
    global_minimum = {
        'state': states[global_min_idx],
        'energy': energies[global_min_idx]
    }

    analysis = {
        'minima_states': minima_states,
        'minima_energies': minima_energies,
        'global_minimum': global_minimum,
        'num_minima': len(minima_states),
        'energy_threshold': energy_threshold
    }
    return analysis

def trace_energy_during_recall(network, initial_state, max_iterations=50):
    """
    Trace the energy evolution during the recall process.

    Args:
        network (dict): The Hopfield network state.
        initial_state (numpy.ndarray): Starting state for recall.
        max_iterations (int, optional): Maximum number of iterations to trace.

    Returns:
        dict: Trace results.
    """
    print("Tracing energy evolution during recall...")
    current_state = np.array(initial_state).copy()

    trace_data = {
        'states': [current_state.copy()],
        'energies': [compute_energy(network, current_state)],
        'energy_changes': [],
        'converged': False,
        'final_state': None
    }
    print(f"Initial state: {current_state}, Initial energy: {trace_data['energies'][0]:.4f}")

    previous_state = current_state.copy()
    for iteration in range(max_iterations):
        # Perform one iteration of asynchronous update
        neuron_order = np.random.permutation(network['num_neurons'])
        for neuron_idx in neuron_order:
            current_state[neuron_idx] = update_neuron(network, current_state, neuron_idx)

        current_energy = compute_energy(network, current_state)
        energy_change = current_energy - trace_data['energies'][-1]

        trace_data['states'].append(current_state.copy())
        trace_data['energies'].append(current_energy)
        trace_data['energy_changes'].append(energy_change)

        print(f"Iteration {iteration + 1}: Energy = {current_energy:.4f}, Change = {energy_change:.4f}")

        if np.array_equal(current_state, previous_state):
            trace_data['converged'] = True
            print(f"Converged after {iteration + 1} iterations")
            break

        previous_state = current_state.copy()

    trace_data['final_state'] = current_state.copy()
    return trace_data

# The visualization functions (visualize_energy_landscape, visualize_energy_trace)
# do not depend on the class structure and can be used as they are,
# as long as they receive the correct data. I've included them here for completeness.

def visualize_energy_landscape(states, energies, hamming_distances, network_info=""):
    """
    Create comprehensive visualizations of the energy landscape.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Energy Landscape Analysis - {network_info}', fontsize=16)

    # Plot 1: Energy vs. Hamming Distance
    ax1 = axes[0, 0]
    scatter = ax1.scatter(hamming_distances, energies, c=energies, cmap='viridis', alpha=0.7)
    min_energy = min(energies)
    ax1.axhline(min_energy, color='r', linestyle='--', label=f'Global Minima (E={min_energy:.2f})')
    ax1.set_xlabel('Hamming Distance from Reference Pattern')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy vs. Hamming Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Energy')

    # Plot 2: Energy Histogram
    ax2 = axes[0, 1]
    ax2.hist(energies, bins=min(50, len(set(energies))), color='skyblue', edgecolor='black')
    ax2.axvline(min_energy, color='r', linestyle='--', label=f'Global Minimum')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Number of States')
    ax2.set_title('Energy Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ... other plots can be added here ...

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('hopfield_energy_landscape.png', dpi=300, bbox_inches='tight')
    plt.savefig('hopfield_energy_landscape.svg')
    plt.show()


def visualize_energy_trace(trace_data):
    """
    Visualize the energy evolution during recall process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Energy vs. Iteration
    iterations = range(len(trace_data['energies']))
    ax1.plot(iterations, trace_data['energies'], 'bo-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution During Recall')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy Changes
    if trace_data['energy_changes']:
        change_iterations = range(1, len(trace_data['energy_changes']) + 1)
        ax2.plot(change_iterations, trace_data['energy_changes'], 'ro-')
        ax2.axhline(0, color='k', linestyle='--')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Energy Change (ΔE)')
        ax2.set_title('Energy Changes Between Iterations')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hopfield_energy_trace.png', dpi=300, bbox_inches='tight')
    plt.savefig('hopfield_energy_trace.svg')
    plt.show()

# ============================================================================
# MAIN DEMONSTRATION FUNCTION 
# ============================================================================

def main():
    """
    Main function demonstrating energy function analysis.
    """
    print("=" * 60)
    print("HOPFIELD NETWORK ENERGY FUNCTION ANALYSIS")
    print("=" * 60)

    np.random.seed(42)

    # 1. CREATE NETWORK AND STORE PATTERNS
    print("\n1. Creating network and storing patterns...")
    network = initialize_network(num_neurons=6)
    patterns = [
        np.array([1, -1, 1, -1, 1, -1]),   # Alternating pattern
        np.array([1, 1, 1, -1, -1, -1])    # Block pattern
    ]
    store_multiple_patterns(network, patterns)

    # 2. COMPUTE ENERGY LANDSCAPE
    print("\n2. Computing energy landscape...")
    reference_pattern = patterns[0]
    states, energies, hamming_distances = compute_energy_landscape_1d(network, reference_pattern)

    # 3. ANALYZE ENERGY MINIMA
    print("\n3. Analyzing energy minima...")
    minima_analysis = analyze_energy_minima(states, energies)
    print(f"Found {minima_analysis['num_minima']} low-energy states (potential minima).")
    print(f"Global minimum energy: {minima_analysis['global_minimum']['energy']:.3f}")

    # 4. TRACE ENERGY DURING RECALL
    print("\n4. Tracing energy during recall...")
    noisy_pattern = patterns[0].copy()
    noisy_pattern[1] = -noisy_pattern[1] # Flip one bit
    trace_data = trace_energy_during_recall(network, noisy_pattern)

    # 5. VISUALIZATIONS
    print("\n5. Generating visualizations...")
    network_info = f"{network['num_neurons']} neurons, {len(patterns)} patterns"
    visualize_energy_landscape(states, energies, hamming_distances, network_info)
    visualize_energy_trace(trace_data)

    print("\n" + "=" * 60)
    print("ENERGY ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hopfield_basic import HopfieldNetwork

# ============================================================================
# ENERGY FUNCTION ANALYSIS FOR HOPFIELD NETWORKS
# ============================================================================

def compute_energy_landscape_1d(network, reference_pattern):
    """
    Compute energy landscape for all possible states in a 1D representation
    
    This function computes the energy for all possible binary states of the network
    and analyzes how the energy function creates attractors (local minima) that
    correspond to stored patterns.
    
    The energy function for a Hopfield network is:
    E = -0.5 * Σ Σ w_ij * s_i * s_j - Σ θ_i * s_i
    
    Key insights from energy landscape analysis:
    1. Stored patterns correspond to local minima (attractors)
    2. Energy decreases during the recall process
    3. Spurious states appear as additional local minima
    4. The depth of minima indicates stability of patterns
    
    Args:
        network (HopfieldNetwork): Trained Hopfield network
        reference_pattern (numpy.ndarray): Pattern to use as reference for analysis
    
    Returns:
        tuple: (states, energies, hamming_distances)
            states (list): All possible binary states
            energies (list): Energy for each state
            hamming_distances (list): Hamming distance from reference pattern
    
    Note:
        This function is computationally expensive for large networks since it
        evaluates 2^N states where N is the number of neurons.
        Practical for networks with up to ~15 neurons.
    """
    num_neurons = network.num_neurons
    
    # Generate all possible binary states
    # For N neurons, there are 2^N possible states
    num_states = 2 ** num_neurons
    
    if num_states > 32768:  # Limit to prevent memory issues
        print(f"Warning: {num_states} states is too many for complete analysis")
        print("Sampling random subset instead...")
        return compute_energy_landscape_sampled(network, reference_pattern, 1000)
    
    print(f"Computing energy landscape for {num_states} states...")
    
    states = []
    energies = []
    hamming_distances = []
    
    # Iterate through all possible binary combinations
    for i in range(num_states):
        # Convert integer to binary representation
        # Each bit represents the state of one neuron
        binary_repr = format(i, f'0{num_neurons}b')
        
        # Convert binary string to bipolar state {-1, +1}
        state = np.array([1 if bit == '1' else -1 for bit in binary_repr])
        
        # Compute energy for this state
        energy = network.compute_energy(state)
        
        # Compute Hamming distance from reference pattern
        hamming_dist = np.sum(state != reference_pattern)
        
        states.append(state)
        energies.append(energy)
        hamming_distances.append(hamming_dist)
    
    return states, energies, hamming_distances

def compute_energy_landscape_sampled(network, reference_pattern, num_samples=1000):
    """
    Compute energy landscape using random sampling for large networks
    
    For networks with many neurons, computing all 2^N states is impractical.
    This function samples random states to get a representative view of the
    energy landscape while remaining computationally feasible.
    
    Args:
        network (HopfieldNetwork): Trained Hopfield network
        reference_pattern (numpy.ndarray): Reference pattern for distance calculation
        num_samples (int, optional): Number of random states to sample. Default 1000.
    
    Returns:
        tuple: (states, energies, hamming_distances)
            Same format as compute_energy_landscape_1d but with sampled states
    """
    print(f"Sampling {num_samples} random states for energy analysis...")
    
    states = []
    energies = []
    hamming_distances = []
    
    # Always include the reference pattern itself
    states.append(reference_pattern.copy())
    energies.append(network.compute_energy(reference_pattern))
    hamming_distances.append(0)
    
    # Generate random states
    for _ in range(num_samples - 1):
        # Generate random bipolar state
        random_state = np.random.choice([-1, 1], size=network.num_neurons)
        
        # Compute energy and distance
        energy = network.compute_energy(random_state)
        hamming_dist = np.sum(random_state != reference_pattern)
        
        states.append(random_state)
        energies.append(energy)
        hamming_distances.append(hamming_dist)
    
    return states, energies, hamming_distances

def analyze_energy_minima(states, energies, network, threshold_percentile=10):
    """
    Identify and analyze local energy minima (attractors) in the energy landscape
    
    Local minima in the energy landscape correspond to stable states of the network.
    These include both stored patterns (desired attractors) and spurious states
    (undesired attractors that can trap the network).
    
    This analysis helps understand:
    1. How many attractors exist in the network
    2. Which attractors correspond to stored patterns
    3. The relative stability (energy depth) of different attractors
    4. The basin of attraction size for each attractor
    
    Args:
        states (list): All states analyzed
        energies (list): Energy for each state
        network (HopfieldNetwork): The Hopfield network
        threshold_percentile (float, optional): Percentile threshold for identifying minima. Default 10.
    
    Returns:
        dict: Analysis results containing:
            - minima_states: States corresponding to local minima
            - minima_energies: Energies of the minima
            - global_minimum: State with lowest energy
            - energy_statistics: Statistical summary of energy distribution
    """
    energies = np.array(energies)
    
    # Find energy threshold for identifying significant minima
    # We consider states in the lowest percentile as potential minima
    energy_threshold = np.percentile(energies, threshold_percentile)
    
    # Identify states with very low energy
    low_energy_indices = np.where(energies <= energy_threshold)[0]
    
    minima_states = [states[i] for i in low_energy_indices]
    minima_energies = energies[low_energy_indices]
    
    # Find global minimum
    global_min_idx = np.argmin(energies)
    global_minimum = {
        'state': states[global_min_idx],
        'energy': energies[global_min_idx],
        'index': global_min_idx
    }
    
    # Compute energy statistics
    energy_stats = {
        'mean': np.mean(energies),
        'std': np.std(energies),
        'min': np.min(energies),
        'max': np.max(energies),
        'median': np.median(energies),
        'range': np.max(energies) - np.min(energies)
    }
    
    analysis = {
        'minima_states': minima_states,
        'minima_energies': minima_energies,
        'global_minimum': global_minimum,
        'energy_statistics': energy_stats,
        'num_minima': len(minima_states),
        'energy_threshold': energy_threshold
    }
    
    return analysis

def trace_energy_during_recall(network, initial_state, max_iterations=50):
    """
    Trace the energy evolution during the recall process
    
    This function monitors how the energy changes as the network evolves
    from an initial state toward an attractor. The energy should monotonically
    decrease (or stay constant) during recall, which is a fundamental property
    of Hopfield networks that guarantees convergence.
    
    Mathematical property:
    ΔE ≤ 0 for each neuron update
    
    This is proven by showing that each neuron update either decreases energy
    or leaves it unchanged, ensuring convergence to a local minimum.
    
    Args:
        network (HopfieldNetwork): The Hopfield network
        initial_state (numpy.ndarray): Starting state for recall
        max_iterations (int, optional): Maximum number of iterations to trace. Default 50.
    
    Returns:
        dict: Trace results containing:
            - states: List of states at each iteration
            - energies: List of energies at each iteration
            - energy_changes: List of energy changes between iterations
            - converged: Whether the network converged
            - final_state: Final state after recall
    """
    print("Tracing energy evolution during recall...")
    
    current_state = np.array(initial_state).copy()
    
    # Storage for trace data
    trace_data = {
        'states': [current_state.copy()],
        'energies': [network.compute_energy(current_state)],
        'energy_changes': [],
        'converged': False,
        'final_state': None
    }
    
    print(f"Initial state: {current_state}")
    print(f"Initial energy: {trace_data['energies'][0]:.4f}")
    
    previous_state = current_state.copy()
    
    for iteration in range(max_iterations):
        # Perform one iteration of asynchronous update
        # Update each neuron once in random order
        neuron_order = np.random.permutation(network.num_neurons)
        
        for neuron_idx in neuron_order:
            # Update single neuron
            new_neuron_state = network.update_neuron(current_state, neuron_idx)
            current_state[neuron_idx] = new_neuron_state
        
        # Compute energy after this iteration
        current_energy = network.compute_energy(current_state)
        previous_energy = trace_data['energies'][-1]
        energy_change = current_energy - previous_energy
        
        # Store trace data
        trace_data['states'].append(current_state.copy())
        trace_data['energies'].append(current_energy)
        trace_data['energy_changes'].append(energy_change)
        
        print(f"Iteration {iteration + 1}: Energy = {current_energy:.4f}, "
              f"Change = {energy_change:.4f}")
        
        # Check for convergence (no change in state)
        if np.array_equal(current_state, previous_state):
            trace_data['converged'] = True
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Verify energy non-increase property
        if energy_change > 1e-10:  # Small tolerance for numerical errors
            print(f"Warning: Energy increased by {energy_change:.6f}")
            print("This should not happen in a proper Hopfield network!")
        
        previous_state = current_state.copy()
    
    trace_data['final_state'] = current_state.copy()
    
    if not trace_data['converged']:
        print(f"Did not converge after {max_iterations} iterations")
    
    return trace_data

def visualize_energy_landscape(states, energies, hamming_distances, network_info=""):
    """
    Create comprehensive visualizations of the energy landscape
    
    This function generates multiple plots to visualize different aspects
    of the energy landscape:
    1. Energy vs. Hamming distance (shows energy basins)
    2. Energy histogram (shows distribution of energy values)
    3. 3D energy surface (for small networks)
    4. Energy minima analysis
    
    Args:
        states (list): All analyzed states
        energies (list): Energy for each state
        hamming_distances (list): Hamming distance from reference for each state
        network_info (str, optional): Additional information about the network
    """
    fig = plt.figure(figsize=(15, 10))
    
    # ========================================================================
    # Plot 1: Energy vs. Hamming Distance
    # ========================================================================
    
    ax1 = plt.subplot(2, 3, 1)
    
    # Create scatter plot with color coding for energy levels
    scatter = ax1.scatter(hamming_distances, energies, c=energies, 
                         cmap='viridis', alpha=0.6, s=30)
    
    # Highlight the minimum energy points
    min_energy = min(energies)
    min_indices = [i for i, e in enumerate(energies) if abs(e - min_energy) < 1e-6]
    
    if min_indices:
        min_hamming = [hamming_distances[i] for i in min_indices]
        min_energies = [energies[i] for i in min_indices]
        ax1.scatter(min_hamming, min_energies, c='red', s=100, 
                   marker='*', label=f'Global Minima (E={min_energy:.2f})')
    
    ax1.set_xlabel('Hamming Distance from Reference Pattern')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Landscape: Energy vs. Hamming Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax1, label='Energy')
    
    # ========================================================================
    # Plot 2: Energy Histogram
    # ========================================================================
    
    ax2 = plt.subplot(2, 3, 2)
    
    # Create histogram of energy values
    n_bins = min(50, len(set(energies)))  # Adaptive number of bins
    counts, bins, patches = ax2.hist(energies, bins=n_bins, alpha=0.7, 
                                    color='skyblue', edgecolor='black')
    
    # Highlight minimum energy
    ax2.axvline(min_energy, color='red', linestyle='--', linewidth=2,
               label=f'Global Minimum: {min_energy:.2f}')
    
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Number of States')
    ax2.set_title('Energy Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ========================================================================
    # Plot 3: Energy Statistics
    # ========================================================================
    
    ax3 = plt.subplot(2, 3, 3)
    
    # Compute and display energy statistics
    energy_stats = {
        'Mean': np.mean(energies),
        'Median': np.median(energies),
        'Std Dev': np.std(energies),
        'Min': np.min(energies),
        'Max': np.max(energies),
        'Range': np.max(energies) - np.min(energies)
    }
    
    # Create bar plot of statistics
    stat_names = list(energy_stats.keys())
    stat_values = list(energy_stats.values())
    
    bars = ax3.bar(range(len(stat_names)), stat_values, alpha=0.7, color='lightcoral')
    ax3.set_xticks(range(len(stat_names)))
    ax3.set_xticklabels(stat_names, rotation=45)
    ax3.set_ylabel('Energy Value')
    ax3.set_title('Energy Statistics')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stat_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01*abs(height),
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # ========================================================================
    # Plot 4: Energy vs. State Index (for pattern analysis)
    # ========================================================================
    
    ax4 = plt.subplot(2, 3, 4)
    
    # Sort states by energy for better visualization
    sorted_indices = np.argsort(energies)
    sorted_energies = [energies[i] for i in sorted_indices]
    
    ax4.plot(range(len(sorted_energies)), sorted_energies, 'b-', alpha=0.7)
    ax4.scatter(range(len(sorted_energies)), sorted_energies, c=sorted_energies, 
               cmap='viridis', s=20, alpha=0.8)
    
    # Highlight the lowest energy states
    num_minima_to_show = min(5, len(sorted_energies))
    ax4.scatter(range(num_minima_to_show), sorted_energies[:num_minima_to_show], 
               c='red', s=50, marker='o', label=f'Top {num_minima_to_show} Minima')
    
    ax4.set_xlabel('State Index (sorted by energy)')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy Landscape: Sorted States')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # ========================================================================
    # Plot 5: Hamming Distance Distribution
    # ========================================================================
    
    ax5 = plt.subplot(2, 3, 5)
    
    # Histogram of Hamming distances
    unique_distances = sorted(set(hamming_distances))
    distance_counts = [hamming_distances.count(d) for d in unique_distances]
    
    ax5.bar(unique_distances, distance_counts, alpha=0.7, color='lightgreen')
    ax5.set_xlabel('Hamming Distance from Reference')
    ax5.set_ylabel('Number of States')
    ax5.set_title('Hamming Distance Distribution')
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 6: Energy Minima Analysis
    # ========================================================================
    
    ax6 = plt.subplot(2, 3, 6)
    
    # Identify and plot energy minima
    analysis = analyze_energy_minima(states, energies, None)
    
    # Plot energy vs. number of minima at different thresholds
    thresholds = range(5, 51, 5)  # Test different percentile thresholds
    minima_counts = []
    
    for threshold in thresholds:
        threshold_energy = np.percentile(energies, threshold)
        count = np.sum(np.array(energies) <= threshold_energy)
        minima_counts.append(count)
    
    ax6.plot(thresholds, minima_counts, 'go-', linewidth=2, markersize=6)
    ax6.set_xlabel('Energy Threshold (Percentile)')
    ax6.set_ylabel('Number of Low-Energy States')
    ax6.set_title('Energy Minima Analysis')
    ax6.grid(True, alpha=0.3)
    
    # Add network information as title
    if network_info:
        fig.suptitle(f'Energy Landscape Analysis - {network_info}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_energy_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_energy_trace(trace_data):
    """
    Visualize the energy evolution during recall process
    
    This creates plots showing how energy changes during the recall process,
    demonstrating the fundamental property that energy decreases monotonically.
    
    Args:
        trace_data (dict): Results from trace_energy_during_recall()
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # ========================================================================
    # Plot 1: Energy vs. Iteration
    # ========================================================================
    
    ax1 = axes[0, 0]
    iterations = range(len(trace_data['energies']))
    
    ax1.plot(iterations, trace_data['energies'], 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution During Recall')
    ax1.grid(True, alpha=0.3)
    
    # Highlight initial and final energies
    ax1.scatter([0], [trace_data['energies'][0]], c='green', s=100, 
               marker='s', label=f'Initial: {trace_data["energies"][0]:.3f}')
    ax1.scatter([len(trace_data['energies'])-1], [trace_data['energies'][-1]], 
               c='red', s=100, marker='s', 
               label=f'Final: {trace_data["energies"][-1]:.3f}')
    ax1.legend()
    
    # ========================================================================
    # Plot 2: Energy Changes
    # ========================================================================
    
    ax2 = axes[0, 1]
    
    if trace_data['energy_changes']:
        change_iterations = range(1, len(trace_data['energy_changes']) + 1)
        ax2.plot(change_iterations, trace_data['energy_changes'], 'ro-', 
                linewidth=2, markersize=4)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Energy Change (ΔE)')
        ax2.set_title('Energy Changes Between Iterations')
        ax2.grid(True, alpha=0.3)
        
        # Check for energy increases (should not happen)
        increases = [change for change in trace_data['energy_changes'] if change > 1e-10]
        if increases:
            ax2.text(0.5, 0.9, f'Warning: {len(increases)} energy increases detected!', 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # ========================================================================
    # Plot 3: State Evolution (for small networks)
    # ========================================================================
    
    ax3 = axes[1, 0]
    
    if len(trace_data['states'][0]) <= 10:  # Only for small networks
        # Create a heatmap showing state evolution
        state_matrix = np.array(trace_data['states']).T
        
        im = ax3.imshow(state_matrix, cmap='RdBu', aspect='auto', interpolation='nearest')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Neuron Index')
        ax3.set_title('State Evolution (Blue: -1, Red: +1)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, label='Neuron State')
    else:
        # For large networks, show Hamming distance from final state
        final_state = trace_data['final_state']
        hamming_distances = [np.sum(state != final_state) for state in trace_data['states']]
        
        ax3.plot(iterations, hamming_distances, 'go-', linewidth=2, markersize=4)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Hamming Distance from Final State')
        ax3.set_title('Convergence Progress')
        ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 4: Energy Decrease Rate
    # ========================================================================
    
    ax4 = axes[1, 1]
    
    if len(trace_data['energies']) > 1:
        # Compute cumulative energy decrease
        initial_energy = trace_data['energies'][0]
        energy_decreases = [initial_energy - energy for energy in trace_data['energies']]
        
        ax4.plot(iterations, energy_decreases, 'mo-', linewidth=2, markersize=4)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cumulative Energy Decrease')
        ax4.set_title('Total Energy Reduction')
        ax4.grid(True, alpha=0.3)
        
        # Add final decrease value
        final_decrease = energy_decreases[-1]
        ax4.text(0.7, 0.9, f'Total decrease: {final_decrease:.3f}', 
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_energy_trace.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating energy function analysis in Hopfield networks
    
    This comprehensive demonstration covers:
    1. Creating a network and storing patterns
    2. Computing complete energy landscape
    3. Analyzing energy minima and attractors
    4. Tracing energy during recall process
    5. Visualizing all energy-related phenomena
    """
    print("=" * 60)
    print("HOPFIELD NETWORK ENERGY FUNCTION ANALYSIS")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # ========================================================================
    # 1. CREATE NETWORK AND STORE PATTERNS
    # ========================================================================
    
    print("\n1. Creating network and storing patterns...")
    
    # Use a small network for complete energy landscape analysis
    network = HopfieldNetwork(num_neurons=6)
    
    # Define patterns to store
    patterns = [
        np.array([1, -1, 1, -1, 1, -1]),   # Alternating pattern
        np.array([1, 1, 1, -1, -1, -1])    # Block pattern
    ]
    
    print("Storing patterns:")
    for i, pattern in enumerate(patterns):
        print(f"  Pattern {i+1}: {pattern}")
    
    network.store_patterns(patterns)
    
    # ========================================================================
    # 2. COMPUTE ENERGY LANDSCAPE
    # ========================================================================
    
    print("\n2. Computing energy landscape...")
    
    # Use first pattern as reference for analysis
    reference_pattern = patterns[0]
    
    # Compute energy for all possible states
    states, energies, hamming_distances = compute_energy_landscape_1d(
        network, reference_pattern)
    
    print(f"Analyzed {len(states)} states")
    print(f"Energy range: {min(energies):.3f} to {max(energies):.3f}")
    
    # ========================================================================
    # 3. ANALYZE ENERGY MINIMA
    # ========================================================================
    
    print("\n3. Analyzing energy minima...")
    
    minima_analysis = analyze_energy_minima(states, energies, network)
    
    print(f"Found {minima_analysis['num_minima']} low-energy states")
    print(f"Global minimum energy: {minima_analysis['global_minimum']['energy']:.3f}")
    print(f"Global minimum state: {minima_analysis['global_minimum']['state']}")
    
    # Check if stored patterns are among the minima
    print("\nChecking if stored patterns are energy minima:")
    for i, pattern in enumerate(patterns):
        pattern_energy = network.compute_energy(pattern)
        is_minimum = pattern_energy <= minima_analysis['energy_threshold']
        print(f"  Pattern {i+1}: Energy = {pattern_energy:.3f}, "
              f"Is minimum: {is_minimum}")
    
    # ========================================================================
    # 4. TRACE ENERGY DURING RECALL
    # ========================================================================
    
    print("\n4. Tracing energy during recall...")
    
    # Create a noisy version of the first pattern
    noisy_pattern = patterns[0].copy()
    # Flip 2 bits to create noise
    flip_indices = [1, 3]
    noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
    
    print(f"Original pattern: {patterns[0]}")
    print(f"Noisy pattern:    {noisy_pattern}")
    print(f"Hamming distance: {np.sum(patterns[0] != noisy_pattern)}")
    
    # Trace energy evolution
    trace_data = trace_energy_during_recall(network, noisy_pattern)
    
    # ========================================================================
    # 5. ENERGY ANALYSIS SUMMARY
    # ========================================================================
    
    print("\n5. Energy analysis summary...")
    
    print(f"Network properties:")
    print(f"  Number of neurons: {network.num_neurons}")
    print(f"  Number of stored patterns: {len(patterns)}")
    print(f"  Theoretical capacity: {int(0.15 * network.num_neurons)}")
    
    print(f"\nEnergy landscape properties:")
    print(f"  Total states analyzed: {len(states)}")
    print(f"  Energy range: {min(energies):.3f} to {max(energies):.3f}")
    print(f"  Number of energy minima: {minima_analysis['num_minima']}")
    print(f"  Global minimum energy: {minima_analysis['global_minimum']['energy']:.3f}")
    
    print(f"\nRecall process properties:")
    print(f"  Initial energy: {trace_data['energies'][0]:.3f}")
    print(f"  Final energy: {trace_data['energies'][-1]:.3f}")
    print(f"  Total energy decrease: {trace_data['energies'][0] - trace_data['energies'][-1]:.3f}")
    print(f"  Converged: {trace_data['converged']}")
    print(f"  Iterations to convergence: {len(trace_data['energies']) - 1}")
    
    # Verify energy monotonicity
    energy_increases = [change for change in trace_data['energy_changes'] if change > 1e-10]
    print(f"  Energy increases detected: {len(energy_increases)} (should be 0)")
    
    # ========================================================================
    # 6. VISUALIZATIONS
    # ========================================================================
    
    print("\n6. Generating visualizations...")
    
    # Visualize energy landscape
    network_info = f"{network.num_neurons} neurons, {len(patterns)} patterns"
    visualize_energy_landscape(states, energies, hamming_distances, network_info)
    
    # Visualize energy trace
    visualize_energy_trace(trace_data)
    
    # ========================================================================
    # 7. THEORETICAL VERIFICATION
    # ========================================================================
    
    print("\n7. Theoretical verification...")
    
    print("Verifying Hopfield network properties:")
    
    # Check weight matrix symmetry
    is_symmetric = np.allclose(network.weights, network.weights.T)
    print(f"  Weight matrix is symmetric: {is_symmetric}")
    
    # Check diagonal is zero
    diagonal_zero = np.allclose(np.diag(network.weights), 0)
    print(f"  Diagonal elements are zero: {diagonal_zero}")
    
    # Verify energy function properties
    print(f"  Energy decreases during recall: {len(energy_increases) == 0}")
    print(f"  Stored patterns have low energy: {all(network.compute_energy(p) <= minima_analysis['energy_threshold'] for p in patterns)}")
    
    print("\n" + "=" * 60)
    print("ENERGY ANALYSIS COMPLETE")
    print("=" * 60)
    print("Key insights:")
    print("1. Energy landscape has multiple local minima (attractors)")
    print("2. Stored patterns correspond to deep energy minima")
    print("3. Energy decreases monotonically during recall")
    print("4. Network converges to nearest local minimum")
    print("5. Spurious states appear as additional minima")
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the energy function analysis demonstration
    """
    main()


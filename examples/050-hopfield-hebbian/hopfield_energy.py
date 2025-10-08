#/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
# Make sure hopfield_basic.py is in the same directory
from hopfield_basic import initialize_network, store_multiple_patterns, compute_energy, update_neuron

def trace_energy_during_recall(network, initial_state, max_iterations=50):
    """
    Trace the energy evolution during the recall process.
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
            print(f"Converged after {iteration + 1} iterations.")
            break

        previous_state = current_state.copy()

    trace_data['final_state'] = current_state.copy()
    return trace_data

def main_energy():
    """
    Main function for demonstrating energy analysis.
    """
    print("=" * 60)
    print(" HOPFIELD NETWORK ENERGY ANALYSIS")
    print("=" * 60)

    np.random.seed(42)

    # 1. Create network and store patterns
    network = initialize_network(num_neurons=6)
    patterns = [
        np.array([1, -1, 1, -1, 1, -1]),
        np.array([1, 1, 1, -1, -1, -1])
    ]
    store_multiple_patterns(network, patterns)

    # 2. Trace energy during recall
    print("\n4. Tracing energy during recall...")
    noisy_pattern = patterns[0].copy()
    noisy_pattern[[1, 3]] = -noisy_pattern[[1, 3]] # Flip two bits

    trace_data = trace_energy_during_recall(network, noisy_pattern)

    # 3. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(trace_data['energies'])), trace_data['energies'], 'bo-')
    plt.title("Energy Evolution During Recall ")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.savefig('hopfield_energy.png', dpi=300, bbox_inches='tight')
    plt.savefig('hopfield_energy.svg')
    plt.show()

if __name__ == '__main__':
    main_energy()


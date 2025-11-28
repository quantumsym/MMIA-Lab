#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SIMPLE HOPFIELD NETWORK IMPLEMENTATION
# ============================================================================

def initialize_network(num_neurons):
    """
    Initialize a Hopfield network with a specified number of neurons.
    Returns a dictionary representing the network's state.
    """
    network = {
        'num_neurons': num_neurons,
        'weights': np.zeros((num_neurons, num_neurons)),
        'thresholds': np.zeros(num_neurons)
    }
    print(f"Initialized Hopfield network with {num_neurons} neurons")
    return network

def store_single_pattern(network, pattern):
    """
    Store a single pattern in the network using the Hebbian learning rule.
    Modifies the 'network' dictionary directly.
    """
    pattern = np.array(pattern).flatten()
    if len(pattern) != network['num_neurons']:
        raise ValueError(f"Pattern length ({len(pattern)}) must match "
                         f"number of neurons ({network['num_neurons']})")

    # Convert binary {0, 1} to bipolar {-1, +1} if necessary
    if np.all(np.isin(pattern, [0, 1])):
        pattern = 2 * pattern - 1
        print("Converted pattern from {0, 1} to {-1, +1} representation")

    print(f"Storing pattern: {pattern}")
    # Apply Hebbian learning rule
    network['weights'] = np.outer(pattern, pattern)
    # Set diagonal elements to zero (no self-connections)
    np.fill_diagonal(network['weights'], 0)
    print("Pattern stored successfully.")

def store_multiple_patterns(network, patterns):
    """
    Store multiple patterns using the generalized Hebbian rule.
    """
    patterns = np.array(patterns)
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)

    num_patterns, pattern_length = patterns.shape
    if pattern_length != network['num_neurons']:
        raise ValueError(f"Each pattern length ({pattern_length}) must match "
                         f"number of neurons ({network['num_neurons']})")

    print(f"Storing {num_patterns} patterns...")
    # Convert all patterns to bipolar representation if necessary
    bipolar_patterns = np.array([2 * p - 1 if np.all(np.isin(p, [0, 1])) else p for p in patterns])

    # Initialize weight matrix
    network['weights'] = np.zeros((network['num_neurons'], network['num_neurons']))

    # Apply generalized Hebbian learning rule
    for p in bipolar_patterns:
        network['weights'] += np.outer(p, p)

    # Normalize by the number of patterns
    network['weights'] /= num_patterns

    # Set diagonal to zero
    np.fill_diagonal(network['weights'], 0)
    print(f"Successfully stored {num_patterns} patterns.")

def update_neuron(network, current_state, neuron_index):
    """
    Update a single neuron's state based on the current network state.
    """
    # Compute net input to the neuron
    net_input = np.dot(network['weights'][neuron_index], current_state) + network['thresholds'][neuron_index]

    # Apply sign activation function
    if net_input > 0:
        return 1
    elif net_input < 0:
        return -1
    else:
        # If net input is zero, keep the current state
        return current_state[neuron_index]

def recall_pattern(network, input_pattern, max_iterations=100, verbose=False):
    """
    Recall a stored pattern from a potentially noisy or incomplete input.
    """
    current_state = np.array(input_pattern).flatten()
    if np.all(np.isin(current_state, [0, 1])):
        current_state = 2 * current_state - 1

    if verbose:
        print(f"Starting recall with input: {current_state}")

    converged = False
    iteration = 0
    for iteration in range(max_iterations):
        previous_state = current_state.copy()

        # Asynchronous update with random neuron order
        neuron_order = np.random.permutation(network['num_neurons'])
        for neuron_idx in neuron_order:
            current_state[neuron_idx] = update_neuron(network, current_state, neuron_idx)

        if verbose:
            print(f"Iteration {iteration + 1}: {current_state}")

        # Check for convergence
        if np.array_equal(current_state, previous_state):
            converged = True
            if verbose:
                print(f"Converged after {iteration + 1} iterations.")
            break

    return current_state, converged, iteration + 1

def compute_energy(network, state):
    """
    Compute the energy of a given network state.
    """
    state = np.array(state)
    if np.all(np.isin(state, [0, 1])):
        state = 2 * state - 1

    interaction_energy = -0.5 * np.dot(state, np.dot(network['weights'], state))
    threshold_energy = -np.dot(network['thresholds'], state)
    return interaction_energy + threshold_energy

def get_network_info(network):
    """
    Get comprehensive information about the current network state.
    """
    info = {
        'num_neurons': network['num_neurons'],
        'weight_matrix_shape': network['weights'].shape,
        'weight_matrix_symmetric': np.allclose(network['weights'], network['weights'].T),
        'weight_matrix_diagonal_zero': np.allclose(np.diag(network['weights']), 0),
        'max_weight': np.max(network['weights']),
        'min_weight': np.min(network['weights']),
        'mean_weight': np.mean(network['weights']),
    }
    return info

def demonstrate_basic_hopfield():
    """
    Demonstrate basic Hopfield network functionality.
    """
    print("=" * 60)
    print("BASIC  HOPFIELD NETWORK DEMONSTRATION")
    print("=" * 60)

    # 1. Create network and store a pattern
    network = initialize_network(num_neurons=5)
    original_pattern = np.array([1, -1, 1, -1, 1])
    store_single_pattern(network, original_pattern)

    # 2. Perfect recall test
    print("\n2. Testing perfect recall...")
    recalled, converged, iters = recall_pattern(network, original_pattern, verbose=True)
    print(f"Perfect recall successful: {np.array_equal(original_pattern, recalled)}")

    # 3. Noisy recall test
    print("\n3. Testing recall with noisy input...")
    noisy_pattern = original_pattern.copy()
    noisy_pattern[2] = -noisy_pattern[2]
    recalled_noisy, _, _ = recall_pattern(network, noisy_pattern, verbose=True)
    print(f"Recall from noise successful: {np.array_equal(original_pattern, recalled_noisy)}")

if __name__ == '__main__':
    demonstrate_basic_hopfield()


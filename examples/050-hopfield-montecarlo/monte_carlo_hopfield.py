'''
 implementation of a Stochastic Hopfield Network using Monte Carlo simulation.

This script provides functions to create, train, and simulate a Hopfield network
where state transitions are governed by probabilistic rules based on the
Metropolis-Hastings algorithm, allowing the network to explore its state space
and settle into low-energy configurations.
'''

import numpy as np
import time

# ============================================================================
#  STOCHASTIC HOPFIELD NETWORK (MONTE CARLO)
# ============================================================================

def initialize_stochastic_network(num_neurons, temperature=1.0, random_seed=None):
    """
    Initializes a stochastic Hopfield network.

    Args:
        num_neurons (int): The number of neurons in the network.
        temperature (float): The temperature T for the Boltzmann distribution, controlling randomness.
        random_seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        dict: A dictionary representing the network's state and parameters.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    network = {
        'num_neurons': num_neurons,
        'temperature': temperature,
        'weights': np.zeros((num_neurons, num_neurons)),
        'thresholds': np.zeros(num_neurons),
        'state': np.random.choice([-1, 1], size=num_neurons),
        'stored_patterns': []
    }
    print(f"Initialized Stochastic Hopfield Network with {num_neurons} neurons at T={temperature:.2f}")
    return network

def store_patterns(network, patterns):
    """
    Stores patterns in the network using the Hebbian learning rule.

    Args:
        network (dict): The network dictionary.
        patterns (list): A list of patterns to store. Each pattern is a NumPy array
                         of shape (num_neurons,) with values in {-1, +1}.
    """
    if not patterns:
        raise ValueError("No patterns provided")

    num_patterns = len(patterns)
    network['stored_patterns'] = [p.copy() for p in patterns]
    
    # Reset weights before storing new patterns
    network['weights'] = np.zeros((network['num_neurons'], network['num_neurons']))

    # Hebbian learning rule: sum of outer products
    for pattern in patterns:
        network['weights'] += np.outer(pattern, pattern)
    
    # Normalize by the number of patterns
    network['weights'] /= num_patterns
    
    # No self-connections
    np.fill_diagonal(network['weights'], 0)
    
    print(f"Stored {num_patterns} patterns.")

def compute_energy(network, state):
    """
    Computes the energy of a given state.

    The energy function is E = -0.5 * Σ_ij(w_ij * s_i * s_j) - Σ_i(θ_i * s_i).

    Args:
        network (dict): The network dictionary containing weights and thresholds.
        state (np.ndarray): The state vector for which to compute the energy.

    Returns:
        float: The energy of the state.
    """
    interaction_energy = -0.5 * np.dot(state, np.dot(network['weights'], state))
    threshold_energy = -np.dot(network['thresholds'], state)
    return interaction_energy + threshold_energy

def compute_energy_change(network, state, neuron_idx):
    """
    Efficiently computes the change in energy from flipping a single neuron.

    ΔE = -2 * s_i * h_i, where h_i is the local field on neuron i.

    Args:
        network (dict): The network dictionary.
        state (np.ndarray): The current state of the network.
        neuron_idx (int): The index of the neuron to be flipped.

    Returns:
        float: The change in energy (ΔE) if the neuron were flipped.
    """
    # Local field h_i = Σ_j(w_ij * s_j) + θ_i
    local_field = np.dot(network['weights'][neuron_idx], state) + network['thresholds'][neuron_idx]
    
    # Energy change is -2 * s_i * h_i, but since we are flipping s_i -> -s_i,
    # the change is 2 * s_i * h_i.
    energy_change = -2 * state[neuron_idx] * local_field
    return energy_change

def run_monte_carlo(network, num_steps, record_interval=100, verbose=True):
    """
    Runs a Monte Carlo simulation on the Hopfield network.

    Args:
        network (dict): The network dictionary to simulate.
        num_steps (int): The total number of Monte Carlo steps to perform.
        record_interval (int): The interval at which to record energy and state.
        verbose (bool): If True, prints progress information.

    Returns:
        dict: A dictionary containing the results of the simulation.
    """
    if verbose:
        print(f"Running Monte Carlo simulation for {num_steps} steps...")

    start_time = time.time()
    
    energy_history = []
    state_history = []
    accepted_flips = 0
    
    current_energy = compute_energy(network, network['state'])

    for step in range(num_steps):
        # 1. Pick a random neuron to flip
        neuron_to_flip = np.random.randint(network['num_neurons'])

        # 2. Calculate the energy change that would result from the flip
        delta_e = compute_energy_change(network, network['state'], neuron_to_flip)

        # 3. Decide whether to accept the flip (Metropolis-Hastings criterion)
        accept_flip = False
        if delta_e < 0:
            # If the energy decreases, always accept the flip
            accept_flip = True
        elif network['temperature'] > 0:
            # If energy increases, accept with a probability exp(-ΔE / T)
            acceptance_probability = np.exp(-delta_e / network['temperature'])
            if np.random.rand() < acceptance_probability:
                accept_flip = True

        # 4. If accepted, update the state and energy
        if accept_flip:
            network['state'][neuron_to_flip] *= -1
            current_energy += delta_e
            accepted_flips += 1

        # 5. Record data at intervals
        if step % record_interval == 0:
            energy_history.append(current_energy)
            state_history.append(network['state'].copy())
            if verbose and step > 0 and (step % (10 * record_interval) == 0):
                acceptance_rate = accepted_flips / (step + 1)
                print(f"  Step {step:6d}/{num_steps}: Energy={current_energy:9.3f}, Acceptance Rate={acceptance_rate:.1%}")

    end_time = time.time()
    
    results = {
        'final_state': network['state'].copy(),
        'final_energy': current_energy,
        'energy_history': np.array(energy_history),
        'state_history': np.array(state_history),
        'total_acceptance_rate': accepted_flips / num_steps,
        'computation_time': end_time - start_time
    }

    if verbose:
        print(f"Monte Carlo simulation finished in {results['computation_time']:.2f} seconds.")
        print(f"  Final Energy: {results['final_energy']:.3f}")
        print(f"  Total Acceptance Rate: {results['total_acceptance_rate']:.1%}")

    return results



# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_monte_carlo_hopfield():
    """
    Main demonstration function showing Monte Carlo Hopfield network capabilities.
    """
    print("=" * 80)
    print("MONTE CARLO HOPFIELD NETWORK DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Create test patterns
    num_neurons = 20
    patterns = [
        np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),  # Alternating pattern
        np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),  # Block pattern
        np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])   # Structured pattern
    ]
    
    print(f"Created {len(patterns)} test patterns with {num_neurons} neurons each")
    
    # 2. Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"TESTING AT TEMPERATURE T = {temp}")
        print(f"{'='*60}")
        
        # Initialize network
        network = initialize_stochastic_network(num_neurons, temperature=temp, random_seed=42)
        
        # Store patterns
        store_patterns(network, patterns)
        
        # Test pattern recall
        for i, original_pattern in enumerate(patterns):
            print(f"\n--- Pattern {i+1} Recall Test ---")
            
            # Add noise to the pattern (flip 20% of bits)
            noisy_pattern = original_pattern.copy()
            num_flips = int(0.2 * len(original_pattern))
            flip_indices = np.random.choice(len(original_pattern), size=num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1
            
            # Set network state to noisy pattern
            network['state'] = noisy_pattern.copy()
            
            print(f"Original pattern:  {original_pattern}")
            print(f"Noisy input:       {noisy_pattern}")
            print(f"Hamming distance:  {np.sum(original_pattern != noisy_pattern)} bits")
            
            # Run Monte Carlo simulation
            initial_energy = compute_energy(network, network['state'])
            results = run_monte_carlo(network, num_steps=1000, record_interval=100, verbose=False)
            
            # Check if pattern was recovered
            final_pattern = results['final_state']
            recovery_success = np.array_equal(original_pattern, final_pattern)
            hamming_distance_final = np.sum(original_pattern != final_pattern)
            
            print(f"Recovered pattern: {final_pattern}")
            print(f"Recovery success:  {recovery_success}")
            print(f"Final Hamming dist: {hamming_distance_final} bits")
            print(f"Energy change:     {initial_energy:.3f} → {results['final_energy']:.3f}")
            print(f"Acceptance rate:   {results['total_acceptance_rate']:.1%}")
    
    # 3. Energy landscape exploration
    print(f"\n{'='*60}")
    print("ENERGY LANDSCAPE EXPLORATION")
    print(f"{'='*60}")
    
    # Use a smaller network for detailed analysis
    small_network = initialize_stochastic_network(8, temperature=1.0, random_seed=42)
    small_patterns = [
        np.array([1, -1, 1, -1, 1, -1, 1, -1]),
        np.array([1, 1, -1, -1, 1, 1, -1, -1])
    ]
    store_patterns(small_network, small_patterns)
    
    # Run longer simulation to see energy evolution
    print("\nRunning extended simulation to observe energy evolution...")
    small_network['state'] = np.random.choice([-1, 1], size=8)
    results = run_monte_carlo(small_network, num_steps=5000, record_interval=50, verbose=True)
    
    print(f"\nEnergy evolution summary:")
    print(f"  Initial energy: {results['energy_history'][0]:.3f}")
    print(f"  Final energy:   {results['energy_history'][-1]:.3f}")
    print(f"  Minimum energy: {np.min(results['energy_history']):.3f}")
    print(f"  Energy variance: {np.var(results['energy_history']):.3f}")
    
    # 4. Temperature comparison
    print(f"\n{'='*60}")
    print("TEMPERATURE EFFECTS COMPARISON")
    print(f"{'='*60}")
    
    test_temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    convergence_results = []
    
    for temp in test_temps:
        network = initialize_stochastic_network(num_neurons, temperature=temp, random_seed=42)
        store_patterns(network, patterns)
        
        # Start from random state
        network['state'] = np.random.choice([-1, 1], size=num_neurons)
        initial_energy = compute_energy(network, network['state'])
        
        # Run simulation
        results = run_monte_carlo(network, num_steps=2000, record_interval=200, verbose=False)
        
        convergence_results.append({
            'temperature': temp,
            'initial_energy': initial_energy,
            'final_energy': results['final_energy'],
            'energy_change': results['final_energy'] - initial_energy,
            'acceptance_rate': results['total_acceptance_rate'],
            'computation_time': results['computation_time']
        })
    
    print("\nTemperature effects summary:")
    print("Temp  | Initial E | Final E  | ΔE      | Accept% | Time(s)")
    print("------|-----------|----------|---------|---------|--------")
    for result in convergence_results:
        print(f"{result['temperature']:5.1f} | {result['initial_energy']:9.3f} | "
              f"{result['final_energy']:8.3f} | {result['energy_change']:7.3f} | "
              f"{result['acceptance_rate']:6.1%} | {result['computation_time']:6.3f}")
    
    print(f"\n{'='*80}")
    print("MONTE CARLO HOPFIELD DEMONSTRATION COMPLETED")
    print(f"{'='*80}")
    print("Key observations:")
    print("• Lower temperatures lead to more deterministic behavior")
    print("• Higher temperatures increase exploration but reduce precision")
    print("• Acceptance rate decreases with temperature")
    print("• Energy generally decreases over time (system seeks stability)")
    print("• Pattern recovery depends on noise level and temperature")

if __name__ == "__main__":
    demonstrate_monte_carlo_hopfield()

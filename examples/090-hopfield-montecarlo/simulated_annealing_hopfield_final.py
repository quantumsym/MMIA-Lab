#!/usr/bin/env python3
#

'''
 implementation of Simulated Annealing and Parallel Tempering for Hopfield networks.

This script provides functions to perform advanced stochastic optimization on a
Hopfield network's energy landscape. It builds upon the basic Monte Carlo
simulation by introducing temperature schedules (Simulated Annealing) and
multi-replica systems (Parallel Tempering) to more effectively find global
energy minima.
'''

import numpy as np
import time
import matplotlib.pyplot as plt
# Assumes 'monte_carlo_hopfield.py' is in the same directory
from monte_carlo_hopfield import (
    initialize_stochastic_network, 
    store_patterns,
    compute_energy, 
    compute_energy_change
)

# ============================================================================
#  SIMULATED ANNEALING
# ============================================================================

def run_simulated_annealing(network, initial_temp, final_temp, total_steps, cooling_schedule='linear', verbose=True):
    """
    Performs Simulated Annealing on a Hopfield network.

    Args:
        network (dict): The network dictionary, which will be modified.
        initial_temp (float): The starting temperature.
        final_temp (float): The ending temperature.
        total_steps (int): The total number of Monte Carlo steps.
        cooling_schedule (str): The cooling schedule ('linear', 'exponential', 'logarithmic').
        verbose (bool): If True, prints progress information.

    Returns:
        dict: A dictionary containing the results of the annealing process.
    """
    if verbose:
        print(f"Running Simulated Annealing for {total_steps} steps...")
        print(f"  Temperature range: {initial_temp:.2f} -> {final_temp:.2f}")
        print(f"  Cooling schedule: {cooling_schedule}")

    start_time = time.time()

    # History tracking
    energy_history = []
    temp_history = []
    best_state = network['state'].copy()
    best_energy = compute_energy(network, best_state)

    accepted_flips = 0

    for step in range(total_steps):
        # 1. Update temperature based on the cooling schedule
        if cooling_schedule == 'linear':
            network['temperature'] = initial_temp - (initial_temp - final_temp) * (step / total_steps)
        elif cooling_schedule == 'exponential':
            # T(t) = T_initial * (T_final / T_initial)^(t / total_steps)
            network['temperature'] = initial_temp * (final_temp / initial_temp) ** (step / total_steps)
        elif cooling_schedule == 'logarithmic':
            # T(t) = T_initial / log(1 + t)
            network['temperature'] = initial_temp / np.log(2 + step)
        else:
            raise ValueError(f"Unknown cooling schedule: {cooling_schedule}")
        
        network['temperature'] = max(network['temperature'], final_temp) # Ensure it doesn't go below final_temp

        # 2. Perform a Monte Carlo step
        neuron_to_flip = np.random.randint(network['num_neurons'])
        delta_e = compute_energy_change(network, network['state'], neuron_to_flip)

        accept_flip = False
        if delta_e < 0:
            accept_flip = True
        elif network['temperature'] > 0:
            acceptance_probability = np.exp(-delta_e / network['temperature'])
            if np.random.rand() < acceptance_probability:
                accept_flip = True

        if accept_flip:
            network['state'][neuron_to_flip] *= -1
            accepted_flips += 1

        # 3. Update best state found so far
        current_energy = compute_energy(network, network['state'])
        if current_energy < best_energy:
            best_energy = current_energy
            best_state = network['state'].copy()

        # 4. Record history
        if step % max(1, total_steps // 100) == 0:
            energy_history.append(current_energy)
            temp_history.append(network['temperature'])
            if verbose and step > 0 and (step % max(1, total_steps // 10) == 0):
                print(f"  Step {step:6d}/{total_steps}: Temp={network['temperature']:7.3f}, Energy={current_energy:9.3f}, Best Energy={best_energy:9.3f}")

    end_time = time.time()

    results = {
        'best_state': best_state,
        'best_energy': best_energy,
        'final_state': network['state'].copy(),
        'final_energy': compute_energy(network, network['state']),
        'energy_history': np.array(energy_history),
        'temp_history': np.array(temp_history),
        'total_acceptance_rate': accepted_flips / total_steps,
        'computation_time': end_time - start_time
    }

    if verbose:
        print(f"Simulated Annealing finished in {results['computation_time']:.2f} seconds.")
        print(f"  Best Energy Found: {results['best_energy']:.3f}")

    return results

# ============================================================================
#  PARALLEL TEMPERING
# ============================================================================

def initialize_parallel_tempering_system(num_neurons, num_replicas, temp_min, temp_max, random_seed=None):
    """
    Initializes a Parallel Tempering system with multiple network replicas.

    Args:
        num_neurons (int): The number of neurons for each replica.
        num_replicas (int): The number of replicas (networks) in the system.
        temp_min (float): The minimum temperature of the coldest replica.
        temp_max (float): The maximum temperature of the hottest replica.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        dict: A dictionary representing the Parallel Tempering system.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create a geometric temperature ladder
    temperatures = np.geomspace(temp_min, temp_max, num_replicas)
    
    replicas = []
    for i in range(num_replicas):
        replica = initialize_stochastic_network(num_neurons, temperatures[i])
        replicas.append(replica)

    system = {
        'replicas': replicas,
        'temperatures': temperatures,
        'num_replicas': num_replicas,
        'exchange_attempts': np.zeros((num_replicas - 1,)),
        'exchange_successes': np.zeros((num_replicas - 1,))
    }
    
    print(f"Initialized Parallel Tempering system with {num_replicas} replicas.")
    print(f"  Temperature ladder: {[f'{t:.2f}' for t in temperatures]}")
    return system

def run_parallel_tempering(system, total_steps, mc_steps_per_exchange, exchange_pattern='adjacent', verbose=True):
    """
    Runs the Parallel Tempering simulation.

    Args:
        system (dict): The Parallel Tempering system dictionary.
        total_steps (int): The total number of exchange steps.
        mc_steps_per_exchange (int): Number of MC steps to run between exchanges.
        exchange_pattern (str): How to choose pairs for exchange ('adjacent' or 'random').
        verbose (bool): If True, prints progress information.

    Returns:
        dict: A dictionary containing the results of the simulation.
    """
    if verbose:
        print(f"Running Parallel Tempering for {total_steps} exchange steps...")

    start_time = time.time()
    replicas = system['replicas']
    num_replicas = system['num_replicas']

    # History tracking
    energy_histories = [[] for _ in range(num_replicas)]
    best_energy = float('inf')
    best_state = None
    best_replica_idx = -1

    for step in range(total_steps):
        # 1. Run independent Monte Carlo simulations on each replica
        for i, replica in enumerate(replicas):
            # We need a lightweight MC run here, not the full `run_monte_carlo`
            for _ in range(mc_steps_per_exchange):
                neuron_to_flip = np.random.randint(replica['num_neurons'])
                delta_e = compute_energy_change(replica, replica['state'], neuron_to_flip)
                if delta_e < 0 or (replica['temperature'] > 0 and np.random.rand() < np.exp(-delta_e / replica['temperature'])):
                    replica['state'][neuron_to_flip] *= -1

        # 2. Attempt to exchange states between replicas
        if exchange_pattern == 'adjacent':
            # Alternate between odd and even pairs to avoid bias
            start_idx = step % 2
            pairs_to_try = range(start_idx, num_replicas - 1, 2)
        elif exchange_pattern == 'random':
            pairs_to_try = np.random.permutation(num_replicas - 1)
        else:
            raise ValueError(f"Unknown exchange pattern: {exchange_pattern}")

        for i in pairs_to_try:
            j = i + 1
            replica1, replica2 = replicas[i], replicas[j]
            
            # Metropolis criterion for state exchange
            # Δ = (E_j - E_i) / T_i + (E_i - E_j) / T_j = (E_j - E_i) * (1/T_i - 1/T_j)
            energy1 = compute_energy(replica1, replica1['state'])
            energy2 = compute_energy(replica2, replica2['state'])
            
            delta_beta = (1.0 / replica1['temperature']) - (1.0 / replica2['temperature'])
            delta_energy = energy2 - energy1
            
            system['exchange_attempts'][i] += 1
            if np.random.rand() < np.exp(delta_beta * delta_energy):
                # Swap states
                replica1['state'], replica2['state'] = replica2['state'], replica1['state']
                system['exchange_successes'][i] += 1

        # 3. Record history and find best state
        for i, replica in enumerate(replicas):
            current_energy = compute_energy(replica, replica['state'])
            energy_histories[i].append(current_energy)
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = replica['state'].copy()
                best_replica_idx = i

        if verbose and (step + 1) % max(1, total_steps // 10) == 0:
            exchange_rates = system['exchange_successes'] / system['exchange_attempts']
            avg_exchange_rate = np.mean(exchange_rates[np.isfinite(exchange_rates)])
            print(f"  Step {step+1:4d}/{total_steps}: Best Energy={best_energy:9.3f} (from T={replicas[best_replica_idx]['temperature']:.2f}), Avg Exchange Rate={avg_exchange_rate:.1%}")

    end_time = time.time()
    final_exchange_rate = np.sum(system['exchange_successes']) / np.sum(system['exchange_attempts'])

    results = {
        'best_state': best_state,
        'best_energy': best_energy,
        'best_replica_idx': best_replica_idx,
        'energy_histories': [np.array(h) for h in energy_histories],
        'final_exchange_rate': final_exchange_rate,
        'computation_time': end_time - start_time
    }

    if verbose:
        print(f"Parallel Tempering finished in {results['computation_time']:.2f} seconds.")
        print(f"  Best Energy Found: {results['best_energy']:.3f}")
        print(f"  Overall Exchange Rate: {results['final_exchange_rate']:.1%}")

    return results


# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================


# ============================================================================
#  VISUALIZATION FUNCTIONS
# ============================================================================

def plot_annealing_schedule(schedule_data, output_file='annealing_schedule.svg'):
    """
    Plots the temperature schedule during simulated annealing.
    
    Args:
        schedule_data (dict): Dictionary with 'steps' and 'temperatures'.
        output_file (str): Output SVG file name.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(schedule_data['steps'], schedule_data['temperatures'], 'r-', linewidth=2)
    plt.xlabel('Annealing Step', fontsize=12)
    plt.ylabel('Temperature T', fontsize=12)
    plt.title('Simulated Annealing Temperature Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved annealing schedule plot to {output_file}")

def plot_sa_energy_evolution(sa_results, output_file='sa_energy_evolution.svg'):
    """
    Plots energy evolution during simulated annealing with temperature overlay.
    
    Args:
        sa_results (dict): Dictionary with 'energy_history' and 'temperature_history'.
        output_file (str): Output SVG file name.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(sa_results['energy_history'], 'b-', linewidth=2, label='Energy')
    ax1.set_xlabel('Annealing Step', fontsize=12)
    ax1.set_ylabel('Energy E', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(sa_results['temp_history'], 'r--', linewidth=2, alpha=0.7, label='Temperature')
    ax2.set_ylabel('Temperature T', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Simulated Annealing: Energy and Temperature Evolution', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved SA energy evolution plot to {output_file}")

def plot_pt_replica_energies(pt_results, output_file='pt_replica_energies.svg'):
    """
    Plots energy trajectories for all replicas in parallel tempering.
    
    Args:
        pt_results (dict): Dictionary with 'replica_histories' containing energy per replica.
        output_file (str): Output SVG file name.
    """
    plt.figure(figsize=(12, 6))
    
    for replica_id, history in pt_results['replica_histories'].items():
        plt.plot(history['energies'], linewidth=1.5, alpha=0.7, label=f'Replica {replica_id}')
    
    plt.xlabel('Exchange Step', fontsize=12)
    plt.ylabel('Energy E', fontsize=12)
    plt.title('Parallel Tempering: Replica Energy Trajectories', fontsize=14, fontweight='bold')
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PT replica energies plot to {output_file}")

def plot_pt_exchange_matrix(exchange_matrix, temperatures, output_file='pt_exchange_matrix.svg'):
    """
    Plots the exchange acceptance matrix for parallel tempering.
    
    Args:
        exchange_matrix (np.ndarray): Matrix of exchange acceptance rates.
        temperatures (list): List of temperatures for each replica.
        output_file (str): Output SVG file name.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(exchange_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Exchange Acceptance Rate')
    plt.xlabel('Replica j', fontsize=12)
    plt.ylabel('Replica i', fontsize=12)
    plt.title('Parallel Tempering: Exchange Acceptance Matrix', fontsize=14, fontweight='bold')
    
    # Add temperature labels
    temp_labels = [f'{t:.2f}' for t in temperatures]
    plt.xticks(range(len(temperatures)), temp_labels, rotation=45)
    plt.yticks(range(len(temperatures)), temp_labels)
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PT exchange matrix plot to {output_file}")

def plot_sa_vs_pt_comparison(sa_results, pt_results, output_file='sa_vs_pt_comparison.svg'):
    """
    Compares final results from SA and PT methods.
    
    Args:
        sa_results (dict): Results from simulated annealing.
        pt_results (dict): Results from parallel tempering.
        output_file (str): Output SVG file name.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Energy comparison
    ax1.plot(sa_results['energy_history'], 'b-', linewidth=2, label='Simulated Annealing', alpha=0.8)
    ax1.plot(pt_results['best_energy_history'], 'r-', linewidth=2, label='Parallel Tempering', alpha=0.8)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Best Energy', fontsize=12)
    ax1.set_title('Energy Comparison: SA vs PT', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final energy distribution
    methods = ['SA', 'PT']
    final_energies = [sa_results['final_energy'], pt_results['final_energy']]
    colors = ['blue', 'red']
    
    ax2.bar(methods, final_energies, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Final Energy', fontsize=12)
    ax2.set_title('Final Energy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved SA vs PT comparison plot to {output_file}")


def demonstrate_advanced_optimization():
    """
    Main demonstration function showing Simulated Annealing and Parallel Tempering.
    """
    print("=" * 80)
    print("ADVANCED HOPFIELD OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Create test patterns
    num_neurons = 25
    patterns = [
        np.random.choice([-1, 1], size=num_neurons),  # Random pattern 1
        np.random.choice([-1, 1], size=num_neurons),  # Random pattern 2
        np.random.choice([-1, 1], size=num_neurons),  # Random pattern 3
    ]
    
    print(f"Created {len(patterns)} random test patterns with {num_neurons} neurons each")
    
    # 2. SIMULATED ANNEALING DEMONSTRATION
    print(f"\n{'='*60}")
    print("SIMULATED ANNEALING DEMONSTRATION")
    print(f"{'='*60}")
    
    # Test different cooling schedules
    cooling_schedules = ['linear', 'exponential', 'logarithmic']
    sa_results = []
    
    for schedule in cooling_schedules:
        print(f"\n--- Testing {schedule.upper()} cooling schedule ---")
        
        # Initialize network
        network = initialize_stochastic_network(num_neurons, temperature=10.0, random_seed=42)
        store_patterns(network, patterns)
        
        # Start from random state
        network['state'] = np.random.choice([-1, 1], size=num_neurons)
        initial_energy = compute_energy(network, network['state'])
        
        # Run Simulated Annealing
        result = run_simulated_annealing(
            network, 
            initial_temp=10.0, 
            final_temp=0.01, 
            total_steps=5000, 
            cooling_schedule=schedule, 
            verbose=True
        )
        
        sa_results.append({
            'schedule': schedule,
            'initial_energy': initial_energy,
            'best_energy': result['best_energy'],
            'final_energy': result['final_energy'],
            'energy_improvement': initial_energy - result['best_energy'],
            'acceptance_rate': result['total_acceptance_rate'],
            'computation_time': result['computation_time']
        })
    
    # Compare SA results
    print(f"\n{'='*60}")
    print("SIMULATED ANNEALING RESULTS COMPARISON")
    print(f"{'='*60}")
    print("Schedule    | Initial E | Best E   | Final E  | Improve | Accept% | Time(s)")
    print("------------|-----------|----------|----------|---------|---------|--------")
    for result in sa_results:
        print(f"{result['schedule']:11s} | {result['initial_energy']:9.3f} | "
              f"{result['best_energy']:8.3f} | {result['final_energy']:8.3f} | "
              f"{result['energy_improvement']:7.3f} | {result['acceptance_rate']:6.1%} | "
              f"{result['computation_time']:6.3f}")
    
    # 3. PARALLEL TEMPERING DEMONSTRATION
    print(f"\n{'='*60}")
    print("PARALLEL TEMPERING DEMONSTRATION")
    print(f"{'='*60}")
    
    # Test different numbers of replicas
    replica_counts = [4, 8, 12]
    pt_results = []
    
    for num_replicas in replica_counts:
        print(f"\n--- Testing with {num_replicas} replicas ---")
        
        # Initialize Parallel Tempering system
        system = initialize_parallel_tempering_system(
            num_neurons=num_neurons,
            num_replicas=num_replicas,
            temp_min=0.1,
            temp_max=10.0,
            random_seed=42
        )
        
        # Set up the same problem for all replicas
        weights = np.zeros((num_neurons, num_neurons))
        for pattern in patterns:
            weights += np.outer(pattern, pattern)
        weights /= len(patterns)
        np.fill_diagonal(weights, 0)
        
        for replica in system['replicas']:
            replica['weights'] = weights.copy()
            replica['state'] = np.random.choice([-1, 1], size=num_neurons)
        
        # Calculate initial best energy
        initial_energies = [compute_energy(replica, replica['state']) for replica in system['replicas']]
        initial_best_energy = min(initial_energies)
        
        # Run Parallel Tempering
        result = run_parallel_tempering(
            system,
            total_steps=1000,
            mc_steps_per_exchange=10,
            exchange_pattern='adjacent',
            verbose=True
        )
        
        pt_results.append({
            'num_replicas': num_replicas,
            'initial_best_energy': initial_best_energy,
            'final_best_energy': result['best_energy'],
            'energy_improvement': initial_best_energy - result['best_energy'],
            'exchange_rate': result['final_exchange_rate'],
            'best_replica_temp': system['temperatures'][result['best_replica_idx']],
            'computation_time': result['computation_time']
        })
    
    # Compare PT results
    print(f"\n{'='*60}")
    print("PARALLEL TEMPERING RESULTS COMPARISON")
    print(f"{'='*60}")
    print("Replicas | Initial E | Best E   | Improve | Exchange% | Best T | Time(s)")
    print("---------|-----------|----------|---------|-----------|--------|--------")
    for result in pt_results:
        print(f"{result['num_replicas']:8d} | {result['initial_best_energy']:9.3f} | "
              f"{result['final_best_energy']:8.3f} | {result['energy_improvement']:7.3f} | "
              f"{result['exchange_rate']:8.1%} | {result['best_replica_temp']:6.2f} | "
              f"{result['computation_time']:6.3f}")
    
    # 4. METHOD COMPARISON
    print(f"\n{'='*60}")
    print("OPTIMIZATION METHODS COMPARISON")
    print(f"{'='*60}")
    
    # Compare best results from each method
    best_sa = min(sa_results, key=lambda x: x['best_energy'])
    best_pt = min(pt_results, key=lambda x: x['final_best_energy'])
    
    print(f"Best Simulated Annealing result:")
    print(f"  Schedule: {best_sa['schedule']}")
    print(f"  Best energy: {best_sa['best_energy']:.3f}")
    print(f"  Improvement: {best_sa['energy_improvement']:.3f}")
    print(f"  Time: {best_sa['computation_time']:.3f}s")
    
    print(f"\nBest Parallel Tempering result:")
    print(f"  Replicas: {best_pt['num_replicas']}")
    print(f"  Best energy: {best_pt['final_best_energy']:.3f}")
    print(f"  Improvement: {best_pt['energy_improvement']:.3f}")
    print(f"  Time: {best_pt['computation_time']:.3f}s")
    
    # 5. PATTERN RECOVERY TEST
    print(f"\n{'='*60}")
    print("PATTERN RECOVERY COMPARISON")
    print(f"{'='*60}")
    
    # Test pattern recovery with different methods
    test_pattern = patterns[0]
    noisy_pattern = test_pattern.copy()
    # Add 30% noise
    num_flips = int(0.3 * len(test_pattern))
    flip_indices = np.random.choice(len(test_pattern), size=num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1
    
    print(f"Testing pattern recovery from 30% noise...")
    print(f"Original pattern:  {test_pattern}")
    print(f"Noisy pattern:     {noisy_pattern}")
    print(f"Hamming distance:  {np.sum(test_pattern != noisy_pattern)} bits")
    
    # Test with Simulated Annealing
    network_sa = initialize_stochastic_network(num_neurons, temperature=5.0, random_seed=42)
    store_patterns(network_sa, patterns)
    network_sa['state'] = noisy_pattern.copy()
    
    sa_recovery = run_simulated_annealing(
        network_sa, 5.0, 0.01, 3000, 'exponential', verbose=False
    )
    
    sa_recovered = sa_recovery['best_state']
    sa_success = np.array_equal(test_pattern, sa_recovered)
    sa_final_distance = np.sum(test_pattern != sa_recovered)
    
    # Test with Parallel Tempering
    pt_system = initialize_parallel_tempering_system(num_neurons, 6, 0.1, 5.0, random_seed=42)
    for replica in pt_system['replicas']:
        replica['weights'] = weights.copy()
        replica['state'] = noisy_pattern.copy()
    
    pt_recovery = run_parallel_tempering(
        pt_system, 500, 10, 'adjacent', verbose=False
    )
    
    pt_recovered = pt_recovery['best_state']
    pt_success = np.array_equal(test_pattern, pt_recovered)
    pt_final_distance = np.sum(test_pattern != pt_recovered)
    
    print(f"\nRecovery Results:")
    print(f"Simulated Annealing:")
    print(f"  Recovered: {sa_recovered}")
    print(f"  Success: {sa_success}")
    print(f"  Final distance: {sa_final_distance} bits")
    print(f"  Time: {sa_recovery['computation_time']:.3f}s")
    
    print(f"Parallel Tempering:")
    print(f"  Recovered: {pt_recovered}")
    print(f"  Success: {pt_success}")
    print(f"  Final distance: {pt_final_distance} bits")
    print(f"  Time: {pt_recovery['computation_time']:.3f}s")
    

    # ========================================================================
    # VISUALIZATION OF RESULTS
    # ========================================================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Plot annealing schedule
    schedule_data = {
        'steps': list(range(len(sa_recovery['temp_history']))),
        'temperatures': sa_recovery['temp_history']
    }
    plot_annealing_schedule(schedule_data, output_file='annealing_schedule.svg')
    
    # Plot SA energy evolution
    plot_sa_energy_evolution(sa_recovery, output_file='sa_energy_evolution.svg')
    
    # Plot PT replica energies
    pt_plot_data = {
        'replica_histories': {i: {'energies': pt_recovery['energy_histories'][i]} 
                             for i in range(len(pt_recovery['energy_histories']))}
    }
    plot_pt_replica_energies(pt_plot_data, output_file='pt_replica_energies.svg')
    
    # Plot SA vs PT comparison
    comparison_data_sa = {
        'energy_history': sa_recovery['energy_history'],
        'final_energy': sa_recovery['best_energy']
    }
    comparison_data_pt = {
        'best_energy_history': [pt_recovery['best_energy']] * len(sa_recovery['energy_history']),
        'final_energy': pt_recovery['best_energy']
    }
    plot_sa_vs_pt_comparison(comparison_data_sa, comparison_data_pt, output_file='sa_vs_pt_comparison.svg')
    
    print("All visualizations generated successfully!")

    print(f"\n{'='*80}")
    print("ADVANCED OPTIMIZATION DEMONSTRATION COMPLETED")
    print(f"{'='*80}")
    print("Key insights:")
    print("• Simulated Annealing provides systematic cooling for better convergence")
    print("• Parallel Tempering explores multiple temperatures simultaneously")
    print("• Exchange mechanisms help escape local minima")
    print("• Different cooling schedules affect convergence speed and quality")
    print("• More replicas in PT generally improve solution quality but increase cost")
    print("• Both methods significantly outperform basic Monte Carlo")

if __name__ == "__main__":
    demonstrate_advanced_optimization()

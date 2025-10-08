import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import time
from collections import defaultdict

# ============================================================================
# MONTE CARLO AND METROPOLIS ALGORITHMS FOR HOPFIELD NETWORKS
# ============================================================================

class StochasticHopfieldNetwork:
    """
    Stochastic Hopfield Network implementation using Monte Carlo methods
    and Metropolis algorithm for sampling from the Boltzmann distribution.
    
    This class bridges the gap between deterministic Hopfield networks and
    statistical physics by introducing temperature and stochastic dynamics.
    The network can operate in different regimes:
    - T=0: Deterministic Hopfield dynamics (greedy descent)
    - T>0: Stochastic dynamics with thermal fluctuations
    - T→∞: Random walk (maximum entropy)
    
    Mathematical foundation:
    - Energy function: E = -0.5 * s^T W s - θ^T s
    - Boltzmann distribution: P(s) ∝ exp(-E(s)/T)
    - Metropolis acceptance: P(accept) = min(1, exp(-ΔE/T))
    """
    
    def __init__(self, num_neurons, temperature=1.0, random_seed=None):
        """
        Initialize the stochastic Hopfield network
        
        Args:
            num_neurons (int): Number of neurons in the network
            temperature (float): Temperature parameter T controlling stochasticity
                               T=0: deterministic, T>0: stochastic, T→∞: random
            random_seed (int, optional): Seed for reproducible random behavior
        
        The temperature parameter is crucial for the physics analogy:
        - Low T: System prefers low-energy states (exploitation)
        - High T: System explores high-energy states (exploration)
        - T controls the trade-off between exploitation and exploration
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.num_neurons = num_neurons
        self.temperature = temperature
        
        # Initialize weight matrix (symmetric with zero diagonal)
        # This ensures the energy function is well-defined and the dynamics converge
        self.weights = np.zeros((num_neurons, num_neurons))
        
        # Initialize thresholds (external fields in physics terminology)
        self.thresholds = np.zeros(num_neurons)
        
        # Initialize network state randomly
        # In physics: random initial configuration of spins
        self.state = np.random.choice([-1, 1], size=num_neurons)
        
        # Statistics tracking for analysis
        self.energy_history = []
        self.state_history = []
        self.acceptance_rate = 0.0
        self.total_updates = 0
        self.accepted_updates = 0
        
        print(f"Initialized Stochastic Hopfield Network:")
        print(f"  Neurons: {num_neurons}")
        print(f"  Temperature: {temperature}")
        print(f"  Initial energy: {self.compute_energy():.3f}")
    
    def store_patterns(self, patterns):
        """
        Store patterns using the Hebbian learning rule
        
        This creates the "disorder" in the weight matrix that makes the network
        analogous to a spin glass. Multiple patterns create competing interactions
        leading to frustration and complex energy landscapes.
        
        Args:
            patterns (list): List of patterns to store, each pattern is a numpy array
                           of shape (num_neurons,) with values in {-1, +1}
        
        Mathematical details:
        - Hebbian rule: w_ij = (1/N) * Σ_μ ξ^μ_i * ξ^μ_j
        - This creates correlations between neurons that fire together
        - Multiple patterns create interference and frustration
        - The capacity is approximately 0.14 * N for random patterns
        """
        if not patterns:
            raise ValueError("No patterns provided")
        
        # Validate pattern format
        for i, pattern in enumerate(patterns):
            if len(pattern) != self.num_neurons:
                raise ValueError(f"Pattern {i} has wrong size: {len(pattern)} != {self.num_neurons}")
            if not np.all(np.isin(pattern, [-1, 1])):
                raise ValueError(f"Pattern {i} contains invalid values (must be -1 or +1)")
        
        # Reset weights before storing new patterns
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        
        # Apply Hebbian learning rule
        num_patterns = len(patterns)
        for pattern in patterns:
            # Outer product creates correlation matrix for this pattern
            correlation_matrix = np.outer(pattern, pattern)
            self.weights += correlation_matrix
        
        # Normalize by number of patterns and ensure zero diagonal
        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0)
        
        # Calculate theoretical capacity and utilization
        theoretical_capacity = int(0.14 * self.num_neurons)
        utilization = (num_patterns / theoretical_capacity) * 100
        
        print(f"Stored {num_patterns} patterns using Hebbian rule")
        print(f"Theoretical capacity: ~{theoretical_capacity} patterns")
        print(f"Capacity utilization: {utilization:.1f}%")
        
        # Analyze weight matrix properties
        weight_mean = np.mean(self.weights[np.triu_indices(self.num_neurons, k=1)])
        weight_std = np.std(self.weights[np.triu_indices(self.num_neurons, k=1)])
        print(f"Weight statistics: mean={weight_mean:.3f}, std={weight_std:.3f}")
        
        return patterns
    
    def compute_energy(self, state=None):
        """
        Compute the energy of a given state using the Hopfield energy function
        
        This is the core physics connection: the energy function defines the
        "landscape" that the network explores. Lower energy corresponds to
        better pattern matches or better solutions to optimization problems.
        
        Args:
            state (numpy.ndarray, optional): State to compute energy for.
                                           If None, uses current network state.
        
        Returns:
            float: Energy of the state
        
        Mathematical details:
        - Energy function: E = -0.5 * s^T * W * s - θ^T * s
        - First term: interaction energy between neurons
        - Second term: external field energy (bias terms)
        - Factor 0.5 prevents double counting of symmetric interactions
        """
        if state is None:
            state = self.state
        
        # Compute interaction energy: -0.5 * s^T * W * s
        interaction_energy = -0.5 * np.dot(state, np.dot(self.weights, state))
        
        # Compute external field energy: -θ^T * s
        field_energy = -np.dot(self.thresholds, state)
        
        total_energy = interaction_energy + field_energy
        
        return total_energy
    
    def compute_local_field(self, neuron_idx, state=None):
        """
        Compute the local field (effective field) acting on a specific neuron
        
        The local field determines the "preference" for a neuron to be in
        state +1 or -1. It includes contributions from all other neurons
        (weighted by synaptic strengths) plus the external threshold.
        
        Args:
            neuron_idx (int): Index of the neuron
            state (numpy.ndarray, optional): Network state. If None, uses current state.
        
        Returns:
            float: Local field acting on the neuron
        
        Mathematical details:
        - Local field: h_i = Σ_j w_ij * s_j + θ_i
        - This is the "force" trying to flip neuron i
        - Sign of h_i determines preferred state of neuron i
        """
        if state is None:
            state = self.state
        
        # Compute weighted sum of inputs from other neurons
        synaptic_input = np.dot(self.weights[neuron_idx], state)
        
        # Add external threshold (bias)
        local_field = synaptic_input + self.thresholds[neuron_idx]
        
        return local_field
    
    def metropolis_update(self, neuron_idx):
        """
        Perform a single Metropolis update on a specified neuron
        
        This is the core of the Monte Carlo sampling algorithm. It proposes
        a state change (spin flip) and accepts or rejects it based on the
        energy change and temperature, following the Metropolis criterion.
        
        Args:
            neuron_idx (int): Index of neuron to potentially update
        
        Returns:
            tuple: (accepted, energy_change, old_state, new_state)
        
        Algorithm details:
        1. Propose flip: s_i → -s_i
        2. Compute energy change: ΔE = E_new - E_old
        3. Accept if ΔE ≤ 0 (energy decreases)
        4. Accept with probability exp(-ΔE/T) if ΔE > 0 (energy increases)
        5. This ensures detailed balance and convergence to Boltzmann distribution
        """
        # Store original state for potential rollback
        old_state = self.state[neuron_idx]
        
        # Propose state flip
        new_state = -old_state
        
        # Compute energy change efficiently using local field
        # ΔE = -2 * s_i * h_i where h_i is the local field
        local_field = self.compute_local_field(neuron_idx)
        energy_change = -2 * old_state * local_field
        
        # Metropolis acceptance criterion
        if energy_change <= 0:
            # Energy decreases: always accept (greedy descent)
            accept = True
        else:
            # Energy increases: accept with Boltzmann probability
            if self.temperature > 0:
                boltzmann_factor = np.exp(-energy_change / self.temperature)
                accept = np.random.random() < boltzmann_factor
            else:
                # T=0: never accept energy increases (deterministic dynamics)
                accept = False
        
        # Apply update if accepted
        if accept:
            self.state[neuron_idx] = new_state
            self.accepted_updates += 1
        
        self.total_updates += 1
        
        return accept, energy_change, old_state, new_state
    
    def monte_carlo_step(self, update_order='random'):
        """
        Perform one complete Monte Carlo step (sweep through all neurons)
        
        A Monte Carlo step consists of attempting to update each neuron once.
        The order of updates can be random (parallel-like) or sequential.
        
        Args:
            update_order (str): Order of neuron updates
                              'random': random permutation each step
                              'sequential': fixed order 0,1,2,...,N-1
                              'reverse': reverse order N-1,N-2,...,1,0
        
        Returns:
            dict: Statistics from this Monte Carlo step
        
        Implementation notes:
        - Random order simulates parallel updates (more physical)
        - Sequential order is deterministic and easier to debug
        - One MC step ≈ one unit of "time" in the simulation
        """
        step_start_energy = self.compute_energy()
        step_accepted = 0
        step_attempted = self.num_neurons
        
        # Determine update order
        if update_order == 'random':
            neuron_indices = np.random.permutation(self.num_neurons)
        elif update_order == 'sequential':
            neuron_indices = np.arange(self.num_neurons)
        elif update_order == 'reverse':
            neuron_indices = np.arange(self.num_neurons)[::-1]
        else:
            raise ValueError(f"Unknown update order: {update_order}")
        
        # Attempt to update each neuron
        for neuron_idx in neuron_indices:
            accepted, energy_change, old_state, new_state = self.metropolis_update(neuron_idx)
            if accepted:
                step_accepted += 1
        
        step_end_energy = self.compute_energy()
        step_energy_change = step_end_energy - step_start_energy
        
        # Update acceptance rate (running average)
        if self.total_updates > 0:
            self.acceptance_rate = self.accepted_updates / self.total_updates
        
        return {
            'energy_start': step_start_energy,
            'energy_end': step_end_energy,
            'energy_change': step_energy_change,
            'accepted': step_accepted,
            'attempted': step_attempted,
            'acceptance_rate': step_accepted / step_attempted,
            'state_copy': self.state.copy()
        }
    
    def run_monte_carlo(self, num_steps, record_interval=1, update_order='random', verbose=True):
        """
        Run Monte Carlo simulation for specified number of steps
        
        This is the main simulation loop that evolves the network according
        to stochastic dynamics. The simulation generates a trajectory through
        the state space, sampling from the Boltzmann distribution.
        
        Args:
            num_steps (int): Number of Monte Carlo steps to perform
            record_interval (int): Record statistics every N steps
            update_order (str): Order of neuron updates within each step
            verbose (bool): Print progress information
        
        Returns:
            dict: Complete simulation results and statistics
        
        Simulation details:
        - Each step attempts to update all neurons once
        - Statistics are recorded at specified intervals
        - The trajectory converges to the Boltzmann distribution
        - At T=0: converges to local energy minimum
        - At T>0: samples around energy minima with thermal fluctuations
        """
        if verbose:
            print(f"\nRunning Monte Carlo simulation:")
            print(f"  Steps: {num_steps}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Update order: {update_order}")
            print(f"  Recording interval: {record_interval}")
        
        # Initialize result storage
        results = {
            'steps': [],
            'energies': [],
            'states': [],
            'acceptance_rates': [],
            'energy_changes': [],
            'final_state': None,
            'final_energy': None,
            'total_acceptance_rate': None,
            'convergence_info': {}
        }
        
        # Record initial state
        initial_energy = self.compute_energy()
        if verbose:
            print(f"  Initial energy: {initial_energy:.3f}")
        
        # Main simulation loop
        start_time = time.time()
        
        for step in range(num_steps):
            # Perform one Monte Carlo step
            step_stats = self.monte_carlo_step(update_order=update_order)
            
            # Record statistics at specified intervals
            if step % record_interval == 0:
                results['steps'].append(step)
                results['energies'].append(step_stats['energy_end'])
                results['states'].append(step_stats['state_copy'])
                results['acceptance_rates'].append(step_stats['acceptance_rate'])
                results['energy_changes'].append(step_stats['energy_change'])
            
            # Progress reporting
            if verbose and (step + 1) % max(1, num_steps // 10) == 0:
                progress = (step + 1) / num_steps * 100
                current_energy = step_stats['energy_end']
                current_acceptance = step_stats['acceptance_rate']
                print(f"  Step {step + 1:6d}/{num_steps} ({progress:5.1f}%) | "
                      f"Energy: {current_energy:8.3f} | "
                      f"Acceptance: {current_acceptance:5.1%}")
        
        simulation_time = time.time() - start_time
        
        # Finalize results
        results['final_state'] = self.state.copy()
        results['final_energy'] = self.compute_energy()
        results['total_acceptance_rate'] = self.acceptance_rate
        results['simulation_time'] = simulation_time
        
        # Analyze convergence
        if len(results['energies']) > 10:
            # Check if energy has stabilized (simple convergence criterion)
            recent_energies = results['energies'][-10:]
            energy_variance = np.var(recent_energies)
            energy_trend = np.polyfit(range(len(recent_energies)), recent_energies, 1)[0]
            
            results['convergence_info'] = {
                'final_energy_variance': energy_variance,
                'energy_trend': energy_trend,
                'converged': energy_variance < 0.01 and abs(energy_trend) < 0.001
            }
        
        if verbose:
            print(f"\nSimulation completed:")
            print(f"  Final energy: {results['final_energy']:.3f}")
            print(f"  Energy change: {results['final_energy'] - initial_energy:.3f}")
            print(f"  Overall acceptance rate: {results['total_acceptance_rate']:.1%}")
            print(f"  Simulation time: {simulation_time:.2f} seconds")
            
            if results['convergence_info'].get('converged', False):
                print(f"  Status: Converged")
            else:
                print(f"  Status: May need more steps for convergence")
        
        return results
    
    def set_temperature(self, new_temperature):
        """
        Change the temperature of the system
        
        Temperature controls the level of thermal noise in the system:
        - T=0: Deterministic dynamics (no thermal fluctuations)
        - T>0: Stochastic dynamics (thermal fluctuations allow uphill moves)
        - T→∞: Random walk (maximum entropy, no preference for low energy)
        
        Args:
            new_temperature (float): New temperature value
        """
        old_temperature = self.temperature
        self.temperature = new_temperature
        
        print(f"Temperature changed: {old_temperature:.3f} → {new_temperature:.3f}")
        
        # Reset statistics since dynamics have changed
        self.total_updates = 0
        self.accepted_updates = 0
        self.acceptance_rate = 0.0
    
    def get_pattern_overlap(self, pattern):
        """
        Compute overlap between current state and a stored pattern
        
        Overlap measures how similar the current state is to a stored pattern.
        It's a key quantity in the physics of spin glasses and neural networks.
        
        Args:
            pattern (numpy.ndarray): Pattern to compare with current state
        
        Returns:
            float: Overlap value between -1 (anti-correlated) and +1 (identical)
        
        Mathematical details:
        - Overlap: m = (1/N) * Σ_i s_i * ξ_i
        - m = +1: perfect match
        - m = -1: perfect anti-match
        - m = 0: no correlation (random)
        """
        if len(pattern) != self.num_neurons:
            raise ValueError(f"Pattern size {len(pattern)} != network size {self.num_neurons}")
        
        overlap = np.dot(self.state, pattern) / self.num_neurons
        return overlap
    
    def analyze_energy_landscape(self, num_samples=1000, temperature_range=(0.1, 2.0)):
        """
        Analyze the energy landscape by sampling at different temperatures
        
        This function explores how temperature affects the accessible states
        and provides insights into the complexity of the energy landscape.
        
        Args:
            num_samples (int): Number of samples to collect at each temperature
            temperature_range (tuple): (min_temp, max_temp) for analysis
        
        Returns:
            dict: Analysis results including energy distributions and statistics
        """
        print(f"Analyzing energy landscape with {num_samples} samples...")
        
        # Save original temperature and state
        original_temp = self.temperature
        original_state = self.state.copy()
        
        # Temperature points for analysis
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 10)
        
        analysis_results = {
            'temperatures': temperatures,
            'energy_distributions': [],
            'mean_energies': [],
            'energy_variances': [],
            'acceptance_rates': [],
            'unique_states': []
        }
        
        for temp in temperatures:
            print(f"  Sampling at T = {temp:.2f}...")
            
            # Set temperature and reset to random state
            self.set_temperature(temp)
            self.state = np.random.choice([-1, 1], size=self.num_neurons)
            
            # Equilibration phase (let system reach thermal equilibrium)
            equilibration_steps = min(100, num_samples // 10)
            self.run_monte_carlo(equilibration_steps, verbose=False)
            
            # Sampling phase
            sample_results = self.run_monte_carlo(num_samples, record_interval=1, verbose=False)
            
            # Collect statistics
            energies = sample_results['energies']
            analysis_results['energy_distributions'].append(energies)
            analysis_results['mean_energies'].append(np.mean(energies))
            analysis_results['energy_variances'].append(np.var(energies))
            analysis_results['acceptance_rates'].append(sample_results['total_acceptance_rate'])
            
            # Count unique states visited
            unique_states = len(set(tuple(state) for state in sample_results['states']))
            analysis_results['unique_states'].append(unique_states)
        
        # Restore original state
        self.temperature = original_temp
        self.state = original_state
        
        print("Energy landscape analysis completed.")
        return analysis_results

def demonstrate_monte_carlo_hopfield():
    """
    Comprehensive demonstration of Monte Carlo methods in Hopfield networks
    
    This function showcases the key concepts:
    1. Pattern storage and retrieval
    2. Temperature effects on dynamics
    3. Monte Carlo sampling
    4. Energy landscape exploration
    """
    print("=" * 80)
    print("MONTE CARLO HOPFIELD NETWORK DEMONSTRATION")
    print("=" * 80)
    
    # ========================================================================
    # 1. BASIC SETUP AND PATTERN STORAGE
    # ========================================================================
    
    print("\n1. Setting up network and storing patterns...")
    
    # Create network
    network = StochasticHopfieldNetwork(num_neurons=20, temperature=0.5, random_seed=42)
    
    # Create test patterns
    patterns = []
    
    # Pattern 1: Alternating pattern
    pattern1 = np.array([1, -1] * 10)
    patterns.append(pattern1)
    
    # Pattern 2: Block pattern
    pattern2 = np.array([1] * 10 + [-1] * 10)
    patterns.append(pattern2)
    
    # Pattern 3: Random pattern
    np.random.seed(123)
    pattern3 = np.random.choice([-1, 1], size=20)
    patterns.append(pattern3)
    
    # Store patterns
    network.store_patterns(patterns)
    
    print(f"Stored {len(patterns)} patterns:")
    for i, pattern in enumerate(patterns):
        print(f"  Pattern {i+1}: {pattern[:10]}... (showing first 10 elements)")
    
    # ========================================================================
    # 2. TEMPERATURE EFFECTS DEMONSTRATION
    # ========================================================================
    
    print("\n2. Demonstrating temperature effects...")
    
    temperatures = [0.0, 0.1, 0.5, 1.0, 2.0]
    temp_results = {}
    
    for temp in temperatures:
        print(f"\n--- Temperature T = {temp} ---")
        
        # Set temperature and initialize with noisy version of pattern 1
        network.set_temperature(temp)
        
        # Create noisy version of first pattern (flip 30% of bits)
        noisy_pattern = pattern1.copy()
        flip_indices = np.random.choice(20, size=6, replace=False)
        noisy_pattern[flip_indices] *= -1
        network.state = noisy_pattern
        
        initial_overlap = network.get_pattern_overlap(pattern1)
        initial_energy = network.compute_energy()
        
        print(f"Initial overlap with pattern 1: {initial_overlap:.3f}")
        print(f"Initial energy: {initial_energy:.3f}")
        
        # Run simulation
        results = network.run_monte_carlo(num_steps=200, record_interval=5, verbose=False)
        
        final_overlap = network.get_pattern_overlap(pattern1)
        final_energy = results['final_energy']
        
        print(f"Final overlap with pattern 1: {final_overlap:.3f}")
        print(f"Final energy: {final_energy:.3f}")
        print(f"Energy change: {final_energy - initial_energy:.3f}")
        print(f"Acceptance rate: {results['total_acceptance_rate']:.1%}")
        
        temp_results[temp] = {
            'initial_overlap': initial_overlap,
            'final_overlap': final_overlap,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'acceptance_rate': results['total_acceptance_rate'],
            'energy_trajectory': results['energies']
        }
    
    # ========================================================================
    # 3. ENERGY LANDSCAPE ANALYSIS
    # ========================================================================
    
    print("\n3. Analyzing energy landscape...")
    
    # Reset to moderate temperature for landscape analysis
    network.set_temperature(1.0)
    landscape_analysis = network.analyze_energy_landscape(
        num_samples=500, 
        temperature_range=(0.1, 3.0)
    )
    
    # ========================================================================
    # 4. PATTERN RETRIEVAL PERFORMANCE
    # ========================================================================
    
    print("\n4. Testing pattern retrieval performance...")
    
    retrieval_results = {}
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of bits to flip
    
    for noise_level in noise_levels:
        print(f"\nTesting with {noise_level:.1%} noise...")
        
        pattern_results = []
        
        for pattern_idx, pattern in enumerate(patterns):
            # Create noisy version
            num_flips = int(noise_level * len(pattern))
            noisy_pattern = pattern.copy()
            flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1
            
            # Test retrieval at different temperatures
            for temp in [0.1, 0.5, 1.0]:
                network.set_temperature(temp)
                network.state = noisy_pattern
                
                initial_overlap = network.get_pattern_overlap(pattern)
                
                # Run retrieval simulation
                results = network.run_monte_carlo(num_steps=100, verbose=False)
                
                final_overlap = network.get_pattern_overlap(pattern)
                
                pattern_results.append({
                    'pattern_idx': pattern_idx,
                    'temperature': temp,
                    'noise_level': noise_level,
                    'initial_overlap': initial_overlap,
                    'final_overlap': final_overlap,
                    'energy_change': results['final_energy'] - results['energies'][0],
                    'success': final_overlap > 0.8  # Consider successful if overlap > 0.8
                })
        
        retrieval_results[noise_level] = pattern_results
    
    # ========================================================================
    # 5. RESULTS SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n1. Temperature Effects:")
    for temp, results in temp_results.items():
        print(f"  T={temp:4.1f}: Overlap {results['initial_overlap']:+.3f} → {results['final_overlap']:+.3f}, "
              f"Energy {results['initial_energy']:+.3f} → {results['final_energy']:+.3f}, "
              f"Accept: {results['acceptance_rate']:5.1%}")
    
    print("\n2. Energy Landscape:")
    print(f"  Temperature range: {landscape_analysis['temperatures'][0]:.1f} - {landscape_analysis['temperatures'][-1]:.1f}")
    print(f"  Energy variance range: {min(landscape_analysis['energy_variances']):.3f} - {max(landscape_analysis['energy_variances']):.3f}")
    print(f"  Unique states range: {min(landscape_analysis['unique_states'])} - {max(landscape_analysis['unique_states'])}")
    
    print("\n3. Pattern Retrieval Success Rates:")
    for noise_level in noise_levels:
        results = retrieval_results[noise_level]
        success_rate = sum(r['success'] for r in results) / len(results)
        print(f"  {noise_level:.1%} noise: {success_rate:.1%} success rate")
    
    print("\nKey Insights:")
    print("• Low temperature (T≈0): Deterministic, fast convergence, may get stuck in local minima")
    print("• Medium temperature (T≈0.5-1.0): Good balance of exploration and exploitation")
    print("• High temperature (T≥2): High exploration, slow convergence, may not settle")
    print("• Monte Carlo sampling enables escape from local minima through thermal fluctuations")
    print("• Acceptance rate decreases with lower temperature (fewer uphill moves accepted)")
    
    return {
        'network': network,
        'patterns': patterns,
        'temperature_results': temp_results,
        'landscape_analysis': landscape_analysis,
        'retrieval_results': retrieval_results
    }

def visualize_monte_carlo_results(demo_results):
    """
    Create comprehensive visualizations of Monte Carlo simulation results
    
    Args:
        demo_results (dict): Results from demonstrate_monte_carlo_hopfield()
    """
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Extract data
    temp_results = demo_results['temperature_results']
    landscape_analysis = demo_results['landscape_analysis']
    retrieval_results = demo_results['retrieval_results']
    
    # ========================================================================
    # Plot 1: Temperature Effects on Energy Trajectories
    # ========================================================================
    
    ax1 = plt.subplot(4, 4, 1)
    
    for temp, results in temp_results.items():
        energies = results['energy_trajectory']
        steps = range(len(energies))
        ax1.plot(steps, energies, label=f'T={temp}', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Monte Carlo Steps')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution at Different Temperatures', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Temperature vs Final Overlap
    # ========================================================================
    
    ax2 = plt.subplot(4, 4, 2)
    
    temperatures = list(temp_results.keys())
    final_overlaps = [temp_results[t]['final_overlap'] for t in temperatures]
    acceptance_rates = [temp_results[t]['acceptance_rate'] for t in temperatures]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(temperatures, final_overlaps, 'bo-', label='Final Overlap', linewidth=2, markersize=8)
    line2 = ax2_twin.plot(temperatures, acceptance_rates, 'ro-', label='Acceptance Rate', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Final Overlap', color='blue')
    ax2_twin.set_ylabel('Acceptance Rate', color='red')
    ax2.set_title('Temperature Effects on Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    # ========================================================================
    # Plot 3: Energy Landscape Analysis
    # ========================================================================
    
    ax3 = plt.subplot(4, 4, 3)
    
    temps = landscape_analysis['temperatures']
    mean_energies = landscape_analysis['mean_energies']
    energy_variances = landscape_analysis['energy_variances']
    
    ax3.errorbar(temps, mean_energies, yerr=np.sqrt(energy_variances), 
                fmt='o-', capsize=5, linewidth=2, markersize=6)
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Mean Energy ± σ')
    ax3.set_title('Energy Landscape vs Temperature', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 4: Unique States Explored
    # ========================================================================
    
    ax4 = plt.subplot(4, 4, 4)
    
    unique_states = landscape_analysis['unique_states']
    
    ax4.plot(temps, unique_states, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Unique States Visited')
    ax4.set_title('State Space Exploration', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 5-8: Energy Distributions at Different Temperatures
    # ========================================================================
    
    selected_temp_indices = [0, 3, 6, 9]  # Select 4 temperatures
    
    for i, temp_idx in enumerate(selected_temp_indices):
        ax = plt.subplot(4, 4, 5 + i)
        
        temp = temps[temp_idx]
        energies = landscape_analysis['energy_distributions'][temp_idx]
        
        ax.hist(energies, bins=20, alpha=0.7, density=True, color=f'C{i}')
        ax.axvline(np.mean(energies), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Energy Distribution at T={temp:.2f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 9: Pattern Retrieval Success Rate
    # ========================================================================
    
    ax9 = plt.subplot(4, 4, 9)
    
    noise_levels = list(retrieval_results.keys())
    success_rates = []
    
    for noise_level in noise_levels:
        results = retrieval_results[noise_level]
        success_rate = sum(r['success'] for r in results) / len(results)
        success_rates.append(success_rate)
    
    ax9.plot(noise_levels, success_rates, 'mo-', linewidth=3, markersize=10)
    ax9.set_xlabel('Noise Level')
    ax9.set_ylabel('Success Rate')
    ax9.set_title('Pattern Retrieval vs Noise', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(0, 1)
    
    # Add percentage labels
    for x, y in zip(noise_levels, success_rates):
        ax9.annotate(f'{y:.1%}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # ========================================================================
    # Plot 10: Retrieval Performance by Temperature
    # ========================================================================
    
    ax10 = plt.subplot(4, 4, 10)
    
    # Aggregate results by temperature
    temp_performance = defaultdict(list)
    for noise_level, results in retrieval_results.items():
        for result in results:
            temp_performance[result['temperature']].append(result['success'])
    
    temperatures = sorted(temp_performance.keys())
    temp_success_rates = [np.mean(temp_performance[t]) for t in temperatures]
    
    ax10.bar(range(len(temperatures)), temp_success_rates, alpha=0.7, color='skyblue')
    ax10.set_xlabel('Temperature')
    ax10.set_ylabel('Success Rate')
    ax10.set_title('Retrieval Success by Temperature', fontweight='bold')
    ax10.set_xticks(range(len(temperatures)))
    ax10.set_xticklabels([f'{t:.1f}' for t in temperatures])
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, rate in enumerate(temp_success_rates):
        ax10.text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # Plot 11: Acceptance Rate vs Temperature
    # ========================================================================
    
    ax11 = plt.subplot(4, 4, 11)
    
    landscape_temps = landscape_analysis['temperatures']
    landscape_acceptance = landscape_analysis['acceptance_rates']
    
    ax11.plot(landscape_temps, landscape_acceptance, 'co-', linewidth=2, markersize=8)
    ax11.set_xlabel('Temperature')
    ax11.set_ylabel('Acceptance Rate')
    ax11.set_title('Acceptance Rate vs Temperature', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim(0, 1)
    
    # ========================================================================
    # Plot 12: Energy Variance vs Temperature
    # ========================================================================
    
    ax12 = plt.subplot(4, 4, 12)
    
    ax12.semilogy(landscape_temps, landscape_analysis['energy_variances'], 'yo-', linewidth=2, markersize=8)
    ax12.set_xlabel('Temperature')
    ax12.set_ylabel('Energy Variance (log scale)')
    ax12.set_title('Energy Fluctuations vs Temperature', fontweight='bold')
    ax12.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plots 13-16: Pattern Visualization and Analysis
    # ========================================================================
    
    patterns = demo_results['patterns']
    
    for i, pattern in enumerate(patterns[:4]):
        ax = plt.subplot(4, 4, 13 + i)
        
        # Reshape pattern for visualization (if possible)
        if len(pattern) == 20:
            # Reshape to 4x5 for visualization
            pattern_2d = pattern.reshape(4, 5)
        else:
            # Use 1D visualization
            pattern_2d = pattern.reshape(1, -1)
        
        im = ax.imshow(pattern_2d, cmap='RdBu', interpolation='nearest', aspect='auto')
        ax.set_title(f'Stored Pattern {i+1}', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='State')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_hopfield_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'monte_carlo_hopfield_analysis.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the complete Monte Carlo Hopfield network demonstration
    """
    # Run comprehensive demonstration
    results = demonstrate_monte_carlo_hopfield()
    
    # Generate visualizations
    visualize_monte_carlo_results(results)
    
    print("\n" + "=" * 80)
    print("MONTE CARLO HOPFIELD DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("This demonstration showcased:")
    print("• Stochastic Hopfield network dynamics")
    print("• Monte Carlo sampling with Metropolis algorithm")
    print("• Temperature effects on network behavior")
    print("• Energy landscape exploration")
    print("• Pattern retrieval under noise")
    print("• Statistical physics principles in neural networks")
    print("\nThe implementation bridges neural networks and statistical physics,")
    print("providing the foundation for advanced optimization algorithms.")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve, minimize
from scipy.special import erf
import time
from collections import defaultdict

# ============================================================================
# MEAN FIELD THEORY AND STATISTICAL ANALYSIS FOR HOPFIELD NETWORKS
# ============================================================================

class MeanFieldHopfield:
    """
    Mean Field Theory implementation for Hopfield networks
    
    Mean Field Theory (MFT) is a powerful analytical approach from statistical
    physics that approximates the behavior of many-body systems by replacing
    the complex interactions between all particles with an effective field
    that represents the average effect of all other particles.
    
    In the context of Hopfield networks:
    - Instead of tracking binary states s_i ∈ {-1, +1}
    - We track continuous magnetizations m_i = <s_i> ∈ [-1, +1]
    - This transforms a discrete optimization problem into a continuous one
    - Allows analytical treatment and faster computation
    
    Key advantages:
    - Analytical insights into network behavior
    - Fast computation (no Monte Carlo sampling)
    - Deterministic results
    - Clear connection to thermodynamics
    
    Key limitations:
    - Approximation (ignores correlations)
    - May miss some phase transitions
    - Less accurate for small systems
    """
    
    def __init__(self, num_neurons, temperature=1.0):
        """
        Initialize Mean Field Hopfield network
        
        Args:
            num_neurons (int): Number of neurons in the network
            temperature (float): Temperature parameter for thermal fluctuations
        
        The mean field approximation replaces the discrete neuron states
        with continuous magnetizations that represent the time-averaged
        or ensemble-averaged values of the neurons.
        """
        self.num_neurons = num_neurons
        self.temperature = temperature
        
        # Weight matrix (symmetric, zero diagonal)
        self.weights = np.zeros((num_neurons, num_neurons))
        
        # External fields (biases)
        self.external_fields = np.zeros(num_neurons)
        
        # Mean field variables (magnetizations)
        self.magnetizations = np.random.uniform(-0.1, 0.1, num_neurons)
        
        # Stored patterns for analysis
        self.stored_patterns = []
        
        print(f"Initialized Mean Field Hopfield Network:")
        print(f"  Neurons: {num_neurons}")
        print(f"  Temperature: {temperature}")
        print(f"  Initial magnetizations range: [{np.min(self.magnetizations):.3f}, {np.max(self.magnetizations):.3f}]")
    
    def store_patterns(self, patterns):
        """
        Store patterns using Hebbian learning rule
        
        Args:
            patterns (list): List of patterns to store, each pattern is a numpy array
                           of shape (num_neurons,) with values in {-1, +1}
        
        The Hebbian rule creates correlations in the weight matrix that
        correspond to the stored patterns. In mean field theory, these
        weights determine the effective fields acting on each neuron.
        """
        if not patterns:
            raise ValueError("No patterns provided")
        
        # Validate patterns
        for i, pattern in enumerate(patterns):
            if len(pattern) != self.num_neurons:
                raise ValueError(f"Pattern {i} has wrong size")
            if not np.all(np.isin(pattern, [-1, 1])):
                raise ValueError(f"Pattern {i} contains invalid values")
        
        # Store patterns for later analysis
        self.stored_patterns = [p.copy() for p in patterns]
        
        # Reset weights
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        
        # Apply Hebbian learning rule
        num_patterns = len(patterns)
        for pattern in patterns:
            # Create correlation matrix for this pattern
            correlation_matrix = np.outer(pattern, pattern)
            self.weights += correlation_matrix
        
        # Normalize and ensure zero diagonal
        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0)
        
        print(f"Stored {num_patterns} patterns using Hebbian rule")
        print(f"Weight matrix statistics: mean={np.mean(self.weights):.3f}, std={np.std(self.weights):.3f}")
        
        return patterns
    
    def compute_effective_field(self, neuron_idx, magnetizations=None):
        """
        Compute the effective field acting on a specific neuron
        
        Args:
            neuron_idx (int): Index of the neuron
            magnetizations (numpy.ndarray, optional): Magnetization values to use.
                                                     If None, uses current magnetizations.
        
        Returns:
            float: Effective field acting on the neuron
        
        Mathematical details:
        - Effective field: h_i = Σ_j w_ij * m_j + θ_i
        - This is the mean field approximation of the local field
        - m_j represents the average state of neuron j
        - The field determines the preferred magnetization of neuron i
        """
        if magnetizations is None:
            magnetizations = self.magnetizations
        
        # Compute weighted sum of magnetizations from other neurons
        synaptic_field = np.dot(self.weights[neuron_idx], magnetizations)
        
        # Add external field (bias)
        effective_field = synaptic_field + self.external_fields[neuron_idx]
        
        return effective_field
    
    def compute_all_effective_fields(self, magnetizations=None):
        """
        Compute effective fields for all neurons simultaneously
        
        Args:
            magnetizations (numpy.ndarray, optional): Magnetization values to use
        
        Returns:
            numpy.ndarray: Array of effective fields for all neurons
        
        This vectorized computation is more efficient than computing
        fields one by one, especially for large networks.
        """
        if magnetizations is None:
            magnetizations = self.magnetizations
        
        # Vectorized computation: h = W * m + θ
        effective_fields = np.dot(self.weights, magnetizations) + self.external_fields
        
        return effective_fields
    
    def magnetization_from_field(self, effective_field):
        """
        Compute equilibrium magnetization from effective field
        
        Args:
            effective_field (float): Effective field acting on the neuron
        
        Returns:
            float: Equilibrium magnetization
        
        Mathematical details:
        - At thermal equilibrium: m = tanh(h/T)
        - This comes from the Boltzmann distribution
        - tanh function ensures m ∈ [-1, +1]
        - At T→0: m = sign(h) (deterministic)
        - At T→∞: m = 0 (random)
        """
        if self.temperature > 0:
            magnetization = np.tanh(effective_field / self.temperature)
        else:
            # Zero temperature limit: deterministic
            magnetization = np.sign(effective_field)
        
        return magnetization
    
    def update_magnetizations(self, learning_rate=1.0):
        """
        Update all magnetizations according to mean field equations
        
        Args:
            learning_rate (float): Step size for the update (0 < lr ≤ 1)
        
        Returns:
            float: Maximum change in magnetizations (convergence measure)
        
        The mean field equations are:
        m_i^(new) = tanh(h_i / T) where h_i = Σ_j w_ij * m_j^(old) + θ_i
        
        This is a fixed-point iteration that converges to the mean field solution.
        """
        # Compute effective fields for all neurons
        effective_fields = self.compute_all_effective_fields()
        
        # Compute new magnetizations
        new_magnetizations = np.array([self.magnetization_from_field(h) for h in effective_fields])
        
        # Apply update with learning rate
        old_magnetizations = self.magnetizations.copy()
        self.magnetizations = (1 - learning_rate) * old_magnetizations + learning_rate * new_magnetizations
        
        # Compute maximum change (for convergence checking)
        max_change = np.max(np.abs(self.magnetizations - old_magnetizations))
        
        return max_change
    
    def solve_mean_field_equations(self, max_iterations=1000, tolerance=1e-6, 
                                 learning_rate=0.5, verbose=True):
        """
        Solve the mean field equations iteratively
        
        Args:
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            learning_rate (float): Step size for updates
            verbose (bool): Print convergence information
        
        Returns:
            dict: Solution results and convergence information
        
        This function iterates the mean field equations until convergence:
        m_i = tanh((Σ_j w_ij * m_j + θ_i) / T)
        
        The iteration continues until the magnetizations stabilize.
        """
        if verbose:
            print(f"Solving mean field equations:")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Tolerance: {tolerance}")
            print(f"  Learning rate: {learning_rate}")
        
        # Track convergence
        convergence_history = []
        magnetization_history = []
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Store current state
            magnetization_history.append(self.magnetizations.copy())
            
            # Update magnetizations
            max_change = self.update_magnetizations(learning_rate)
            convergence_history.append(max_change)
            
            # Check convergence
            if max_change < tolerance:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                    print(f"  Final max change: {max_change:.2e}")
                break
            
            # Progress reporting
            if verbose and (iteration + 1) % max(1, max_iterations // 10) == 0:
                print(f"  Iteration {iteration + 1:4d}: max change = {max_change:.2e}")
        
        else:
            if verbose:
                print(f"  Did not converge after {max_iterations} iterations")
                print(f"  Final max change: {max_change:.2e}")
        
        solve_time = time.time() - start_time
        
        # Compile results
        results = {
            'converged': max_change < tolerance,
            'iterations': iteration + 1 if max_change < tolerance else max_iterations,
            'final_max_change': max_change,
            'final_magnetizations': self.magnetizations.copy(),
            'convergence_history': convergence_history,
            'magnetization_history': magnetization_history,
            'solve_time': solve_time
        }
        
        if verbose:
            print(f"  Solution time: {solve_time:.3f} seconds")
            print(f"  Final magnetizations range: [{np.min(self.magnetizations):.3f}, {np.max(self.magnetizations):.3f}]")
        
        return results
    
    def compute_free_energy(self, magnetizations=None):
        """
        Compute the mean field free energy
        
        Args:
            magnetizations (numpy.ndarray, optional): Magnetization values to use
        
        Returns:
            float: Free energy of the system
        
        Mathematical details:
        The mean field free energy is:
        F = -0.5 * Σ_ij w_ij * m_i * m_j - Σ_i θ_i * m_i - T * Σ_i S(m_i)
        
        where S(m_i) is the entropy contribution:
        S(m_i) = -0.5 * [(1+m_i)*ln(1+m_i) + (1-m_i)*ln(1-m_i)]
        
        The free energy is minimized at equilibrium.
        """
        if magnetizations is None:
            magnetizations = self.magnetizations
        
        # Interaction energy: -0.5 * Σ_ij w_ij * m_i * m_j
        interaction_energy = -0.5 * np.dot(magnetizations, np.dot(self.weights, magnetizations))
        
        # External field energy: -Σ_i θ_i * m_i
        field_energy = -np.dot(self.external_fields, magnetizations)
        
        # Entropy contribution: -T * Σ_i S(m_i)
        entropy_contribution = 0.0
        if self.temperature > 0:
            for m in magnetizations:
                # Avoid numerical issues with log(0)
                m_clipped = np.clip(m, -0.999, 0.999)
                
                # Entropy of a single spin: S = -0.5 * [(1+m)*ln(1+m) + (1-m)*ln(1-m)]
                if abs(m_clipped) < 0.999:
                    entropy = -0.5 * ((1 + m_clipped) * np.log(1 + m_clipped) + 
                                    (1 - m_clipped) * np.log(1 - m_clipped))
                    entropy_contribution += entropy
        
        # Total free energy
        free_energy = interaction_energy + field_energy - self.temperature * entropy_contribution
        
        return free_energy
    
    def compute_pattern_overlaps(self, magnetizations=None):
        """
        Compute overlaps between current magnetizations and stored patterns
        
        Args:
            magnetizations (numpy.ndarray, optional): Magnetization values to use
        
        Returns:
            list: Overlaps with each stored pattern
        
        Overlap measures how similar the current state is to each stored pattern:
        m^μ = (1/N) * Σ_i ξ^μ_i * m_i
        
        This is a key quantity for understanding pattern retrieval.
        """
        if magnetizations is None:
            magnetizations = self.magnetizations
        
        if not self.stored_patterns:
            return []
        
        overlaps = []
        for pattern in self.stored_patterns:
            overlap = np.dot(pattern, magnetizations) / self.num_neurons
            overlaps.append(overlap)
        
        return overlaps
    
    def analyze_fixed_points(self, num_random_starts=10, verbose=True):
        """
        Analyze multiple fixed points of the mean field equations
        
        Args:
            num_random_starts (int): Number of random initial conditions to try
            verbose (bool): Print analysis results
        
        Returns:
            dict: Analysis of all fixed points found
        
        The mean field equations can have multiple fixed points corresponding
        to different stable states of the network. This function explores
        the landscape of fixed points.
        """
        if verbose:
            print(f"Analyzing fixed points with {num_random_starts} random starts...")
        
        fixed_points = []
        
        for start in range(num_random_starts):
            # Random initial condition
            self.magnetizations = np.random.uniform(-0.5, 0.5, self.num_neurons)
            
            # Solve from this initial condition
            result = self.solve_mean_field_equations(verbose=False)
            
            if result['converged']:
                # Check if this is a new fixed point
                is_new = True
                for existing_fp in fixed_points:
                    if np.allclose(result['final_magnetizations'], existing_fp['magnetizations'], atol=1e-3):
                        existing_fp['count'] += 1
                        is_new = False
                        break
                
                if is_new:
                    # Compute properties of this fixed point
                    free_energy = self.compute_free_energy(result['final_magnetizations'])
                    overlaps = self.compute_pattern_overlaps(result['final_magnetizations'])
                    
                    fixed_point = {
                        'magnetizations': result['final_magnetizations'].copy(),
                        'free_energy': free_energy,
                        'overlaps': overlaps,
                        'count': 1,
                        'iterations': result['iterations']
                    }
                    fixed_points.append(fixed_point)
        
        # Sort by free energy (most stable first)
        fixed_points.sort(key=lambda fp: fp['free_energy'])
        
        if verbose:
            print(f"Found {len(fixed_points)} distinct fixed points:")
            for i, fp in enumerate(fixed_points):
                print(f"  Fixed point {i+1}:")
                print(f"    Free energy: {fp['free_energy']:.3f}")
                print(f"    Found {fp['count']} times")
                if fp['overlaps']:
                    max_overlap_idx = np.argmax(np.abs(fp['overlaps']))
                    print(f"    Max overlap: {fp['overlaps'][max_overlap_idx]:.3f} (pattern {max_overlap_idx+1})")
        
        return {
            'fixed_points': fixed_points,
            'num_starts': num_random_starts,
            'num_found': len(fixed_points)
        }
    
    def temperature_sweep(self, temp_range=(0.1, 5.0), num_temps=20, verbose=True):
        """
        Analyze system behavior across a range of temperatures
        
        Args:
            temp_range (tuple): (min_temp, max_temp) for the sweep
            num_temps (int): Number of temperature points to analyze
            verbose (bool): Print progress information
        
        Returns:
            dict: Temperature-dependent properties
        
        This function reveals phase transitions and critical phenomena
        in the mean field system.
        """
        if verbose:
            print(f"Temperature sweep from {temp_range[0]:.2f} to {temp_range[1]:.2f}...")
        
        temperatures = np.linspace(temp_range[0], temp_range[1], num_temps)
        
        results = {
            'temperatures': temperatures,
            'free_energies': [],
            'magnetization_norms': [],
            'pattern_overlaps': [],
            'convergence_iterations': [],
            'phase_transition_detected': False,
            'critical_temperature': None
        }
        
        # Store original temperature
        original_temp = self.temperature
        
        for temp in temperatures:
            if verbose and len(results['free_energies']) % max(1, num_temps // 5) == 0:
                print(f"  T = {temp:.2f}...")
            
            # Set temperature
            self.temperature = temp
            
            # Start from random initial condition
            self.magnetizations = np.random.uniform(-0.1, 0.1, self.num_neurons)
            
            # Solve mean field equations
            solution = self.solve_mean_field_equations(verbose=False)
            
            # Compute properties
            free_energy = self.compute_free_energy()
            magnetization_norm = np.linalg.norm(self.magnetizations)
            overlaps = self.compute_pattern_overlaps()
            
            # Store results
            results['free_energies'].append(free_energy)
            results['magnetization_norms'].append(magnetization_norm)
            results['pattern_overlaps'].append(overlaps)
            results['convergence_iterations'].append(solution['iterations'])
        
        # Detect phase transitions (simple criterion: large change in magnetization norm)
        if len(results['magnetization_norms']) > 1:
            mag_changes = np.abs(np.diff(results['magnetization_norms']))
            max_change_idx = np.argmax(mag_changes)
            max_change = mag_changes[max_change_idx]
            
            # Threshold for phase transition detection
            if max_change > 0.5:
                results['phase_transition_detected'] = True
                results['critical_temperature'] = temperatures[max_change_idx]
        
        # Restore original temperature
        self.temperature = original_temp
        
        if verbose:
            print(f"Temperature sweep completed.")
            if results['phase_transition_detected']:
                print(f"Phase transition detected at T ≈ {results['critical_temperature']:.2f}")
            else:
                print("No clear phase transition detected.")
        
        return results

def compare_mean_field_vs_monte_carlo():
    """
    Compare Mean Field Theory predictions with Monte Carlo simulations
    
    This function demonstrates the accuracy and limitations of the mean field
    approximation by comparing its predictions with exact Monte Carlo results.
    
    Returns:
        dict: Comparison results
    """
    print("=" * 80)
    print("MEAN FIELD THEORY vs MONTE CARLO COMPARISON")
    print("=" * 80)
    
    # Import Monte Carlo implementation
    from monte_carlo_hopfield import StochasticHopfieldNetwork
    
    # Problem setup
    num_neurons = 20
    temperature = 1.0
    random_seed = 42
    
    # Create test patterns
    np.random.seed(123)
    patterns = []
    for i in range(3):
        pattern = np.random.choice([-1, 1], size=num_neurons)
        patterns.append(pattern)
    
    print(f"Comparison setup:")
    print(f"  Neurons: {num_neurons}")
    print(f"  Temperature: {temperature}")
    print(f"  Patterns: {len(patterns)}")
    
    results = {}
    
    # ========================================================================
    # 1. MEAN FIELD THEORY ANALYSIS
    # ========================================================================
    
    print(f"\n1. Mean Field Theory analysis...")
    
    mf_network = MeanFieldHopfield(num_neurons, temperature)
    mf_network.store_patterns(patterns)
    
    # Solve mean field equations
    mf_start_time = time.time()
    mf_solution = mf_network.solve_mean_field_equations(verbose=False)
    mf_time = time.time() - mf_start_time
    
    # Analyze fixed points
    fp_analysis = mf_network.analyze_fixed_points(num_random_starts=20, verbose=False)
    
    # Temperature sweep
    temp_sweep = mf_network.temperature_sweep(temp_range=(0.1, 3.0), num_temps=15, verbose=False)
    
    results['mean_field'] = {
        'method': 'Mean Field Theory',
        'magnetizations': mf_solution['final_magnetizations'],
        'free_energy': mf_network.compute_free_energy(),
        'pattern_overlaps': mf_network.compute_pattern_overlaps(),
        'fixed_points': fp_analysis,
        'temperature_sweep': temp_sweep,
        'computation_time': mf_time,
        'converged': mf_solution['converged']
    }
    
    print(f"  Converged: {mf_solution['converged']}")
    print(f"  Free energy: {results['mean_field']['free_energy']:.3f}")
    print(f"  Pattern overlaps: {[f'{o:.3f}' for o in results['mean_field']['pattern_overlaps']]}")
    print(f"  Fixed points found: {fp_analysis['num_found']}")
    print(f"  Computation time: {mf_time:.3f} seconds")
    
    # ========================================================================
    # 2. MONTE CARLO SIMULATION
    # ========================================================================
    
    print(f"\n2. Monte Carlo simulation...")
    
    mc_network = StochasticHopfieldNetwork(num_neurons, temperature, random_seed)
    mc_network.store_patterns(patterns)
    
    # Run Monte Carlo simulation
    mc_start_time = time.time()
    mc_results = mc_network.run_monte_carlo(num_steps=5000, record_interval=10, verbose=False)
    mc_time = time.time() - mc_start_time
    
    # Compute time-averaged magnetizations
    mc_states = mc_results['states']
    mc_magnetizations = np.mean(mc_states, axis=0)
    
    # Compute pattern overlaps
    mc_overlaps = []
    for pattern in patterns:
        overlap = np.dot(pattern, mc_magnetizations) / num_neurons
        mc_overlaps.append(overlap)
    
    results['monte_carlo'] = {
        'method': 'Monte Carlo',
        'magnetizations': mc_magnetizations,
        'final_energy': mc_results['final_energy'],
        'pattern_overlaps': mc_overlaps,
        'computation_time': mc_time,
        'acceptance_rate': mc_results['total_acceptance_rate']
    }
    
    print(f"  Final energy: {mc_results['final_energy']:.3f}")
    print(f"  Pattern overlaps: {[f'{o:.3f}' for o in mc_overlaps]}")
    print(f"  Acceptance rate: {mc_results['total_acceptance_rate']:.1%}")
    print(f"  Computation time: {mc_time:.3f} seconds")
    
    # ========================================================================
    # 3. COMPARISON ANALYSIS
    # ========================================================================
    
    print(f"\n3. Comparison analysis...")
    
    # Compare magnetizations
    mag_correlation = np.corrcoef(results['mean_field']['magnetizations'], 
                                results['monte_carlo']['magnetizations'])[0, 1]
    mag_rmse = np.sqrt(np.mean((results['mean_field']['magnetizations'] - 
                              results['monte_carlo']['magnetizations'])**2))
    
    # Compare pattern overlaps
    overlap_differences = []
    for mf_overlap, mc_overlap in zip(results['mean_field']['pattern_overlaps'],
                                    results['monte_carlo']['pattern_overlaps']):
        overlap_differences.append(abs(mf_overlap - mc_overlap))
    
    avg_overlap_difference = np.mean(overlap_differences)
    max_overlap_difference = np.max(overlap_differences)
    
    # Speed comparison
    speedup = mc_time / mf_time
    
    comparison = {
        'magnetization_correlation': mag_correlation,
        'magnetization_rmse': mag_rmse,
        'avg_overlap_difference': avg_overlap_difference,
        'max_overlap_difference': max_overlap_difference,
        'speedup': speedup,
        'mf_accuracy': 1 - avg_overlap_difference  # Simple accuracy measure
    }
    
    results['comparison'] = comparison
    
    print(f"  Magnetization correlation: {mag_correlation:.3f}")
    print(f"  Magnetization RMSE: {mag_rmse:.3f}")
    print(f"  Average overlap difference: {avg_overlap_difference:.3f}")
    print(f"  Maximum overlap difference: {max_overlap_difference:.3f}")
    print(f"  Mean Field speedup: {speedup:.1f}x")
    print(f"  Mean Field accuracy: {comparison['mf_accuracy']:.1%}")
    
    # ========================================================================
    # 4. TEMPERATURE-DEPENDENT COMPARISON
    # ========================================================================
    
    print(f"\n4. Temperature-dependent comparison...")
    
    test_temperatures = [0.5, 1.0, 2.0, 3.0]
    temp_comparison = {}
    
    for temp in test_temperatures:
        print(f"  Testing at T = {temp}...")
        
        # Mean Field at this temperature
        mf_network.temperature = temp
        mf_network.magnetizations = np.random.uniform(-0.1, 0.1, num_neurons)
        mf_sol = mf_network.solve_mean_field_equations(verbose=False)
        mf_overlaps = mf_network.compute_pattern_overlaps()
        
        # Monte Carlo at this temperature
        mc_network.set_temperature(temp)
        mc_network.state = np.random.choice([-1, 1], size=num_neurons)
        mc_res = mc_network.run_monte_carlo(num_steps=2000, verbose=False)
        mc_states = mc_res['states']
        mc_mags = np.mean(mc_states, axis=0)
        mc_ovs = [np.dot(p, mc_mags) / num_neurons for p in patterns]
        
        # Compare
        overlap_diff = np.mean([abs(mf - mc) for mf, mc in zip(mf_overlaps, mc_ovs)])
        
        temp_comparison[temp] = {
            'mf_overlaps': mf_overlaps,
            'mc_overlaps': mc_ovs,
            'overlap_difference': overlap_diff
        }
        
        print(f"    Overlap difference: {overlap_diff:.3f}")
    
    results['temperature_comparison'] = temp_comparison
    
    return results

def visualize_mean_field_analysis(comparison_results):
    """
    Create comprehensive visualizations of mean field analysis
    
    Args:
        comparison_results (dict): Results from compare_mean_field_vs_monte_carlo()
    """
    print("\nGenerating mean field analysis visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Extract data
    mf_results = comparison_results['mean_field']
    mc_results = comparison_results['monte_carlo']
    comparison = comparison_results['comparison']
    temp_comparison = comparison_results['temperature_comparison']
    
    # ========================================================================
    # Plot 1: Magnetization Comparison
    # ========================================================================
    
    ax1 = plt.subplot(4, 4, 1)
    
    mf_mags = mf_results['magnetizations']
    mc_mags = mc_results['magnetizations']
    
    ax1.scatter(mf_mags, mc_mags, alpha=0.7, s=50)
    
    # Perfect correlation line
    min_mag = min(np.min(mf_mags), np.min(mc_mags))
    max_mag = max(np.max(mf_mags), np.max(mc_mags))
    ax1.plot([min_mag, max_mag], [min_mag, max_mag], 'r--', alpha=0.7, label='Perfect correlation')
    
    ax1.set_xlabel('Mean Field Magnetizations')
    ax1.set_ylabel('Monte Carlo Magnetizations')
    ax1.set_title(f'Magnetization Comparison (r={comparison["magnetization_correlation"]:.3f})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Pattern Overlap Comparison
    # ========================================================================
    
    ax2 = plt.subplot(4, 4, 2)
    
    mf_overlaps = mf_results['pattern_overlaps']
    mc_overlaps = mc_results['pattern_overlaps']
    pattern_indices = range(len(mf_overlaps))
    
    x_pos = np.arange(len(pattern_indices))
    width = 0.35
    
    ax2.bar(x_pos - width/2, mf_overlaps, width, label='Mean Field', alpha=0.7)
    ax2.bar(x_pos + width/2, mc_overlaps, width, label='Monte Carlo', alpha=0.7)
    
    ax2.set_xlabel('Pattern Index')
    ax2.set_ylabel('Overlap')
    ax2.set_title('Pattern Overlap Comparison', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'P{i+1}' for i in pattern_indices])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotations
    for i, (mf_ov, mc_ov) in enumerate(zip(mf_overlaps, mc_overlaps)):
        diff = abs(mf_ov - mc_ov)
        ax2.text(i, max(mf_ov, mc_ov) + 0.05, f'Δ={diff:.3f}', ha='center', va='bottom', fontsize=8)
    
    # ========================================================================
    # Plot 3: Fixed Points Analysis
    # ========================================================================
    
    ax3 = plt.subplot(4, 4, 3)
    
    fp_analysis = mf_results['fixed_points']
    fixed_points = fp_analysis['fixed_points']
    
    if fixed_points:
        free_energies = [fp['free_energy'] for fp in fixed_points]
        counts = [fp['count'] for fp in fixed_points]
        
        bars = ax3.bar(range(len(fixed_points)), free_energies, alpha=0.7)
        
        # Color bars by stability (count)
        max_count = max(counts) if counts else 1
        for i, (bar, count) in enumerate(zip(bars, counts)):
            color_intensity = count / max_count
            bar.set_color(plt.cm.viridis(color_intensity))
        
        ax3.set_xlabel('Fixed Point Index')
        ax3.set_ylabel('Free Energy')
        ax3.set_title(f'Fixed Points Analysis ({len(fixed_points)} found)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax3.text(i, free_energies[i] + 0.1, f'{count}x', ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # Plot 4: Temperature Sweep
    # ========================================================================
    
    ax4 = plt.subplot(4, 4, 4)
    
    temp_sweep = mf_results['temperature_sweep']
    temperatures = temp_sweep['temperatures']
    mag_norms = temp_sweep['magnetization_norms']
    
    ax4.plot(temperatures, mag_norms, 'b-', linewidth=2, marker='o', markersize=4)
    
    # Highlight critical temperature if detected
    if temp_sweep['phase_transition_detected']:
        critical_temp = temp_sweep['critical_temperature']
        ax4.axvline(critical_temp, color='r', linestyle='--', alpha=0.7, 
                   label=f'Critical T ≈ {critical_temp:.2f}')
        ax4.legend()
    
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Magnetization Norm')
    ax4.set_title('Temperature Sweep', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 5: Free Energy Landscape
    # ========================================================================
    
    ax5 = plt.subplot(4, 4, 5)
    
    if fixed_points:
        # Create a simple 1D representation of the free energy landscape
        fp_energies = [fp['free_energy'] for fp in fixed_points]
        fp_positions = range(len(fixed_points))
        
        # Interpolate between fixed points for visualization
        x_interp = np.linspace(0, len(fixed_points)-1, 100)
        y_interp = np.interp(x_interp, fp_positions, fp_energies)
        
        ax5.plot(x_interp, y_interp, 'b-', linewidth=2, alpha=0.7, label='Energy landscape')
        ax5.scatter(fp_positions, fp_energies, c='red', s=100, zorder=5, label='Fixed points')
        
        ax5.set_xlabel('Configuration Space')
        ax5.set_ylabel('Free Energy')
        ax5.set_title('Free Energy Landscape', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 6: Convergence History
    # ========================================================================
    
    ax6 = plt.subplot(4, 4, 6)
    
    # Use the first fixed point's convergence history
    if fixed_points and 'convergence_history' in mf_results:
        # Get convergence from the main solution
        convergence = mf_results.get('convergence_history', [])
        if convergence:
            iterations = range(len(convergence))
            ax6.semilogy(iterations, convergence, 'g-', linewidth=2)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Max Change (log scale)')
            ax6.set_title('Mean Field Convergence', fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 7: Temperature-Dependent Accuracy
    # ========================================================================
    
    ax7 = plt.subplot(4, 4, 7)
    
    test_temps = list(temp_comparison.keys())
    overlap_diffs = [temp_comparison[t]['overlap_difference'] for t in test_temps]
    
    ax7.plot(test_temps, overlap_diffs, 'mo-', linewidth=2, markersize=8)
    ax7.set_xlabel('Temperature')
    ax7.set_ylabel('Average Overlap Difference')
    ax7.set_title('Mean Field Accuracy vs Temperature', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Add accuracy percentage labels
    for temp, diff in zip(test_temps, overlap_diffs):
        accuracy = (1 - diff) * 100
        ax7.annotate(f'{accuracy:.1f}%', (temp, diff), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # ========================================================================
    # Plot 8: Computation Time Comparison
    # ========================================================================
    
    ax8 = plt.subplot(4, 4, 8)
    
    methods = ['Mean Field', 'Monte Carlo']
    times = [mf_results['computation_time'], mc_results['computation_time']]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax8.bar(methods, times, color=colors, alpha=0.7)
    ax8.set_ylabel('Computation Time (seconds)')
    ax8.set_title(f'Speed Comparison (MF {comparison["speedup"]:.1f}x faster)', fontweight='bold')
    
    # Add time labels
    for bar, time in zip(bars, times):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # Plot 9: Pattern Overlap Evolution (Temperature Sweep)
    # ========================================================================
    
    ax9 = plt.subplot(4, 4, 9)
    
    if 'pattern_overlaps' in temp_sweep:
        pattern_overlaps_temp = temp_sweep['pattern_overlaps']
        
        # Plot overlap for each pattern vs temperature
        for pattern_idx in range(len(pattern_overlaps_temp[0]) if pattern_overlaps_temp else 0):
            overlaps_this_pattern = [overlaps[pattern_idx] for overlaps in pattern_overlaps_temp]
            ax9.plot(temperatures, overlaps_this_pattern, 'o-', 
                    label=f'Pattern {pattern_idx+1}', linewidth=2, markersize=4)
        
        ax9.set_xlabel('Temperature')
        ax9.set_ylabel('Pattern Overlap')
        ax9.set_title('Pattern Overlaps vs Temperature', fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 10: Error Analysis
    # ========================================================================
    
    ax10 = plt.subplot(4, 4, 10)
    
    # Error metrics
    error_types = ['Magnetization\nRMSE', 'Avg Overlap\nDifference', 'Max Overlap\nDifference']
    error_values = [comparison['magnetization_rmse'], 
                   comparison['avg_overlap_difference'],
                   comparison['max_overlap_difference']]
    
    bars = ax10.bar(error_types, error_values, alpha=0.7, color=['red', 'orange', 'yellow'])
    ax10.set_ylabel('Error Magnitude')
    ax10.set_title('Mean Field Approximation Errors', fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars, error_values):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # ========================================================================
    # Plot 11: Magnetization Distribution
    # ========================================================================
    
    ax11 = plt.subplot(4, 4, 11)
    
    ax11.hist(mf_mags, bins=15, alpha=0.7, label='Mean Field', density=True, color='blue')
    ax11.hist(mc_mags, bins=15, alpha=0.7, label='Monte Carlo', density=True, color='red')
    ax11.set_xlabel('Magnetization Value')
    ax11.set_ylabel('Density')
    ax11.set_title('Magnetization Distributions', fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 12: Performance Summary
    # ========================================================================
    
    ax12 = plt.subplot(4, 4, 12)
    
    # Performance metrics
    metrics = ['Accuracy', 'Speed', 'Convergence']
    mf_scores = [comparison['mf_accuracy'], 1.0, 1.0 if mf_results['converged'] else 0.5]
    mc_scores = [1.0, 1.0/comparison['speedup'], 1.0]  # MC as baseline
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax12.bar(x_pos - width/2, mf_scores, width, label='Mean Field', alpha=0.7, color='blue')
    ax12.bar(x_pos + width/2, mc_scores, width, label='Monte Carlo', alpha=0.7, color='red')
    
    ax12.set_xlabel('Performance Metric')
    ax12.set_ylabel('Normalized Score')
    ax12.set_title('Performance Summary', fontweight='bold')
    ax12.set_xticks(x_pos)
    ax12.set_xticklabels(metrics)
    ax12.legend()
    ax12.grid(True, alpha=0.3, axis='y')
    ax12.set_ylim(0, 1.1)
    
    # Add score labels
    for i, (mf_score, mc_score) in enumerate(zip(mf_scores, mc_scores)):
        ax12.text(i - width/2, mf_score + 0.02, f'{mf_score:.2f}', ha='center', va='bottom', fontsize=8)
        ax12.text(i + width/2, mc_score + 0.02, f'{mc_score:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('mean_field_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Mean field analysis visualizations saved as 'mean_field_analysis.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the complete mean field theory demonstration
    """
    # Run comprehensive comparison
    comparison_results = compare_mean_field_vs_monte_carlo()
    
    # Generate visualizations
    visualize_mean_field_analysis(comparison_results)
    
    print("\n" + "=" * 80)
    print("MEAN FIELD THEORY DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("This demonstration showcased:")
    print("• Mean Field Theory for Hopfield networks")
    print("• Analytical solution of equilibrium equations")
    print("• Fixed point analysis and phase transitions")
    print("• Comparison with Monte Carlo simulations")
    print("• Temperature-dependent behavior analysis")
    print("\nKey insights:")
    print("• Mean Field Theory provides fast analytical approximations")
    print("• Accuracy depends on system size and temperature")
    print("• Multiple fixed points reveal complex energy landscapes")
    print("• Significant computational speedup over Monte Carlo")
    print("• Foundation for understanding phase transitions in neural networks")


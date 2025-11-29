#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# BASIC HOPFIELD NETWORK IMPLEMENTATION
# ============================================================================

class HopfieldNetwork:
    """
    Basic implementation of a Hopfield Neural Network
    
    A Hopfield network is a form of recurrent artificial neural network that
    serves as a content-addressable memory system with binary threshold nodes.
    It is guaranteed to converge to a local minimum, but convergence to a
    false pattern (spurious state) rather than the stored pattern (attractor)
    can occur.
    
    Key characteristics:
    - Symmetric weight matrix (w_ij = w_ji)
    - No self-connections (w_ii = 0)
    - Binary neurons with states {-1, +1}
    - Asynchronous update rule
    - Energy function that decreases with each update
    
    Mathematical foundation:
    The network state evolves according to:
    s_i(t+1) = sign(Σ w_ij * s_j(t) + θ_i)
    
    Where:
    - s_i is the state of neuron i
    - w_ij is the weight between neurons i and j
    - θ_i is the threshold (bias) of neuron i
    """
    
    def __init__(self, num_neurons):
        """
        Initialize a Hopfield network with specified number of neurons
        
        Args:
            num_neurons (int): Number of neurons in the network.
                              This determines the size of patterns that can be stored.
                              For example, for 5x5 pixel images, use 25 neurons.
        
        Attributes:
            num_neurons (int): Number of neurons in the network
            weights (numpy.ndarray): Symmetric weight matrix of shape (num_neurons, num_neurons)
                                   Initially all zeros, will be set during training
            thresholds (numpy.ndarray): Bias/threshold values for each neuron
                                      Initially all zeros for simplicity
        
        Note:
            The weight matrix is initialized to zeros and will be computed
            using the Hebbian learning rule when patterns are stored.
        """
        self.num_neurons = num_neurons
        
        # Initialize weight matrix as zeros
        # Shape: (num_neurons, num_neurons)
        # This will be filled when patterns are stored using Hebbian learning
        self.weights = np.zeros((num_neurons, num_neurons))
        
        # Initialize thresholds (biases) as zeros
        # Shape: (num_neurons,)
        # Thresholds can be used to bias certain neurons, but we start with zeros
        self.thresholds = np.zeros(num_neurons)
        
        print(f"Initialized Hopfield network with {num_neurons} neurons")
        print(f"Weight matrix shape: {self.weights.shape}")
        print(f"Thresholds shape: {self.thresholds.shape}")
    
    def store_pattern(self, pattern):
        """
        Store a single pattern in the network using Hebbian learning rule
        
        The Hebbian learning rule states: "Neurons that fire together, wire together"
        Mathematically: Δw_ij = η * s_i * s_j
        
        For Hopfield networks, we use the simplified version:
        w_ij = (1/N) * Σ(over all patterns p) s_i^p * s_j^p
        
        For a single pattern, this becomes:
        w_ij = s_i * s_j (if i ≠ j), w_ii = 0
        
        Args:
            pattern (numpy.ndarray): Binary pattern to store, shape (num_neurons,)
                                   Values should be in {-1, +1} or {0, 1}
                                   If {0, 1}, they will be converted to {-1, +1}
        
        Note:
            This method overwrites any previously stored patterns.
            To store multiple patterns, use store_patterns() method.
        """
        # Convert pattern to numpy array and ensure it's the right shape
        pattern = np.array(pattern).flatten()
        
        # Validate pattern length
        if len(pattern) != self.num_neurons:
            raise ValueError(f"Pattern length ({len(pattern)}) must match "
                           f"number of neurons ({self.num_neurons})")
        
        # Convert binary {0, 1} to bipolar {-1, +1} if necessary
        # Hopfield networks work better with bipolar representation
        if np.all(np.isin(pattern, [0, 1])):
            pattern = 2 * pattern - 1  # Convert {0, 1} to {-1, +1}
            print("Converted pattern from {0, 1} to {-1, +1} representation")
        
        # Validate that pattern is now bipolar
        if not np.all(np.isin(pattern, [-1, 1])):
            raise ValueError("Pattern values must be in {-1, +1} or {0, 1}")
        
        print(f"Storing pattern: {pattern}")
        
        # Apply Hebbian learning rule: w_ij = s_i * s_j
        # This creates an outer product of the pattern with itself
        self.weights = np.outer(pattern, pattern)
        
        # Set diagonal elements to zero (no self-connections)
        # This is crucial for network stability
        np.fill_diagonal(self.weights, 0)
        
        print("Pattern stored successfully using Hebbian learning rule")
        print(f"Weight matrix:\n{self.weights}")
    
    def store_patterns(self, patterns):
        """
        Store multiple patterns in the network using generalized Hebbian rule
        
        For multiple patterns, the weight matrix is computed as:
        w_ij = (1/N) * Σ(over all patterns p) s_i^p * s_j^p
        
        Where N is the number of patterns.
        
        Args:
            patterns (list or numpy.ndarray): List of patterns to store
                                            Each pattern should be shape (num_neurons,)
                                            Values should be in {-1, +1} or {0, 1}
        
        Note:
            The capacity of a Hopfield network is approximately 0.15 * N neurons
            where N is the number of neurons. Beyond this, spurious states appear.
        """
        patterns = np.array(patterns)
        
        # Validate patterns shape
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)  # Single pattern case
        
        num_patterns, pattern_length = patterns.shape
        
        if pattern_length != self.num_neurons:
            raise ValueError(f"Each pattern length ({pattern_length}) must match "
                           f"number of neurons ({self.num_neurons})")
        
        print(f"Storing {num_patterns} patterns...")
        
        # Convert all patterns to bipolar representation if necessary
        for i in range(num_patterns):
            if np.all(np.isin(patterns[i], [0, 1])):
                patterns[i] = 2 * patterns[i] - 1
        
        # Validate all patterns are bipolar
        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("All pattern values must be in {-1, +1} or {0, 1}")
        
        # Initialize weight matrix
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        
        # Apply generalized Hebbian learning rule
        for pattern in patterns:
            # Add contribution of this pattern to weight matrix
            self.weights += np.outer(pattern, pattern)
        
        # Normalize by number of patterns
        self.weights = self.weights / num_patterns
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        print(f"Successfully stored {num_patterns} patterns")
        print(f"Network capacity utilization: {num_patterns}/{int(0.15 * self.num_neurons):.0f} "
              f"({100 * num_patterns / (0.15 * self.num_neurons):.1f}%)")
    
    def update_neuron(self, state, neuron_index):
        """
        Update a single neuron's state based on current network state
        
        The update rule for neuron i is:
        s_i(t+1) = sign(Σ w_ij * s_j(t) + θ_i)
        
        Where:
        - s_i is the state of neuron i
        - w_ij is the weight from neuron j to neuron i
        - θ_i is the threshold (bias) of neuron i
        - sign() is the signum function: sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0
        
        Args:
            state (numpy.ndarray): Current state of all neurons, shape (num_neurons,)
            neuron_index (int): Index of the neuron to update (0 to num_neurons-1)
        
        Returns:
            int: New state of the specified neuron (-1 or +1)
        
        Note:
            This implements the asynchronous update rule where only one neuron
            is updated at a time. This guarantees convergence to a local minimum.
        """
        # Compute net input to the neuron
        # This is the weighted sum of all other neurons' states plus the threshold
        net_input = np.dot(self.weights[neuron_index], state) + self.thresholds[neuron_index]
        
        # Apply sign activation function
        # If net_input > 0, neuron fires (+1)
        # If net_input < 0, neuron doesn't fire (-1)
        # If net_input = 0, we keep the current state (no change)
        if net_input > 0:
            new_state = 1
        elif net_input < 0:
            new_state = -1
        else:
            # If net input is exactly zero, keep current state
            new_state = state[neuron_index]
        
        return new_state
    
    def recall_pattern(self, input_pattern, max_iterations=100, verbose=False):
        """
        Recall a stored pattern from a potentially noisy or incomplete input
        
        This method implements the asynchronous update rule where neurons are
        updated one at a time in random order until the network converges to
        a stable state (attractor) or maximum iterations are reached.
        
        Args:
            input_pattern (numpy.ndarray): Initial pattern to start recall from
                                         Shape (num_neurons,), values in {-1, +1} or {0, 1}
            max_iterations (int, optional): Maximum number of update iterations. Default 100.
            verbose (bool, optional): If True, print detailed iteration information. Default False.
        
        Returns:
            tuple: (final_state, converged, num_iterations)
                final_state (numpy.ndarray): Final network state after recall
                converged (bool): True if network converged to stable state
                num_iterations (int): Number of iterations performed
        
        Note:
            Convergence is guaranteed for Hopfield networks, but it might converge
            to a spurious state rather than the desired stored pattern.
        """
        # Convert input to numpy array and ensure correct format
        current_state = np.array(input_pattern).flatten()
        
        # Validate input length
        if len(current_state) != self.num_neurons:
            raise ValueError(f"Input pattern length ({len(current_state)}) must match "
                           f"number of neurons ({self.num_neurons})")
        
        # Convert to bipolar representation if necessary
        if np.all(np.isin(current_state, [0, 1])):
            current_state = 2 * current_state - 1
        
        if verbose:
            print(f"Starting recall with input: {current_state}")
        
        # Track convergence
        converged = False
        iteration = 0
        
        # Store previous state to check for convergence
        previous_state = current_state.copy()
        
        # Main recall loop
        for iteration in range(max_iterations):
            # Create random order for neuron updates
            # This ensures fair treatment of all neurons
            neuron_order = np.random.permutation(self.num_neurons)
            
            # Update each neuron once in random order
            for neuron_idx in neuron_order:
                # Update the selected neuron
                new_neuron_state = self.update_neuron(current_state, neuron_idx)
                current_state[neuron_idx] = new_neuron_state
            
            if verbose:
                print(f"Iteration {iteration + 1}: {current_state}")
            
            # Check for convergence (no change in state)
            if np.array_equal(current_state, previous_state):
                converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update previous state for next iteration
            previous_state = current_state.copy()
        
        if not converged and verbose:
            print(f"Did not converge after {max_iterations} iterations")
        
        return current_state, converged, iteration + 1
    
    def compute_energy(self, state):
        """
        Compute the energy of a given network state
        
        The energy function for a Hopfield network is defined as:
        E = -0.5 * Σ Σ w_ij * s_i * s_j - Σ θ_i * s_i
        
        Where:
        - w_ij is the weight between neurons i and j
        - s_i, s_j are the states of neurons i and j
        - θ_i is the threshold (bias) of neuron i
        
        The energy function has important properties:
        1. It decreases (or stays constant) with each neuron update
        2. Local minima correspond to stored patterns (attractors)
        3. The network converges when energy reaches a local minimum
        
        Args:
            state (numpy.ndarray): Network state, shape (num_neurons,)
                                 Values should be in {-1, +1}
        
        Returns:
            float: Energy value of the given state
        
        Note:
            Lower energy values indicate more stable states.
            Stored patterns should have lower energy than random states.
        """
        state = np.array(state)
        
        # Validate state
        if len(state) != self.num_neurons:
            raise ValueError(f"State length ({len(state)}) must match "
                           f"number of neurons ({self.num_neurons})")
        
        # Convert to bipolar if necessary
        if np.all(np.isin(state, [0, 1])):
            state = 2 * state - 1
        
        # Compute the interaction energy term: -0.5 * Σ Σ w_ij * s_i * s_j
        # We use matrix multiplication for efficiency: -0.5 * s^T * W * s
        interaction_energy = -0.5 * np.dot(state, np.dot(self.weights, state))
        
        # Compute the threshold energy term: -Σ θ_i * s_i
        threshold_energy = -np.dot(self.thresholds, state)
        
        # Total energy
        total_energy = interaction_energy + threshold_energy
        
        return total_energy
    
    def get_network_info(self):
        """
        Get comprehensive information about the current network state
        
        Returns:
            dict: Dictionary containing network information including:
                - num_neurons: Number of neurons
                - weight_matrix_shape: Shape of weight matrix
                - weight_matrix_symmetric: Whether weights are symmetric
                - weight_matrix_diagonal_zero: Whether diagonal is zero
                - max_weight: Maximum weight value
                - min_weight: Minimum weight value
                - mean_weight: Mean weight value
        """
        info = {
            'num_neurons': self.num_neurons,
            'weight_matrix_shape': self.weights.shape,
            'weight_matrix_symmetric': np.allclose(self.weights, self.weights.T),
            'weight_matrix_diagonal_zero': np.allclose(np.diag(self.weights), 0),
            'max_weight': np.max(self.weights),
            'min_weight': np.min(self.weights),
            'mean_weight': np.mean(self.weights),
            'weight_matrix_norm': np.linalg.norm(self.weights)
        }
        return info

# ============================================================================
# DEMONSTRATION AND TESTING FUNCTIONS
# ============================================================================

def demonstrate_basic_hopfield():
    """
    Demonstrate basic Hopfield network functionality with simple examples
    
    This function shows:
    1. Creating a Hopfield network
    2. Storing a simple pattern
    3. Perfect recall (no noise)
    4. Recall with noise
    5. Energy computation
    6. Network analysis
    """
    print("=" * 60)
    print("BASIC HOPFIELD NETWORK DEMONSTRATION")
    print("=" * 60)
    
    # ========================================================================
    # 1. CREATE NETWORK AND STORE PATTERN
    # ========================================================================
    
    print("\n1. Creating Hopfield network and storing pattern...")
    
    # Create a small network for demonstration
    network = HopfieldNetwork(num_neurons=5)
    
    # Define a simple pattern to store
    # Using bipolar representation {-1, +1}
    original_pattern = np.array([1, -1, 1, -1, 1])
    
    # Store the pattern
    network.store_pattern(original_pattern)
    
    # ========================================================================
    # 2. PERFECT RECALL TEST
    # ========================================================================
    
    print("\n2. Testing perfect recall...")
    
    # Test recall with the exact stored pattern
    recalled_pattern, converged, iterations = network.recall_pattern(
        original_pattern, verbose=True)
    
    print(f"Original pattern:  {original_pattern}")
    print(f"Recalled pattern:  {recalled_pattern}")
    print(f"Perfect recall:    {np.array_equal(original_pattern, recalled_pattern)}")
    print(f"Converged:         {converged}")
    print(f"Iterations:        {iterations}")
    
    # ========================================================================
    # 3. NOISY RECALL TEST
    # ========================================================================
    
    print("\n3. Testing recall with noisy input...")
    
    # Create a noisy version of the pattern (flip one bit)
    noisy_pattern = original_pattern.copy()
    noisy_pattern[2] = -noisy_pattern[2]  # Flip the middle bit
    
    print(f"Original pattern:  {original_pattern}")
    print(f"Noisy input:       {noisy_pattern}")
    
    # Attempt to recall from noisy input
    recalled_noisy, converged_noisy, iterations_noisy = network.recall_pattern(
        noisy_pattern, verbose=True)
    
    print(f"Recalled pattern:  {recalled_noisy}")
    print(f"Successful recall: {np.array_equal(original_pattern, recalled_noisy)}")
    print(f"Converged:         {converged_noisy}")
    print(f"Iterations:        {iterations_noisy}")
    
    # ========================================================================
    # 4. ENERGY ANALYSIS
    # ========================================================================
    
    print("\n4. Energy analysis...")
    
    # Compute energies for different states
    energy_original = network.compute_energy(original_pattern)
    energy_noisy = network.compute_energy(noisy_pattern)
    energy_recalled = network.compute_energy(recalled_noisy)
    
    print(f"Energy of original pattern: {energy_original:.4f}")
    print(f"Energy of noisy pattern:    {energy_noisy:.4f}")
    print(f"Energy of recalled pattern: {energy_recalled:.4f}")
    
    # Generate a random pattern for comparison
    random_pattern = np.random.choice([-1, 1], size=network.num_neurons)
    energy_random = network.compute_energy(random_pattern)
    print(f"Energy of random pattern:   {energy_random:.4f}")
    
    # ========================================================================
    # 5. NETWORK INFORMATION
    # ========================================================================
    
    print("\n5. Network information...")
    
    info = network.get_network_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nWeight matrix:")
    print(network.weights)

def test_multiple_patterns():
    """
    Test storing and recalling multiple patterns
    
    This demonstrates the capacity limitations of Hopfield networks
    and the emergence of spurious states when too many patterns are stored.
    """
    print("\n" + "=" * 60)
    print("MULTIPLE PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    # Create network
    network = HopfieldNetwork(num_neurons=6)
    
    # Define multiple patterns to store
    patterns = [
        np.array([1, -1, 1, -1, 1, -1]),   # Pattern 1: alternating
        np.array([1, 1, -1, -1, 1, 1]),    # Pattern 2: pairs
        np.array([-1, 1, 1, -1, -1, 1])    # Pattern 3: mixed
    ]
    
    print(f"Storing {len(patterns)} patterns...")
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i+1}: {pattern}")
    
    # Store all patterns
    network.store_patterns(patterns)
    
    # Test recall for each pattern
    print("\nTesting recall for each stored pattern...")
    
    for i, pattern in enumerate(patterns):
        print(f"\nTesting pattern {i+1}: {pattern}")
        
        # Test perfect recall
        recalled, converged, iterations = network.recall_pattern(pattern)
        success = np.array_equal(pattern, recalled)
        
        print(f"Recalled: {recalled}")
        print(f"Success:  {success}")
        print(f"Energy:   {network.compute_energy(recalled):.4f}")
        
        # Test with noise
        noisy = pattern.copy()
        flip_idx = np.random.randint(0, len(pattern))
        noisy[flip_idx] = -noisy[flip_idx]
        
        print(f"Noisy:    {noisy}")
        recalled_noisy, _, _ = network.recall_pattern(noisy)
        success_noisy = np.array_equal(pattern, recalled_noisy)
        
        print(f"Recalled from noisy: {recalled_noisy}")
        print(f"Success from noisy:  {success_noisy}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block for demonstrating Hopfield network functionality
    
    This runs comprehensive demonstrations of:
    1. Basic network operations
    2. Pattern storage and recall
    3. Noise tolerance
    4. Energy function
    5. Multiple pattern storage
    """
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run basic demonstration
    demonstrate_basic_hopfield()
    
    # Run multiple patterns test
    test_multiple_patterns()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("1. Hopfield networks can store and recall patterns")
    print("2. They have some tolerance to noise")
    print("3. Energy decreases during recall process")
    print("4. Network capacity is limited (~0.15 * num_neurons)")
    print("5. Symmetric weights and zero diagonal are crucial")
    print("=" * 60)


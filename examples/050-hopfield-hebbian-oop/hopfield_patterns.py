import numpy as np
import matplotlib.pyplot as plt
from hopfield_basic import HopfieldNetwork

# ============================================================================
# PATTERN STORAGE AND ANALYSIS FOR HOPFIELD NETWORKS
# ============================================================================

def create_simple_patterns():
    """
    Create a set of simple binary patterns for demonstration
    
    This function generates various types of simple patterns that are
    commonly used to test Hopfield networks. These patterns are designed
    to be easily distinguishable and have clear visual representations.
    
    Returns:
        dict: Dictionary containing different pattern sets:
            - 'alternating': Alternating +1/-1 patterns
            - 'blocks': Block patterns with consecutive same values
            - 'symmetric': Symmetric patterns
            - 'random': Random patterns for comparison
    
    Note:
        All patterns use bipolar representation {-1, +1} which is
        preferred for Hopfield networks over binary {0, 1}.
    """
    patterns = {}
    
    # Alternating patterns - these have clear structure and are easy to visualize
    patterns['alternating'] = [
        np.array([1, -1, 1, -1, 1, -1, 1, -1]),    # Pattern A: starts with +1
        np.array([-1, 1, -1, 1, -1, 1, -1, 1]),   # Pattern B: starts with -1
    ]
    
    # Block patterns - consecutive groups of same values
    patterns['blocks'] = [
        np.array([1, 1, 1, 1, -1, -1, -1, -1]),    # Pattern C: first half +1, second half -1
        np.array([-1, -1, 1, 1, 1, 1, -1, -1]),   # Pattern D: middle section +1
        np.array([1, -1, -1, 1, 1, -1, -1, 1]),   # Pattern E: symmetric blocks
    ]
    
    # Symmetric patterns - palindromic structure
    patterns['symmetric'] = [
        np.array([1, -1, 1, 1, 1, -1, 1]),        # Pattern F: 7-element palindrome
        np.array([-1, 1, 1, -1, 1, 1, -1]),       # Pattern G: 7-element palindrome
    ]
    
    # Random patterns for comparison and capacity testing
    np.random.seed(42)  # For reproducible results
    patterns['random'] = [
        np.random.choice([-1, 1], size=8) for _ in range(5)
    ]
    
    return patterns

def analyze_pattern_orthogonality(patterns):
    """
    Analyze the orthogonality between different patterns
    
    Orthogonal patterns (dot product = 0) are ideal for Hopfield networks
    because they don't interfere with each other during storage and recall.
    Non-orthogonal patterns can create spurious states and reduce recall accuracy.
    
    Mathematical background:
    For two patterns p1 and p2, their dot product is:
    p1 · p2 = Σ p1_i * p2_i
    
    - If p1 · p2 = 0: patterns are orthogonal (ideal)
    - If |p1 · p2| is small: patterns have low interference
    - If |p1 · p2| is large: patterns will interfere significantly
    
    Args:
        patterns (list): List of patterns to analyze
    
    Returns:
        tuple: (correlation_matrix, orthogonality_analysis)
            correlation_matrix (numpy.ndarray): Matrix of dot products between all pattern pairs
            orthogonality_analysis (dict): Analysis results including statistics
    """
    patterns = np.array(patterns)
    num_patterns = len(patterns)
    
    # Compute correlation matrix (dot products between all pairs)
    correlation_matrix = np.zeros((num_patterns, num_patterns))
    
    for i in range(num_patterns):
        for j in range(num_patterns):
            if i == j:
                # Diagonal elements: pattern with itself (should equal pattern length)
                correlation_matrix[i, j] = len(patterns[i])
            else:
                # Off-diagonal: correlation between different patterns
                correlation_matrix[i, j] = np.dot(patterns[i], patterns[j])
    
    # Analyze orthogonality
    off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    
    analysis = {
        'num_patterns': num_patterns,
        'pattern_length': len(patterns[0]),
        'correlation_matrix': correlation_matrix,
        'off_diagonal_correlations': off_diagonal,
        'mean_correlation': np.mean(np.abs(off_diagonal)),
        'max_correlation': np.max(np.abs(off_diagonal)),
        'min_correlation': np.min(np.abs(off_diagonal)),
        'num_orthogonal_pairs': np.sum(off_diagonal == 0),
        'num_pattern_pairs': len(off_diagonal),
        'orthogonality_ratio': np.sum(off_diagonal == 0) / len(off_diagonal)
    }
    
    return correlation_matrix, analysis

def test_pattern_storage_capacity(pattern_length=10, max_patterns=8):
    """
    Test the storage capacity of a Hopfield network with increasing number of patterns
    
    This experiment demonstrates the fundamental capacity limitation of Hopfield networks.
    The theoretical capacity is approximately 0.15 * N neurons, where N is the number
    of neurons. Beyond this limit, spurious states appear and recall accuracy degrades.
    
    The experiment procedure:
    1. Create a network with specified number of neurons
    2. Generate random patterns
    3. Store increasing numbers of patterns
    4. Test recall accuracy for each configuration
    5. Measure the degradation of performance
    
    Args:
        pattern_length (int, optional): Number of neurons/pattern length. Default 10.
        max_patterns (int, optional): Maximum number of patterns to test. Default 8.
    
    Returns:
        dict: Results containing:
            - pattern_counts: List of pattern counts tested
            - recall_accuracies: Recall accuracy for each pattern count
            - perfect_recalls: Number of perfect recalls for each pattern count
            - average_energies: Average energy of stored patterns
            - spurious_states: Number of spurious states encountered
    """
    print(f"Testing storage capacity with {pattern_length} neurons...")
    print(f"Theoretical capacity: {int(0.15 * pattern_length)} patterns")
    print("-" * 50)
    
    # Generate a fixed set of random patterns for consistent testing
    np.random.seed(123)  # Fixed seed for reproducible results
    all_patterns = [np.random.choice([-1, 1], size=pattern_length) 
                   for _ in range(max_patterns)]
    
    results = {
        'pattern_counts': [],
        'recall_accuracies': [],
        'perfect_recalls': [],
        'average_energies': [],
        'spurious_states': []
    }
    
    # Test with increasing number of patterns
    for num_patterns in range(1, max_patterns + 1):
        print(f"\nTesting with {num_patterns} patterns...")
        
        # Create fresh network for each test
        network = HopfieldNetwork(pattern_length)
        
        # Store the first num_patterns patterns
        patterns_to_store = all_patterns[:num_patterns]
        network.store_patterns(patterns_to_store)
        
        # Test recall for each stored pattern
        perfect_recalls = 0
        total_accuracy = 0
        total_energy = 0
        spurious_count = 0
        
        for i, pattern in enumerate(patterns_to_store):
            # Test perfect recall (no noise)
            recalled, converged, iterations = network.recall_pattern(pattern)
            
            # Check if recall was perfect
            if np.array_equal(pattern, recalled):
                perfect_recalls += 1
                accuracy = 1.0
            else:
                # Measure partial accuracy (fraction of correct bits)
                accuracy = np.mean(pattern == recalled)
                spurious_count += 1
            
            total_accuracy += accuracy
            total_energy += network.compute_energy(recalled)
            
            print(f"  Pattern {i+1}: Accuracy = {accuracy:.3f}, "
                  f"Energy = {network.compute_energy(recalled):.3f}")
        
        # Store results
        results['pattern_counts'].append(num_patterns)
        results['recall_accuracies'].append(total_accuracy / num_patterns)
        results['perfect_recalls'].append(perfect_recalls)
        results['average_energies'].append(total_energy / num_patterns)
        results['spurious_states'].append(spurious_count)
        
        print(f"  Overall accuracy: {total_accuracy / num_patterns:.3f}")
        print(f"  Perfect recalls: {perfect_recalls}/{num_patterns}")
        print(f"  Spurious states: {spurious_count}")
    
    return results

def test_noise_tolerance(network, stored_patterns, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Test how well the network can recall patterns with different levels of noise
    
    This experiment is crucial for understanding the robustness of Hopfield networks.
    Real-world applications often involve noisy or incomplete inputs, so understanding
    the noise tolerance is essential for practical applications.
    
    Noise is introduced by randomly flipping bits in the input pattern.
    The noise level represents the fraction of bits that are flipped.
    
    Args:
        network (HopfieldNetwork): Trained Hopfield network
        stored_patterns (list): List of patterns stored in the network
        noise_levels (list, optional): List of noise levels to test (0.0 to 1.0)
    
    Returns:
        dict: Results containing:
            - noise_levels: Tested noise levels
            - recall_accuracies: Average recall accuracy for each noise level
            - successful_recalls: Number of successful recalls for each noise level
            - convergence_rates: Fraction of tests that converged
    """
    print("\nTesting noise tolerance...")
    print("Noise level: fraction of bits flipped randomly")
    print("-" * 50)
    
    results = {
        'noise_levels': noise_levels,
        'recall_accuracies': [],
        'successful_recalls': [],
        'convergence_rates': []
    }
    
    for noise_level in noise_levels:
        print(f"\nTesting noise level: {noise_level:.1%}")
        
        total_accuracy = 0
        successful_recalls = 0
        convergence_count = 0
        total_tests = 0
        
        # Test each stored pattern with current noise level
        for pattern_idx, original_pattern in enumerate(stored_patterns):
            # Run multiple trials for statistical significance
            trials_per_pattern = 10
            
            for trial in range(trials_per_pattern):
                # Create noisy version of the pattern
                noisy_pattern = original_pattern.copy()
                
                # Determine number of bits to flip
                num_bits_to_flip = int(noise_level * len(original_pattern))
                
                if num_bits_to_flip > 0:
                    # Randomly select bits to flip
                    flip_indices = np.random.choice(len(original_pattern), 
                                                  size=num_bits_to_flip, 
                                                  replace=False)
                    noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
                
                # Attempt recall
                recalled, converged, iterations = network.recall_pattern(noisy_pattern)
                
                # Measure accuracy
                if np.array_equal(original_pattern, recalled):
                    accuracy = 1.0
                    successful_recalls += 1
                else:
                    accuracy = np.mean(original_pattern == recalled)
                
                total_accuracy += accuracy
                if converged:
                    convergence_count += 1
                total_tests += 1
        
        # Store results for this noise level
        avg_accuracy = total_accuracy / total_tests
        convergence_rate = convergence_count / total_tests
        
        results['recall_accuracies'].append(avg_accuracy)
        results['successful_recalls'].append(successful_recalls)
        results['convergence_rates'].append(convergence_rate)
        
        print(f"  Average accuracy: {avg_accuracy:.3f}")
        print(f"  Successful recalls: {successful_recalls}/{total_tests}")
        print(f"  Convergence rate: {convergence_rate:.3f}")
    
    return results

def visualize_pattern_storage_results(capacity_results, noise_results):
    """
    Create comprehensive visualizations of pattern storage experiments
    
    This function generates multiple plots to visualize:
    1. Storage capacity vs. recall accuracy
    2. Noise tolerance curves
    3. Energy landscapes
    4. Spurious state analysis
    
    Args:
        capacity_results (dict): Results from test_pattern_storage_capacity()
        noise_results (dict): Results from test_noise_tolerance()
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========================================================================
    # Plot 1: Storage Capacity Analysis
    # ========================================================================
    
    ax1 = axes[0, 0]
    pattern_counts = capacity_results['pattern_counts']
    recall_accuracies = capacity_results['recall_accuracies']
    
    ax1.plot(pattern_counts, recall_accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Perfect Recall')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Chance')
    
    # Mark theoretical capacity
    theoretical_capacity = int(0.15 * 10)  # Assuming 10 neurons
    if theoretical_capacity in pattern_counts:
        ax1.axvline(x=theoretical_capacity, color='orange', linestyle=':', 
                   label=f'Theoretical Capacity ({theoretical_capacity})')
    
    ax1.set_xlabel('Number of Stored Patterns')
    ax1.set_ylabel('Average Recall Accuracy')
    ax1.set_title('Storage Capacity vs. Recall Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # ========================================================================
    # Plot 2: Perfect Recalls vs. Pattern Count
    # ========================================================================
    
    ax2 = axes[0, 1]
    perfect_recalls = capacity_results['perfect_recalls']
    
    ax2.bar(pattern_counts, perfect_recalls, alpha=0.7, color='skyblue')
    ax2.plot(pattern_counts, pattern_counts, 'r--', alpha=0.7, label='Ideal (All Perfect)')
    
    ax2.set_xlabel('Number of Stored Patterns')
    ax2.set_ylabel('Number of Perfect Recalls')
    ax2.set_title('Perfect Recalls vs. Storage Load')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ========================================================================
    # Plot 3: Noise Tolerance
    # ========================================================================
    
    ax3 = axes[1, 0]
    noise_levels = [level * 100 for level in noise_results['noise_levels']]  # Convert to percentage
    noise_accuracies = noise_results['recall_accuracies']
    
    ax3.plot(noise_levels, noise_accuracies, 'ro-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Perfect Recall')
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Chance')
    
    ax3.set_xlabel('Noise Level (%)')
    ax3.set_ylabel('Average Recall Accuracy')
    ax3.set_title('Noise Tolerance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # ========================================================================
    # Plot 4: Energy Analysis
    # ========================================================================
    
    ax4 = axes[1, 1]
    average_energies = capacity_results['average_energies']
    
    ax4.plot(pattern_counts, average_energies, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Stored Patterns')
    ax4.set_ylabel('Average Energy of Recalled Patterns')
    ax4.set_title('Energy vs. Storage Load')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_pattern_interference():
    """
    Demonstrate how similar patterns interfere with each other in Hopfield networks
    
    This demonstration shows why pattern selection is crucial for Hopfield networks.
    Similar patterns (high correlation) will interfere with each other, leading to
    spurious states and poor recall performance.
    """
    print("\n" + "=" * 60)
    print("PATTERN INTERFERENCE DEMONSTRATION")
    print("=" * 60)
    
    # Create two very similar patterns (differ by only one bit)
    pattern1 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    pattern2 = np.array([1, -1, 1, -1, 1, -1, 1, 1])  # Last bit different
    
    print("Testing interference between similar patterns:")
    print(f"Pattern 1: {pattern1}")
    print(f"Pattern 2: {pattern2}")
    print(f"Correlation: {np.dot(pattern1, pattern2)}")
    print(f"Hamming distance: {np.sum(pattern1 != pattern2)}")
    
    # Store both patterns
    network = HopfieldNetwork(8)
    network.store_patterns([pattern1, pattern2])
    
    # Test recall
    print("\nRecall test results:")
    for i, pattern in enumerate([pattern1, pattern2], 1):
        recalled, converged, iterations = network.recall_pattern(pattern)
        success = np.array_equal(pattern, recalled)
        energy = network.compute_energy(recalled)
        
        print(f"Pattern {i}: Success = {success}, Energy = {energy:.3f}")
        print(f"  Original: {pattern}")
        print(f"  Recalled: {recalled}")
    
    # Compare with orthogonal patterns
    print("\n" + "-" * 40)
    print("Comparison with orthogonal patterns:")
    
    # Create orthogonal patterns
    pattern3 = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    pattern4 = np.array([1, -1, 1, -1, -1, 1, -1, 1])
    
    print(f"Pattern 3: {pattern3}")
    print(f"Pattern 4: {pattern4}")
    print(f"Correlation: {np.dot(pattern3, pattern4)}")
    
    # Store orthogonal patterns
    network_ortho = HopfieldNetwork(8)
    network_ortho.store_patterns([pattern3, pattern4])
    
    print("\nRecall test results for orthogonal patterns:")
    for i, pattern in enumerate([pattern3, pattern4], 3):
        recalled, converged, iterations = network_ortho.recall_pattern(pattern)
        success = np.array_equal(pattern, recalled)
        energy = network_ortho.compute_energy(recalled)
        
        print(f"Pattern {i}: Success = {success}, Energy = {energy:.3f}")

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating pattern storage and analysis in Hopfield networks
    
    This comprehensive demonstration covers:
    1. Creating and analyzing different types of patterns
    2. Testing storage capacity limitations
    3. Evaluating noise tolerance
    4. Demonstrating pattern interference effects
    5. Visualizing all results
    """
    print("=" * 60)
    print("HOPFIELD NETWORK PATTERN STORAGE ANALYSIS")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # ========================================================================
    # 1. CREATE AND ANALYZE PATTERNS
    # ========================================================================
    
    print("\n1. Creating and analyzing pattern sets...")
    
    pattern_sets = create_simple_patterns()
    
    # Analyze each pattern set
    for set_name, patterns in pattern_sets.items():
        if len(patterns) > 1:  # Only analyze sets with multiple patterns
            print(f"\nAnalyzing {set_name} patterns:")
            correlation_matrix, analysis = analyze_pattern_orthogonality(patterns)
            
            print(f"  Number of patterns: {analysis['num_patterns']}")
            print(f"  Pattern length: {analysis['pattern_length']}")
            print(f"  Mean correlation: {analysis['mean_correlation']:.3f}")
            print(f"  Max correlation: {analysis['max_correlation']:.3f}")
            print(f"  Orthogonal pairs: {analysis['num_orthogonal_pairs']}/{analysis['num_pattern_pairs']}")
            print(f"  Orthogonality ratio: {analysis['orthogonality_ratio']:.3f}")
    
    # ========================================================================
    # 2. TEST STORAGE CAPACITY
    # ========================================================================
    
    print("\n2. Testing storage capacity...")
    capacity_results = test_pattern_storage_capacity(pattern_length=10, max_patterns=8)
    
    # ========================================================================
    # 3. TEST NOISE TOLERANCE
    # ========================================================================
    
    print("\n3. Testing noise tolerance...")
    
    # Create a network with a few well-separated patterns for noise testing
    test_network = HopfieldNetwork(8)
    test_patterns = pattern_sets['alternating']  # Use alternating patterns
    test_network.store_patterns(test_patterns)
    
    noise_results = test_noise_tolerance(test_network, test_patterns)
    
    # ========================================================================
    # 4. DEMONSTRATE PATTERN INTERFERENCE
    # ========================================================================
    
    demonstrate_pattern_interference()
    
    # ========================================================================
    # 5. VISUALIZE RESULTS
    # ========================================================================
    
    print("\n5. Generating visualizations...")
    visualize_pattern_storage_results(capacity_results, noise_results)
    
    # ========================================================================
    # 6. SUMMARY AND CONCLUSIONS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 60)
    
    print("Key findings from pattern storage analysis:")
    print("1. Storage capacity is limited (~0.15 × number of neurons)")
    print("2. Orthogonal patterns store and recall better than correlated ones")
    print("3. Noise tolerance decreases as more patterns are stored")
    print("4. Similar patterns interfere and create spurious states")
    print("5. Energy provides a good measure of pattern stability")
    
    print("\nPractical implications:")
    print("- Choose patterns that are as different as possible")
    print("- Don't exceed the theoretical storage capacity")
    print("- Consider preprocessing to orthogonalize patterns")
    print("- Monitor energy levels to detect spurious states")
    
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the pattern storage demonstration when script is run directly
    """
    main()


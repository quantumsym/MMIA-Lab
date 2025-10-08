#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
# This script assumes 'hopfield_basic.py' is in the same directory.
from hopfield_basic import initialize_network, store_multiple_patterns, recall_pattern, compute_energy

# ============================================================================
# PATTERN STORAGE AND ANALYSIS FOR HOPFIELD NETWORKS 
# ============================================================================

def create_simple_patterns():
    """
    Create a set of simple binary patterns for demonstration.
    """
    patterns = {}
    patterns['alternating'] = [
        np.array([1, -1, 1, -1, 1, -1, 1, -1]),
        np.array([-1, 1, -1, 1, -1, 1, -1, 1]),
    ]
    patterns['blocks'] = [
        np.array([1, 1, 1, 1, -1, -1, -1, -1]),
        np.array([-1, -1, 1, 1, 1, 1, -1, -1]),
    ]
    np.random.seed(42)
    patterns['random'] = [np.random.choice([-1, 1], size=8) for _ in range(5)]
    return patterns

def analyze_pattern_orthogonality(patterns):
    """
    Analyze the orthogonality between different patterns.
    """
    patterns = np.array(patterns)
    num_patterns = len(patterns)
    correlation_matrix = np.dot(patterns, patterns.T)

    # The diagonal is the dot product of a pattern with itself, which is just its length.
    # We are interested in the off-diagonal elements for interference.
    off_diagonal_indices = np.triu_indices_from(correlation_matrix, k=1)
    off_diagonal_correlations = correlation_matrix[off_diagonal_indices]

    analysis = {
        'num_patterns': num_patterns,
        'pattern_length': patterns.shape[1],
        'correlation_matrix': correlation_matrix,
        'mean_correlation': np.mean(np.abs(off_diagonal_correlations)),
        'max_correlation': np.max(np.abs(off_diagonal_correlations)),
        'orthogonality_ratio': np.sum(off_diagonal_correlations == 0) / len(off_diagonal_correlations)
    }
    return analysis

def test_pattern_storage_capacity(pattern_length=10, max_patterns=8):
    """
    Test the storage capacity of a Hopfield network with an increasing number of patterns.
    """
    print(f"Testing storage capacity with {pattern_length} neurons...")
    print(f"Theoretical capacity: ~{0.15 * pattern_length:.2f} patterns")
    print("-" * 50)

    np.random.seed(123)
    all_patterns = [np.random.choice([-1, 1], size=pattern_length) for _ in range(max_patterns)]

    results = {
        'pattern_counts': [],
        'recall_accuracies': [],
        'perfect_recalls': [],
        'spurious_states': []
    }

    for num_patterns_to_store in range(1, max_patterns + 1):
        print(f"\nTesting with {num_patterns_to_store} patterns...")

        # Create a fresh network for each test
        network = initialize_network(pattern_length)

        patterns_to_store = all_patterns[:num_patterns_to_store]
        store_multiple_patterns(network, patterns_to_store)

        perfect_recalls_count = 0
        total_accuracy = 0
        spurious_count = 0

        for pattern in patterns_to_store:
            recalled_pattern, _, _ = recall_pattern(network, pattern)

            if np.array_equal(pattern, recalled_pattern):
                perfect_recalls_count += 1
                accuracy = 1.0
            else:
                accuracy = np.mean(pattern == recalled_pattern)
                spurious_count += 1

        total_accuracy += accuracy

        results['pattern_counts'].append(num_patterns_to_store)
        results['recall_accuracies'].append(total_accuracy / num_patterns_to_store)
        results['perfect_recalls'].append(perfect_recalls_count)
        results['spurious_states'].append(spurious_count)

        print(f"  Overall accuracy: {total_accuracy / num_patterns_to_store:.3f}")
        print(f"  Perfect recalls: {perfect_recalls_count}/{num_patterns_to_store}")

        return results

def test_noise_tolerance(network, stored_patterns, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Test how well the network can recall patterns with different levels of noise.
    """
    print("\nTesting noise tolerance...")
    print("-" * 50)

    results = {'noise_levels': noise_levels, 'recall_accuracies': []}

    for noise_level in noise_levels:
        print(f"\nTesting noise level: {noise_level:.1%}")

        total_accuracy = 0
        num_trials = 10 * len(stored_patterns) # 10 trials per stored pattern

        for _ in range(num_trials):
            # Pick a random pattern to corrupt
            original_pattern = stored_patterns[np.random.randint(len(stored_patterns))]

            noisy_pattern = original_pattern.copy()
            num_bits_to_flip = int(noise_level * len(original_pattern))

            if num_bits_to_flip > 0:
                flip_indices = np.random.choice(len(original_pattern), size=num_bits_to_flip, replace=False)
                noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]

            recalled_pattern, _, _ = recall_pattern(network, noisy_pattern)

            accuracy = np.mean(original_pattern == recalled_pattern)
            total_accuracy += accuracy

        avg_accuracy = total_accuracy / num_trials
        results['recall_accuracies'].append(avg_accuracy)
        print(f"  Average accuracy: {avg_accuracy:.3f}")

    return results

def visualize_pattern_storage_results(capacity_results, noise_results):
    """
    Create visualizations for pattern storage experiments.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Hopfield Network Performance Analysis", fontsize=16)

    # Plot 1: Storage Capacity
    ax1 = axes[0]
    pattern_counts = capacity_results['pattern_counts']
    recall_accuracies = capacity_results['recall_accuracies']
    ax1.plot(pattern_counts, recall_accuracies, 'bo-', label='Recall Accuracy')
    ax1.axhline(1.0, color='g', linestyle='--', label='Perfect Recall')
    theoretical_capacity = 0.15 * 10 # Assuming 10 neurons from the test
    ax1.axvline(theoretical_capacity, color='r', linestyle=':', label=f'Theoretical Capacity ({theoretical_capacity:.1f})')
    ax1.set_xlabel('Number of Stored Patterns')
    ax1.set_ylabel('Average Recall Accuracy')
    ax1.set_title('Storage Capacity vs. Accuracy')
    ax1.grid(True, alpha=0.5)
    ax1.legend()

    # Plot 2: Noise Tolerance
    ax2 = axes[1]
    noise_levels_percent = [level * 100 for level in noise_results['noise_levels']]
    noise_accuracies = noise_results['recall_accuracies']
    ax2.plot(noise_levels_percent, noise_accuracies, 'ro-', label='Recall Accuracy')
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('Average Recall Accuracy')
    ax2.set_title('Noise Tolerance')
    ax2.grid(True, alpha=0.5)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('hopfield_patterns.png', dpi=300, bbox_inches='tight')
    plt.savefig('hopfield_patterns.svg')
    plt.show()

def demonstrate_pattern_interference():
    """
    Demonstrate how similar (highly correlated) patterns interfere with each other.
    """
    print("\n" + "=" * 60)
    print("PATTERN INTERFERENCE DEMONSTRATION")
    print("=" * 60)

    # 1. Highly correlated patterns
    pattern1 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    pattern2 = np.array([1, -1, 1, -1, 1, -1, 1, 1]) # Only 1 bit difference
    print(f"Testing highly correlated patterns (Correlation: {np.dot(pattern1, pattern2)})")

    network_interfering = initialize_network(8)
    store_multiple_patterns(network_interfering, [pattern1, pattern2])

    recalled1, _, _ = recall_pattern(network_interfering, pattern1)
    print(f"  Recall of pattern 1 successful: {np.array_equal(pattern1, recalled1)}")

    # 2. Orthogonal patterns
    pattern3 = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    pattern4 = np.array([1, -1, 1, -1, -1, 1, -1, 1])
    print(f"\nTesting orthogonal patterns (Correlation: {np.dot(pattern3, pattern4)})")

    network_orthogonal = initialize_network(8)
    store_multiple_patterns(network_orthogonal, [pattern3, pattern4])

    recalled3, _, _ = recall_pattern(network_orthogonal, pattern3)
    print(f"  Recall of pattern 3 successful: {np.array_equal(pattern3, recalled3)}")

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating pattern storage and analysis.
    """
    print("=" * 60)
    print("HOPFIELD NETWORK PATTERN STORAGE ANALYSIS ")
    print("=" * 60)

    np.random.seed(42)

    # 1. ANALYZE PATTERN ORTHOGONALITY
    print("\n1. Analyzing pattern orthogonality...")
    pattern_sets = create_simple_patterns()
    analysis = analyze_pattern_orthogonality(pattern_sets['alternating'])
    print(f"  Alternating patterns | Mean Correlation: {analysis['mean_correlation']:.2f}")
    analysis = analyze_pattern_orthogonality(pattern_sets['blocks'])
    print(f"  Block patterns       | Mean Correlation: {analysis['mean_correlation']:.2f}")

    # 2. TEST STORAGE CAPACITY
    print("\n2. Testing storage capacity...")
    capacity_results = test_pattern_storage_capacity(pattern_length=10, max_patterns=8)

    # 3. TEST NOISE TOLERANCE
    print("\n3. Testing noise tolerance...")
    test_network = initialize_network(8)
    test_patterns = pattern_sets['alternating']
    store_multiple_patterns(test_network, test_patterns)
    noise_results = test_noise_tolerance(test_network, test_patterns)

    # 4. DEMONSTRATE PATTERN INTERFERENCE
    demonstrate_pattern_interference()

    # 5. VISUALIZE RESULTS
    print("\n5. Generating visualizations...")
    visualize_pattern_storage_results(capacity_results, noise_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()



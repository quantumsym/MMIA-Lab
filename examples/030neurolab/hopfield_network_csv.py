
# -*- coding: utf-8 -*-
"""
Hopfield Recurrent Neural Network with CSV Data Loading
=======================================================

Task: Pattern Recognition (Letters or Binary Patterns)

This script demonstrates how to use a Hopfield neural network for pattern
recognition. The Hopfield network is an associative memory that can store
and recall binary patterns. It's particularly useful for pattern completion
and noise reduction in binary data.

"""

import numpy as np
import pandas as pd
import neurolab as nl

def load_patterns_from_csv(filename):
    """
    Load binary patterns from a CSV file using pandas.

    The CSV file should contain:
    - Each row represents one pattern to memorize
    - Values should be 0 or 1 (will be converted to -1 and 1)
    - No header row expected

    Parameters:
    -----------
    filename : str
        Path to the CSV file containing binary patterns

    Returns:
    --------
    patterns : numpy.ndarray
        Array of patterns converted to bipolar format (-1, 1)
        Shape: (n_patterns, pattern_length)

    Example CSV format:
    1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1
    1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1
    """
    print(f"Loading patterns from: {filename}")

    # Load data using pandas - assumes no header
    data = pd.read_csv(filename, header=None)

    # Convert to numpy array
    patterns = data.to_numpy()

    print(f"Loaded {patterns.shape[0]} patterns, each with {patterns.shape[1]} elements")

    # Convert from binary (0,1) to bipolar (-1,1) format
    # Hopfield networks work better with bipolar encoding
    patterns = patterns.astype(float)
    patterns[patterns == 0] = -1

    print("Converted patterns to bipolar format (-1, 1)")

    return patterns

def create_hopfield_network(target_patterns):
    """
    Create and train a Hopfield neural network.

    The Hopfield network is a recurrent neural network that serves as 
    associative memory. It can store multiple patterns and recall them
    from partial or noisy inputs.

    Parameters:
    -----------
    target_patterns : numpy.ndarray
        Array of patterns to store in the network
        Shape: (n_patterns, pattern_length)
        Values should be in bipolar format (-1, 1)

    Returns:
    --------
    network : neurolab.net
        Trained Hopfield network ready for pattern recall

    Network Properties:
    - Fully connected (each neuron connects to every other)
    - Symmetric weights (w_ij = w_ji)
    - Zero diagonal weights (w_ii = 0)
    - Energy function ensures convergence to stored patterns
    """
    print(f"Creating Hopfield network for {target_patterns.shape[0]} patterns...")

    # Create Hopfield network using neurolab
    # neurolab.net.newhop() creates a Hopfield network and trains it
    # automatically using Hebbian learning rule
    network = nl.net.newhop(target_patterns)

    print("Network created and trained successfully")
    print(f"Network has {len(network.layers[0].np)} neurons")

    return network

def test_network_recall(network, test_patterns, pattern_names=None):
    """
    Test the network's ability to recall stored patterns.

    Parameters:
    -----------
    network : neurolab.net
        Trained Hopfield network
    test_patterns : numpy.ndarray
        Patterns to test (can be original patterns or noisy versions)
    pattern_names : list, optional
        Names/labels for each pattern (for display purposes)

    Returns:
    --------
    recalled_patterns : numpy.ndarray
        Network output for each test pattern
    """
    print("\nTesting network recall...")

    # Simulate network response to test patterns
    recalled_patterns = network.sim(test_patterns)

    # Analyze results
    for i, (original, recalled) in enumerate(zip(test_patterns, recalled_patterns)):
        # Calculate similarity (percentage of matching elements)
        similarity = np.mean(original == recalled) * 100

        pattern_label = pattern_names[i] if pattern_names else f"Pattern {i+1}"
        print(f"{pattern_label}: {similarity:.1f}% similarity")

        if similarity == 100.0:
            print(f"  ✓ Perfect recall for {pattern_label}")
        else:
            print(f"  ⚠ Imperfect recall for {pattern_label}")

    return recalled_patterns

def add_noise_to_pattern(pattern, noise_level=0.1):
    """
    Add random noise to a pattern for testing network robustness.

    Parameters:
    -----------
    pattern : numpy.ndarray
        Original pattern
    noise_level : float
        Fraction of elements to flip (0.0 to 1.0)

    Returns:
    --------
    noisy_pattern : numpy.ndarray
        Pattern with added noise
    """
    noisy_pattern = pattern.copy()
    n_elements = len(pattern)
    n_flips = int(n_elements * noise_level)

    # Randomly select elements to flip
    flip_indices = np.random.choice(n_elements, n_flips, replace=False)

    # Flip selected elements (1 -> -1, -1 -> 1)
    noisy_pattern[flip_indices] *= -1

    return noisy_pattern

def visualize_pattern(pattern, width=5, height=5, pattern_name="Pattern"):
    """
    Display a binary pattern as a 2D grid (useful for letter patterns).

    Parameters:
    -----------
    pattern : numpy.ndarray
        1D pattern to visualize
    width : int
        Width of the 2D representation
    height : int  
        Height of the 2D representation
    pattern_name : str
        Name to display above the pattern
    """
    print(f"\n{pattern_name}:")

    # Reshape pattern to 2D grid
    grid = pattern.reshape(height, width)

    # Convert to visual representation
    for row in grid:
        line = ""
        for val in row:
            line += "█" if val == 1 else "░"
        print(line)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function demonstrating Hopfield network usage with CSV data.
    """
    print("HOPFIELD NEURAL NETWORK - PATTERN RECOGNITION")
    print("=" * 60)

    # Configuration
    csv_filename = 'patterns.csv'  # <-- Change this to your CSV file path
    pattern_width = 5   # Width of each pattern (for visualization)
    pattern_height = 5  # Height of each pattern (for visualization)

    try:
        # Step 1: Load patterns from CSV file
        target_patterns = load_patterns_from_csv(csv_filename)

        # Optional: Define pattern names for better output
        # Adjust this list based on your actual patterns
        pattern_names = [f"Letter_{i+1}" for i in range(len(target_patterns))]

        # Step 2: Create and train Hopfield network
        hopfield_net = create_hopfield_network(target_patterns)

        # Step 3: Test perfect recall with original patterns
        print("\n" + "=" * 60)
        print("TESTING PERFECT RECALL")
        print("=" * 60)

        recalled_patterns = test_network_recall(
            hopfield_net, 
            target_patterns, 
            pattern_names
        )

        # Step 4: Visualize original and recalled patterns
        print("\n" + "=" * 60)
        print("PATTERN VISUALIZATION")
        print("=" * 60)

        for i, (original, recalled) in enumerate(zip(target_patterns, recalled_patterns)):
            print(f"\n--- {pattern_names[i]} ---")
            visualize_pattern(original, pattern_width, pattern_height, "Original")
            visualize_pattern(recalled, pattern_width, pattern_height, "Recalled")

        # Step 5: Test with noisy patterns
        print("\n" + "=" * 60)
        print("TESTING NOISE TOLERANCE")
        print("=" * 60)

        # Add 20% noise to first pattern and test recall
        if len(target_patterns) > 0:
            noisy_pattern = add_noise_to_pattern(target_patterns[0], noise_level=0.2)
            noisy_recalled = hopfield_net.sim([noisy_pattern])

            print("Testing with 20% noise added to first pattern:")
            visualize_pattern(target_patterns[0], pattern_width, pattern_height, "Original")
            visualize_pattern(noisy_pattern, pattern_width, pattern_height, "Noisy Input")
            visualize_pattern(noisy_recalled[0], pattern_width, pattern_height, "Network Output")

            # Calculate recovery success
            similarity = np.mean(target_patterns[0] == noisy_recalled[0]) * 100
            print(f"Recovery accuracy: {similarity:.1f}%")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_filename}'")
        print("Please create a CSV file with binary patterns (0s and 1s)")
        print("Each row should represent one pattern to store in the network")

        # Create example CSV for demonstration
        create_example_csv()

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your CSV file format and try again")

def create_example_csv():
    """
    Create an example CSV file with letter patterns for demonstration.
    """
    print("\nCreating example CSV file: 'patterns.csv'")

    # Example patterns: N, E, R, O (5x5 pixel letters)
    example_patterns = {
        'N': [1,0,0,0,1, 1,1,0,0,1, 1,0,1,0,1, 1,0,0,1,1, 1,0,0,0,1],
        'E': [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1],
        'R': [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,0,1],
        'O': [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0]
    }

    # Convert to DataFrame and save
    df = pd.DataFrame(list(example_patterns.values()))
    df.to_csv('patterns.csv', index=False, header=False)

    print("Example CSV created with letters N, E, R, O")
    print("You can now run the script again to see the demonstration")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()

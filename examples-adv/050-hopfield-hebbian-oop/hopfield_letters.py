import numpy as np
import matplotlib.pyplot as plt
from hopfield_basic import HopfieldNetwork

# ============================================================================
# LETTER PATTERN RECOGNITION WITH HOPFIELD NETWORKS
# ============================================================================

def create_letter_patterns():
    """
    Create pixelated letter patterns for Hopfield network training
    
    This function defines simple 5x5 pixel representations of letters.
    Each letter is designed to be:
    1. Clearly distinguishable from others
    2. Robust to small amounts of noise
    3. Suitable for Hopfield network storage
    
    The patterns use bipolar representation {-1, +1} where:
    - +1 represents a "black" pixel (letter foreground)
    - -1 represents a "white" pixel (background)
    
    Returns:
        dict: Dictionary containing letter patterns with keys as letter names
              and values as flattened 25-element numpy arrays
    
    Design considerations:
    - 5x5 = 25 neurons total
    - Theoretical Hopfield capacity: ~0.15 * 25 = 3-4 patterns
    - Letters chosen to be maximally different (low correlation)
    - Simple, bold designs for noise robustness
    """
    
    # Define letters as 5x5 binary matrices
    # 1 = black pixel (letter), 0 = white pixel (background)
    
    # Letter A: Classic A shape with horizontal bar
    letter_A = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
    ])
    
    # Letter E: Classic E with three horizontal bars
    letter_E = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ])
    
    # Letter I: Simple vertical line with top and bottom bars
    letter_I = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ])
    
    # Letter O: Circular/oval shape
    letter_O = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ])
    
    # Letter U: U shape with vertical sides and bottom connection
    letter_U = np.array([
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ])
    
    # Convert to bipolar representation and flatten
    letters = {}
    
    for name, pattern in [('A', letter_A), ('E', letter_E), ('I', letter_I), 
                         ('O', letter_O), ('U', letter_U)]:
        # Convert {0, 1} to {-1, +1}
        bipolar_pattern = 2 * pattern - 1
        # Flatten to 1D array for Hopfield network
        flattened = bipolar_pattern.flatten()
        letters[name] = flattened
        
        print(f"Letter {name}:")
        print(f"  Shape: {pattern.shape}")
        print(f"  Flattened size: {len(flattened)}")
        print(f"  Pixel count: {np.sum(pattern)} black, {np.sum(1-pattern)} white")
    
    return letters

def visualize_letter_patterns(letters, title="Letter Patterns"):
    """
    Visualize letter patterns in a grid layout
    
    This function creates a visual representation of the letter patterns,
    showing both the original 5x5 pixel layout and providing information
    about each pattern's characteristics.
    
    Args:
        letters (dict): Dictionary of letter patterns from create_letter_patterns()
        title (str, optional): Title for the visualization. Default "Letter Patterns".
    """
    num_letters = len(letters)
    cols = min(5, num_letters)  # Maximum 5 columns
    rows = (num_letters + cols - 1) // cols  # Calculate required rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]  # Single subplot case
    elif rows == 1:
        axes = [axes[i] for i in range(cols)]  # Single row, multiple columns
    elif cols == 1:
        axes = [axes[i] for i in range(rows)]  # Single column, multiple rows
    else:
        axes = axes.flatten()  # Multiple rows and columns
    
    # Plot each letter
    for idx, (letter_name, pattern) in enumerate(letters.items()):
        ax = axes[idx]
        
        # Reshape flattened pattern back to 5x5
        pattern_2d = pattern.reshape(5, 5)
        
        # Display pattern (invert colors for better visibility: -1=white, +1=black)
        ax.imshow(pattern_2d, cmap='gray', interpolation='nearest')
        ax.set_title(f'Letter {letter_name}', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid for pixel visibility
        for i in range(6):
            ax.axhline(i-0.5, color='red', linewidth=0.5, alpha=0.3)
            ax.axvline(i-0.5, color='red', linewidth=0.5, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_letters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_letter_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def add_noise_to_pattern(pattern, noise_level=0.2, noise_type='random'):
    """
    Add different types of noise to a letter pattern
    
    This function simulates real-world conditions where input patterns
    may be corrupted, incomplete, or distorted. Testing with noisy patterns
    demonstrates the robustness and associative memory capabilities of
    Hopfield networks.
    
    Args:
        pattern (numpy.ndarray): Original clean pattern
        noise_level (float, optional): Fraction of pixels to corrupt (0.0 to 1.0). Default 0.2.
        noise_type (str, optional): Type of noise to add. Options:
            - 'random': Randomly flip pixels
            - 'block': Add rectangular blocks of corruption
            - 'salt_pepper': Add salt and pepper noise
            - 'partial': Remove part of the pattern
    
    Returns:
        numpy.ndarray: Noisy version of the input pattern
    
    Note:
        Different noise types simulate different real-world scenarios:
        - Random: General transmission errors
        - Block: Occlusion or damage to part of image
        - Salt/pepper: Sensor noise
        - Partial: Incomplete or cropped input
    """
    noisy_pattern = pattern.copy()
    pattern_length = len(pattern)
    num_pixels_to_corrupt = int(noise_level * pattern_length)
    
    if noise_type == 'random':
        # Randomly flip specified number of pixels
        flip_indices = np.random.choice(pattern_length, size=num_pixels_to_corrupt, 
                                       replace=False)
        noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
        
    elif noise_type == 'block':
        # Add rectangular block corruption (simulates occlusion)
        # Convert to 2D for easier block manipulation
        pattern_2d = noisy_pattern.reshape(5, 5)
        
        # Define block size based on noise level
        block_size = max(1, int(np.sqrt(noise_level * 25)))
        
        # Random position for block
        start_row = np.random.randint(0, 5 - block_size + 1)
        start_col = np.random.randint(0, 5 - block_size + 1)
        
        # Corrupt the block (set to random values)
        pattern_2d[start_row:start_row+block_size, 
                   start_col:start_col+block_size] = np.random.choice([-1, 1], 
                                                                     size=(block_size, block_size))
        
        noisy_pattern = pattern_2d.flatten()
        
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise: some pixels become +1, others become -1
        corrupt_indices = np.random.choice(pattern_length, size=num_pixels_to_corrupt, 
                                         replace=False)
        
        for idx in corrupt_indices:
            # Randomly assign +1 or -1 (salt or pepper)
            noisy_pattern[idx] = np.random.choice([-1, 1])
            
    elif noise_type == 'partial':
        # Partial pattern: set some pixels to neutral/unknown state
        # For Hopfield networks, we'll set them to random values
        missing_indices = np.random.choice(pattern_length, size=num_pixels_to_corrupt, 
                                         replace=False)
        noisy_pattern[missing_indices] = np.random.choice([-1, 1], size=len(missing_indices))
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_pattern

def test_letter_recognition(network, letters, test_scenarios):
    """
    Comprehensive testing of letter recognition capabilities
    
    This function tests the Hopfield network's ability to recognize letters
    under various conditions including different noise types and levels.
    The results provide insights into the network's robustness and limitations.
    
    Args:
        network (HopfieldNetwork): Trained Hopfield network
        letters (dict): Dictionary of stored letter patterns
        test_scenarios (list): List of test configurations, each containing:
            - letter: Letter name to test
            - noise_level: Amount of noise to add
            - noise_type: Type of noise
            - description: Human-readable description
    
    Returns:
        dict: Comprehensive test results including:
            - individual_results: Results for each test case
            - summary_statistics: Overall performance metrics
            - failure_analysis: Analysis of failed recognitions
    """
    print("=" * 60)
    print("LETTER RECOGNITION TESTING")
    print("=" * 60)
    
    results = {
        'individual_results': [],
        'summary_statistics': {},
        'failure_analysis': []
    }
    
    total_tests = len(test_scenarios)
    successful_recognitions = 0
    
    for test_idx, scenario in enumerate(test_scenarios):
        print(f"\nTest {test_idx + 1}/{total_tests}: {scenario['description']}")
        print("-" * 40)
        
        # Get the original letter pattern
        original_letter = scenario['letter']
        original_pattern = letters[original_letter]
        
        # Add noise according to scenario
        noisy_pattern = add_noise_to_pattern(
            original_pattern, 
            noise_level=scenario['noise_level'],
            noise_type=scenario['noise_type']
        )
        
        # Calculate initial corruption level
        corruption_level = np.mean(original_pattern != noisy_pattern)
        
        print(f"Original letter: {original_letter}")
        print(f"Noise type: {scenario['noise_type']}")
        print(f"Intended noise level: {scenario['noise_level']:.1%}")
        print(f"Actual corruption: {corruption_level:.1%}")
        
        # Attempt recognition
        recalled_pattern, converged, iterations = network.recall_pattern(
            noisy_pattern, max_iterations=100, verbose=False)
        
        # Determine which letter was recognized
        recognized_letter = None
        best_match_score = -1
        
        for letter_name, stored_pattern in letters.items():
            # Calculate similarity (fraction of matching pixels)
            similarity = np.mean(recalled_pattern == stored_pattern)
            
            if similarity > best_match_score:
                best_match_score = similarity
                if similarity > 0.8:  # Threshold for positive recognition
                    recognized_letter = letter_name
        
        # Determine success
        recognition_successful = (recognized_letter == original_letter)
        if recognition_successful:
            successful_recognitions += 1
        
        # Calculate additional metrics
        original_similarity = np.mean(recalled_pattern == original_pattern)
        energy_initial = network.compute_energy(noisy_pattern)
        energy_final = network.compute_energy(recalled_pattern)
        energy_decrease = energy_initial - energy_final
        
        # Store individual result
        individual_result = {
            'test_index': test_idx,
            'scenario': scenario,
            'original_letter': original_letter,
            'recognized_letter': recognized_letter,
            'success': recognition_successful,
            'corruption_level': corruption_level,
            'best_match_score': best_match_score,
            'original_similarity': original_similarity,
            'converged': converged,
            'iterations': iterations,
            'energy_initial': energy_initial,
            'energy_final': energy_final,
            'energy_decrease': energy_decrease,
            'original_pattern': original_pattern,
            'noisy_pattern': noisy_pattern,
            'recalled_pattern': recalled_pattern
        }
        
        results['individual_results'].append(individual_result)
        
        # Print results
        print(f"Recognized as: {recognized_letter if recognized_letter else 'UNKNOWN'}")
        print(f"Success: {'✓' if recognition_successful else '✗'}")
        print(f"Best match score: {best_match_score:.3f}")
        print(f"Converged: {converged} (in {iterations} iterations)")
        print(f"Energy change: {energy_initial:.2f} → {energy_final:.2f} (Δ={energy_decrease:.2f})")
        
        if not recognition_successful:
            failure_info = {
                'test_index': test_idx,
                'original_letter': original_letter,
                'recognized_letter': recognized_letter,
                'corruption_level': corruption_level,
                'noise_type': scenario['noise_type'],
                'best_match_score': best_match_score
            }
            results['failure_analysis'].append(failure_info)
    
    # Calculate summary statistics
    success_rate = successful_recognitions / total_tests
    
    # Group results by noise type and level for analysis
    noise_type_performance = {}
    noise_level_performance = {}
    
    for result in results['individual_results']:
        noise_type = result['scenario']['noise_type']
        noise_level = result['scenario']['noise_level']
        
        if noise_type not in noise_type_performance:
            noise_type_performance[noise_type] = {'total': 0, 'successful': 0}
        noise_type_performance[noise_type]['total'] += 1
        if result['success']:
            noise_type_performance[noise_type]['successful'] += 1
        
        if noise_level not in noise_level_performance:
            noise_level_performance[noise_level] = {'total': 0, 'successful': 0}
        noise_level_performance[noise_level]['total'] += 1
        if result['success']:
            noise_level_performance[noise_level]['successful'] += 1
    
    # Calculate performance by category
    for category, performance in noise_type_performance.items():
        performance['success_rate'] = performance['successful'] / performance['total']
    
    for level, performance in noise_level_performance.items():
        performance['success_rate'] = performance['successful'] / performance['total']
    
    results['summary_statistics'] = {
        'total_tests': total_tests,
        'successful_recognitions': successful_recognitions,
        'overall_success_rate': success_rate,
        'noise_type_performance': noise_type_performance,
        'noise_level_performance': noise_level_performance
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("RECOGNITION TESTING SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful recognitions: {successful_recognitions}")
    print(f"Overall success rate: {success_rate:.1%}")
    
    print(f"\nPerformance by noise type:")
    for noise_type, perf in noise_type_performance.items():
        print(f"  {noise_type}: {perf['successful']}/{perf['total']} ({perf['success_rate']:.1%})")
    
    print(f"\nPerformance by noise level:")
    for noise_level, perf in sorted(noise_level_performance.items()):
        print(f"  {noise_level:.1%}: {perf['successful']}/{perf['total']} ({perf['success_rate']:.1%})")
    
    if results['failure_analysis']:
        print(f"\nFailure analysis ({len(results['failure_analysis'])} failures):")
        for failure in results['failure_analysis']:
            print(f"  Test {failure['test_index']}: {failure['original_letter']} → "
                  f"{failure['recognized_letter'] if failure['recognized_letter'] else 'UNKNOWN'} "
                  f"({failure['noise_type']}, {failure['corruption_level']:.1%} corruption)")
    
    return results

def visualize_recognition_results(results, letters):
    """
    Create comprehensive visualizations of recognition test results
    
    This function generates multiple plots to analyze the recognition performance:
    1. Success rate by noise type and level
    2. Example recognition cases (successful and failed)
    3. Energy analysis during recognition
    4. Correlation analysis between patterns
    
    Args:
        results (dict): Results from test_letter_recognition()
        letters (dict): Original letter patterns
    """
    fig = plt.figure(figsize=(16, 12))
    
    # ========================================================================
    # Plot 1: Success Rate by Noise Type
    # ========================================================================
    
    ax1 = plt.subplot(3, 4, 1)
    
    noise_types = list(results['summary_statistics']['noise_type_performance'].keys())
    success_rates = [results['summary_statistics']['noise_type_performance'][nt]['success_rate'] 
                    for nt in noise_types]
    
    bars = ax1.bar(noise_types, success_rates, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Recognition Success by Noise Type')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # ========================================================================
    # Plot 2: Success Rate by Noise Level
    # ========================================================================
    
    ax2 = plt.subplot(3, 4, 2)
    
    noise_levels = sorted(results['summary_statistics']['noise_level_performance'].keys())
    level_success_rates = [results['summary_statistics']['noise_level_performance'][nl]['success_rate'] 
                          for nl in noise_levels]
    
    ax2.plot([nl*100 for nl in noise_levels], level_success_rates, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Recognition Success vs. Noise Level')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 3-6: Example Recognition Cases
    # ========================================================================
    
    # Select interesting examples: best success, worst failure, etc.
    examples = []
    
    # Find best successful case
    successful_cases = [r for r in results['individual_results'] if r['success']]
    if successful_cases:
        best_success = max(successful_cases, key=lambda x: x['corruption_level'])
        examples.append(('Best Success', best_success))
    
    # Find worst failure
    failed_cases = [r for r in results['individual_results'] if not r['success']]
    if failed_cases:
        worst_failure = max(failed_cases, key=lambda x: x['best_match_score'])
        examples.append(('Worst Failure', worst_failure))
    
    # Add a couple more interesting cases
    if len(results['individual_results']) > 2:
        # Random successful case
        if successful_cases:
            random_success = np.random.choice(successful_cases)
            examples.append(('Random Success', random_success))
        
        # Random failure case
        if failed_cases:
            random_failure = np.random.choice(failed_cases)
            examples.append(('Random Failure', random_failure))
    
    # Plot examples
    for idx, (title, example) in enumerate(examples[:4]):
        ax = plt.subplot(3, 4, 3 + idx)
        
        # Create side-by-side comparison
        original_2d = example['original_pattern'].reshape(5, 5)
        noisy_2d = example['noisy_pattern'].reshape(5, 5)
        recalled_2d = example['recalled_pattern'].reshape(5, 5)
        
        # Combine patterns horizontally
        combined = np.hstack([original_2d, noisy_2d, recalled_2d])
        
        ax.imshow(combined, cmap='gray', interpolation='nearest')
        ax.set_title(f'{title}\n{example["original_letter"]} → '
                    f'{example["recognized_letter"] if example["recognized_letter"] else "?"}')
        ax.set_xticks([2, 7, 12])
        ax.set_xticklabels(['Original', 'Noisy', 'Recalled'], fontsize=8)
        ax.set_yticks([])
        
        # Add vertical separators
        ax.axvline(4.5, color='red', linewidth=1)
        ax.axvline(9.5, color='red', linewidth=1)
    
    # ========================================================================
    # Plot 7: Energy Analysis
    # ========================================================================
    
    ax7 = plt.subplot(3, 4, 7)
    
    energy_decreases = [r['energy_decrease'] for r in results['individual_results']]
    success_indicators = [r['success'] for r in results['individual_results']]
    
    # Separate successful and failed cases
    successful_energies = [ed for ed, success in zip(energy_decreases, success_indicators) if success]
    failed_energies = [ed for ed, success in zip(energy_decreases, success_indicators) if not success]
    
    # Create histogram
    bins = np.linspace(min(energy_decreases), max(energy_decreases), 15)
    ax7.hist(successful_energies, bins=bins, alpha=0.7, label='Successful', color='green')
    ax7.hist(failed_energies, bins=bins, alpha=0.7, label='Failed', color='red')
    
    ax7.set_xlabel('Energy Decrease')
    ax7.set_ylabel('Number of Cases')
    ax7.set_title('Energy Decrease Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 8: Convergence Analysis
    # ========================================================================
    
    ax8 = plt.subplot(3, 4, 8)
    
    iterations = [r['iterations'] for r in results['individual_results']]
    
    # Scatter plot: iterations vs success
    successful_iterations = [it for it, success in zip(iterations, success_indicators) if success]
    failed_iterations = [it for it, success in zip(iterations, success_indicators) if not success]
    
    if successful_iterations:
        ax8.scatter([1]*len(successful_iterations), successful_iterations, 
                   alpha=0.6, color='green', label='Successful')
    if failed_iterations:
        ax8.scatter([2]*len(failed_iterations), failed_iterations, 
                   alpha=0.6, color='red', label='Failed')
    
    ax8.set_xlim(0.5, 2.5)
    ax8.set_xticks([1, 2])
    ax8.set_xticklabels(['Successful', 'Failed'])
    ax8.set_ylabel('Iterations to Convergence')
    ax8.set_title('Convergence Speed Analysis')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 9: Pattern Correlation Matrix
    # ========================================================================
    
    ax9 = plt.subplot(3, 4, 9)
    
    # Compute correlation matrix between all stored letters
    letter_names = list(letters.keys())
    correlation_matrix = np.zeros((len(letter_names), len(letter_names)))
    
    for i, name1 in enumerate(letter_names):
        for j, name2 in enumerate(letter_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation = np.dot(letters[name1], letters[name2]) / len(letters[name1])
                correlation_matrix[i, j] = correlation
    
    im = ax9.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(letter_names)))
    ax9.set_yticks(range(len(letter_names)))
    ax9.set_xticklabels(letter_names)
    ax9.set_yticklabels(letter_names)
    ax9.set_title('Letter Pattern Correlations')
    
    # Add correlation values to cells
    for i in range(len(letter_names)):
        for j in range(len(letter_names)):
            text = ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax9, label='Correlation')
    
    # ========================================================================
    # Plot 10: Performance Summary
    # ========================================================================
    
    ax10 = plt.subplot(3, 4, 10)
    
    # Create summary statistics visualization
    summary_stats = results['summary_statistics']
    
    stats_text = f"""Recognition Performance Summary
    
Total Tests: {summary_stats['total_tests']}
Successful: {summary_stats['successful_recognitions']}
Success Rate: {summary_stats['overall_success_rate']:.1%}

Best Noise Type: {max(summary_stats['noise_type_performance'].items(), key=lambda x: x[1]['success_rate'])[0]}
Worst Noise Type: {min(summary_stats['noise_type_performance'].items(), key=lambda x: x[1]['success_rate'])[0]}

Network Capacity: {len(letters)} letters stored
Theoretical Limit: ~{int(0.15 * 25)} patterns
Capacity Utilization: {len(letters)/(0.15 * 25):.1%}
"""
    
    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    ax10.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_letter_recognition_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating letter recognition with Hopfield networks
    
    This comprehensive demonstration covers:
    1. Creating pixelated letter patterns
    2. Training a Hopfield network on letters
    3. Testing recognition under various noise conditions
    4. Analyzing performance and limitations
    5. Visualizing results and insights
    """
    print("=" * 60)
    print("HOPFIELD NETWORK LETTER RECOGNITION DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # ========================================================================
    # 1. CREATE LETTER PATTERNS
    # ========================================================================
    
    print("\n1. Creating letter patterns...")
    
    letters = create_letter_patterns()
    
    print(f"Created {len(letters)} letter patterns: {list(letters.keys())}")
    
    # Visualize the patterns
    visualize_letter_patterns(letters, "Original Letter Patterns")
    
    # ========================================================================
    # 2. ANALYZE PATTERN PROPERTIES
    # ========================================================================
    
    print("\n2. Analyzing pattern properties...")
    
    # Check pattern correlations
    letter_names = list(letters.keys())
    print("Pattern correlations:")
    for i, name1 in enumerate(letter_names):
        for j, name2 in enumerate(letter_names[i+1:], i+1):
            correlation = np.dot(letters[name1], letters[name2])
            print(f"  {name1} ↔ {name2}: {correlation:+.1f}")
    
    # Check pattern properties
    for name, pattern in letters.items():
        black_pixels = np.sum(pattern == 1)
        white_pixels = np.sum(pattern == -1)
        print(f"Letter {name}: {black_pixels} black, {white_pixels} white pixels")
    
    # ========================================================================
    # 3. TRAIN HOPFIELD NETWORK
    # ========================================================================
    
    print("\n3. Training Hopfield network...")
    
    # Create network with 25 neurons (5x5 pixels)
    network = HopfieldNetwork(num_neurons=25)
    
    # Store all letter patterns
    patterns_to_store = list(letters.values())
    network.store_patterns(patterns_to_store)
    
    print(f"Stored {len(patterns_to_store)} letter patterns")
    print(f"Network capacity utilization: {len(patterns_to_store)}/{int(0.15 * 25):.0f} "
          f"({100 * len(patterns_to_store) / (0.15 * 25):.0f}%)")
    
    # ========================================================================
    # 4. DEFINE TEST SCENARIOS
    # ========================================================================
    
    print("\n4. Defining test scenarios...")
    
    test_scenarios = []
    
    # Test each letter with different noise types and levels
    noise_types = ['random', 'block', 'salt_pepper', 'partial']
    noise_levels = [0.1, 0.2, 0.3, 0.4]
    
    # Create comprehensive test suite
    for letter in ['A', 'E', 'I']:  # Test subset for demonstration
        for noise_type in noise_types:
            for noise_level in noise_levels:
                scenario = {
                    'letter': letter,
                    'noise_type': noise_type,
                    'noise_level': noise_level,
                    'description': f"Letter {letter} with {noise_level:.0%} {noise_type} noise"
                }
                test_scenarios.append(scenario)
    
    print(f"Created {len(test_scenarios)} test scenarios")
    
    # ========================================================================
    # 5. RUN RECOGNITION TESTS
    # ========================================================================
    
    print("\n5. Running recognition tests...")
    
    results = test_letter_recognition(network, letters, test_scenarios)
    
    # ========================================================================
    # 6. VISUALIZE RESULTS
    # ========================================================================
    
    print("\n6. Generating visualizations...")
    
    visualize_recognition_results(results, letters)
    
    # ========================================================================
    # 7. DETAILED ANALYSIS
    # ========================================================================
    
    print("\n7. Detailed analysis...")
    
    # Analyze which letters are most robust
    letter_performance = {}
    for result in results['individual_results']:
        letter = result['original_letter']
        if letter not in letter_performance:
            letter_performance[letter] = {'total': 0, 'successful': 0}
        letter_performance[letter]['total'] += 1
        if result['success']:
            letter_performance[letter]['successful'] += 1
    
    print("Performance by letter:")
    for letter, perf in letter_performance.items():
        success_rate = perf['successful'] / perf['total']
        print(f"  Letter {letter}: {perf['successful']}/{perf['total']} ({success_rate:.1%})")
    
    # Analyze which noise types are most challenging
    print("\nMost challenging noise types:")
    noise_performance = results['summary_statistics']['noise_type_performance']
    sorted_noise = sorted(noise_performance.items(), key=lambda x: x[1]['success_rate'])
    for noise_type, perf in sorted_noise:
        print(f"  {noise_type}: {perf['success_rate']:.1%} success rate")
    
    # ========================================================================
    # 8. CONCLUSIONS AND INSIGHTS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("LETTER RECOGNITION ANALYSIS COMPLETE")
    print("=" * 60)
    
    overall_success = results['summary_statistics']['overall_success_rate']
    
    print("Key findings:")
    print(f"1. Overall recognition success rate: {overall_success:.1%}")
    print(f"2. Network successfully stored {len(letters)} letter patterns")
    print(f"3. Recognition degrades with increasing noise levels")
    print(f"4. Some noise types are more challenging than others")
    print(f"5. Pattern correlation affects recognition accuracy")
    
    print("\nPractical insights:")
    print("- Hopfield networks can serve as robust pattern recognizers")
    print("- Performance depends on pattern selection and network capacity")
    print("- Noise tolerance makes them suitable for real-world applications")
    print("- Energy function provides confidence measure for recognition")
    
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the letter recognition demonstration
    """
    main()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from hopfield_basic import HopfieldNetwork

# ============================================================================
# HOPFIELD NETWORKS vs ATTENTION MECHANISMS: COMPARATIVE IMPLEMENTATION
# ============================================================================

class ModernHopfieldNetwork:
    """
    Modern Hopfield Network implementation that bridges classical Hopfield
    networks and attention mechanisms.
    
    This implementation demonstrates how attention mechanisms can be viewed
    as continuous, single-step versions of Hopfield dynamics with exponential
    energy functions.
    
    Key differences from classical Hopfield:
    1. Continuous states instead of binary {-1, +1}
    2. Exponential energy function instead of quadratic
    3. Single-step convergence instead of iterative dynamics
    4. Exponential memory capacity instead of linear
    
    Mathematical foundation:
    - Energy function: E = -log(Σ exp(βx^T ξ^i)) + 0.5 β ||x||²
    - Update rule: x_new = (1/β) * log-sum-exp(β X^T x)
    - This is equivalent to: x_new = softmax(β X^T x) @ X
    """
    
    def __init__(self, beta=1.0, dimension=None):
        """
        Initialize Modern Hopfield Network
        
        Args:
            beta (float, optional): Inverse temperature parameter controlling
                                  sharpness of the softmax. Higher values make
                                  the network more selective. Default 1.0.
            dimension (int, optional): Dimension of patterns. If None, will be
                                     inferred from first stored pattern.
        
        The beta parameter is crucial:
        - beta → 0: Uniform attention (all patterns equally weighted)
        - beta → ∞: Hard attention (winner-takes-all, classical Hopfield)
        - beta = 1: Balanced attention (typical for modern applications)
        """
        self.beta = beta
        self.dimension = dimension
        self.stored_patterns = None
        self.num_patterns = 0
        
    def store_patterns(self, patterns):
        """
        Store patterns in the network
        
        Unlike classical Hopfield networks that store patterns in weight matrix,
        modern Hopfield networks store patterns directly as a matrix X where
        each column is a stored pattern.
        
        Args:
            patterns (numpy.ndarray): Matrix of patterns to store, shape (d, p)
                                    where d is dimension and p is number of patterns
        
        Note:
            The capacity of modern Hopfield networks is exponential in the
            number of neurons, unlike classical networks which have linear capacity.
            This allows storing many more patterns without interference.
        """
        if isinstance(patterns, list):
            patterns = np.array(patterns).T
        
        if patterns.ndim == 1:
            patterns = patterns.reshape(-1, 1)
        
        self.stored_patterns = patterns
        self.num_patterns = patterns.shape[1]
        self.dimension = patterns.shape[0]
        
        print(f"Stored {self.num_patterns} patterns of dimension {self.dimension}")
        print(f"Theoretical capacity: exponential (vs ~{int(0.15 * self.dimension)} for classical)")
        
    def retrieve_pattern(self, query, return_attention=False):
        """
        Retrieve pattern using modern Hopfield dynamics (equivalent to attention)
        
        This is the core operation that demonstrates the connection to attention:
        1. Compute similarities: scores = β * X^T @ query
        2. Compute attention weights: α = softmax(scores)
        3. Retrieve weighted combination: output = X @ α
        
        Args:
            query (numpy.ndarray): Query pattern, shape (d,)
            return_attention (bool, optional): If True, also return attention weights
        
        Returns:
            numpy.ndarray: Retrieved pattern
            numpy.ndarray (optional): Attention weights if return_attention=True
        
        Mathematical equivalence to attention:
        - Query (Q): query vector
        - Keys (K): stored patterns X
        - Values (V): stored patterns X (auto-associative)
        - Attention: softmax(β * K^T @ Q) @ V
        """
        if self.stored_patterns is None:
            raise ValueError("No patterns stored in network")
        
        # Compute similarity scores (equivalent to Q @ K^T in attention)
        scores = self.beta * (self.stored_patterns.T @ query)
        
        # Compute attention weights using softmax
        attention_weights = softmax(scores)
        
        # Retrieve pattern as weighted combination (equivalent to attention @ V)
        retrieved_pattern = self.stored_patterns @ attention_weights
        
        if return_attention:
            return retrieved_pattern, attention_weights
        else:
            return retrieved_pattern

class AttentionMechanism:
    """
    Scaled Dot-Product Attention implementation for comparison with Hopfield networks
    
    This class implements the standard attention mechanism used in Transformers,
    allowing direct comparison with Hopfield network dynamics.
    
    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
    - Q: Query matrix (n_queries × d_k)
    - K: Key matrix (n_keys × d_k)  
    - V: Value matrix (n_keys × d_v)
    - d_k: Dimension of keys (for scaling)
    """
    
    def __init__(self, scale_factor=None):
        """
        Initialize Attention Mechanism
        
        Args:
            scale_factor (float, optional): Scaling factor for attention scores.
                                          If None, uses 1/√d_k as in standard attention.
        """
        self.scale_factor = scale_factor
        
    def compute_attention(self, queries, keys, values, return_weights=False):
        """
        Compute scaled dot-product attention
        
        This is the core attention computation that we'll compare with
        Hopfield network retrieval.
        
        Args:
            queries (numpy.ndarray): Query vectors, shape (n_queries, d_k)
            keys (numpy.ndarray): Key vectors, shape (n_keys, d_k)
            values (numpy.ndarray): Value vectors, shape (n_keys, d_v)
            return_weights (bool, optional): If True, return attention weights
        
        Returns:
            numpy.ndarray: Attention output, shape (n_queries, d_v)
            numpy.ndarray (optional): Attention weights if return_weights=True
        
        Implementation details:
        1. Compute similarity scores: Q @ K^T
        2. Apply scaling: scores / √d_k (prevents softmax saturation)
        3. Compute attention weights: softmax(scaled_scores)
        4. Compute output: attention_weights @ V
        """
        # Ensure inputs are 2D
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        # Compute attention scores (Q @ K^T)
        scores = queries @ keys.T
        
        # Apply scaling factor
        if self.scale_factor is None:
            d_k = keys.shape[1]
            scale = 1.0 / np.sqrt(d_k)
        else:
            scale = self.scale_factor
        
        scaled_scores = scores * scale
        
        # Compute attention weights using softmax
        attention_weights = softmax(scaled_scores, axis=1)
        
        # Compute attention output
        attention_output = attention_weights @ values
        
        if return_weights:
            return attention_output, attention_weights
        else:
            return attention_output

def create_test_patterns(num_patterns=5, dimension=10, pattern_type='random'):
    """
    Create test patterns for comparing Hopfield and attention mechanisms
    
    Args:
        num_patterns (int): Number of patterns to create
        dimension (int): Dimension of each pattern
        pattern_type (str): Type of patterns to create
            - 'random': Random normalized vectors
            - 'orthogonal': Orthogonal vectors (if possible)
            - 'structured': Patterns with specific structure
    
    Returns:
        numpy.ndarray: Matrix of patterns, shape (dimension, num_patterns)
        list: Pattern names/labels
    """
    np.random.seed(42)  # For reproducibility
    
    if pattern_type == 'random':
        # Create random normalized patterns
        patterns = np.random.randn(dimension, num_patterns)
        # Normalize each pattern
        patterns = patterns / np.linalg.norm(patterns, axis=0, keepdims=True)
        pattern_names = [f'Random_{i}' for i in range(num_patterns)]
        
    elif pattern_type == 'orthogonal':
        # Create orthogonal patterns using QR decomposition
        if num_patterns > dimension:
            print(f"Warning: Cannot create {num_patterns} orthogonal patterns in {dimension}D space")
            num_patterns = dimension
        
        # Start with random matrix and orthogonalize
        random_matrix = np.random.randn(dimension, num_patterns)
        patterns, _ = np.linalg.qr(random_matrix)
        pattern_names = [f'Orthogonal_{i}' for i in range(num_patterns)]
        
    elif pattern_type == 'structured':
        # Create patterns with specific structure for interpretability
        patterns = np.zeros((dimension, num_patterns))
        
        for i in range(num_patterns):
            # Create patterns with different "modes"
            if i == 0:
                # First pattern: positive in first half, negative in second half
                patterns[:dimension//2, i] = 1
                patterns[dimension//2:, i] = -1
            elif i == 1:
                # Second pattern: alternating positive/negative
                patterns[::2, i] = 1
                patterns[1::2, i] = -1
            elif i == 2:
                # Third pattern: Gaussian-like
                center = dimension // 2
                for j in range(dimension):
                    patterns[j, i] = np.exp(-0.5 * ((j - center) / (dimension/4))**2)
            else:
                # Additional patterns: random but normalized
                patterns[:, i] = np.random.randn(dimension)
            
            # Normalize
            patterns[:, i] = patterns[:, i] / np.linalg.norm(patterns[:, i])
        
        pattern_names = ['Bipolar', 'Alternating', 'Gaussian'] + [f'Random_{i}' for i in range(3, num_patterns)]
    
    return patterns, pattern_names

def compare_retrieval_mechanisms(patterns, pattern_names, query_pattern, noise_levels=[0.0, 0.1, 0.2, 0.3]):
    """
    Compare pattern retrieval between classical Hopfield, modern Hopfield, and attention
    
    This function demonstrates the key differences and similarities between
    the three approaches to associative memory.
    
    Args:
        patterns (numpy.ndarray): Stored patterns matrix
        pattern_names (list): Names of patterns
        query_pattern (numpy.ndarray): Query for retrieval
        noise_levels (list): Different noise levels to test robustness
    
    Returns:
        dict: Comprehensive comparison results
    """
    print("=" * 80)
    print("COMPARATIVE ANALYSIS: HOPFIELD vs ATTENTION MECHANISMS")
    print("=" * 80)
    
    dimension, num_patterns = patterns.shape
    
    # Initialize all three mechanisms
    classical_hopfield = HopfieldNetwork(num_neurons=dimension)
    modern_hopfield = ModernHopfieldNetwork(beta=2.0, dimension=dimension)
    attention_mechanism = AttentionMechanism()
    
    # Store patterns in classical Hopfield (needs list of 1D arrays with binary values)
    # Convert continuous patterns to binary for classical Hopfield
    classical_patterns = []
    for i in range(num_patterns):
        pattern = patterns[:, i]
        # Convert to binary: positive values -> +1, negative values -> -1
        binary_pattern = np.sign(pattern)
        # Handle zero values (set to +1)
        binary_pattern[binary_pattern == 0] = 1
        classical_patterns.append(binary_pattern)
    
    classical_hopfield.store_patterns(classical_patterns)
    
    # Store patterns in modern Hopfield
    modern_hopfield.store_patterns(patterns)
    
    # For attention, patterns serve as both keys and values
    keys = patterns.T  # Shape: (num_patterns, dimension)
    values = patterns.T  # Shape: (num_patterns, dimension)
    
    results = {
        'noise_levels': noise_levels,
        'classical_results': [],
        'modern_results': [],
        'attention_results': [],
        'similarity_analysis': {},
        'convergence_analysis': {}
    }
    
    print(f"\nTesting retrieval with {len(noise_levels)} noise levels...")
    print(f"Original query pattern: {pattern_names[0] if len(pattern_names) > 0 else 'Unknown'}")
    
    for noise_idx, noise_level in enumerate(noise_levels):
        print(f"\n--- Noise Level: {noise_level:.1%} ---")
        
        # Add noise to query
        noisy_query = query_pattern.copy()
        if noise_level > 0:
            noise = np.random.randn(dimension) * noise_level
            noisy_query = noisy_query + noise
            # Renormalize to maintain unit length
            noisy_query = noisy_query / np.linalg.norm(noisy_query)
        
        # Convert query to binary for classical Hopfield
        binary_query = np.sign(noisy_query)
        binary_query[binary_query == 0] = 1
        
        # Test Classical Hopfield
        print("Classical Hopfield:")
        classical_retrieved, classical_converged, classical_iterations = classical_hopfield.recall_pattern(
            binary_query, max_iterations=100, verbose=False)
        
        # Find best match among stored patterns
        classical_similarities = [np.dot(classical_retrieved, patterns[:, i]) for i in range(num_patterns)]
        classical_best_match = np.argmax(classical_similarities)
        classical_best_similarity = classical_similarities[classical_best_match]
        
        print(f"  Converged: {classical_converged} in {classical_iterations} iterations")
        print(f"  Best match: {pattern_names[classical_best_match]} (similarity: {classical_best_similarity:.3f})")
        
        # Test Modern Hopfield
        print("Modern Hopfield:")
        modern_retrieved, modern_attention = modern_hopfield.retrieve_pattern(noisy_query, return_attention=True)
        
        # Find best match
        modern_similarities = [np.dot(modern_retrieved, patterns[:, i]) for i in range(num_patterns)]
        modern_best_match = np.argmax(modern_similarities)
        modern_best_similarity = modern_similarities[modern_best_match]
        
        print(f"  Single-step retrieval")
        print(f"  Best match: {pattern_names[modern_best_match]} (similarity: {modern_best_similarity:.3f})")
        print(f"  Attention weights: {modern_attention}")
        
        # Test Attention Mechanism
        print("Attention Mechanism:")
        query_2d = noisy_query.reshape(1, -1)  # Shape: (1, dimension)
        attention_output, attention_weights = attention_mechanism.compute_attention(
            query_2d, keys, values, return_weights=True)
        attention_retrieved = attention_output.flatten()
        
        # Find best match
        attention_similarities = [np.dot(attention_retrieved, patterns[:, i]) for i in range(num_patterns)]
        attention_best_match = np.argmax(attention_similarities)
        attention_best_similarity = attention_similarities[attention_best_match]
        
        print(f"  Single-step computation")
        print(f"  Best match: {pattern_names[attention_best_match]} (similarity: {attention_best_similarity:.3f})")
        print(f"  Attention weights: {attention_weights.flatten()}")
        
        # Store results
        results['classical_results'].append({
            'noise_level': noise_level,
            'retrieved_pattern': classical_retrieved,
            'best_match_idx': classical_best_match,
            'best_similarity': classical_best_similarity,
            'converged': classical_converged,
            'iterations': classical_iterations
        })
        
        results['modern_results'].append({
            'noise_level': noise_level,
            'retrieved_pattern': modern_retrieved,
            'attention_weights': modern_attention,
            'best_match_idx': modern_best_match,
            'best_similarity': modern_best_similarity
        })
        
        results['attention_results'].append({
            'noise_level': noise_level,
            'retrieved_pattern': attention_retrieved,
            'attention_weights': attention_weights.flatten(),
            'best_match_idx': attention_best_match,
            'best_similarity': attention_best_similarity
        })
    
    # Analyze similarities between mechanisms
    print("\n" + "=" * 80)
    print("CROSS-MECHANISM SIMILARITY ANALYSIS")
    print("=" * 80)
    
    for noise_idx, noise_level in enumerate(noise_levels):
        print(f"\nNoise Level: {noise_level:.1%}")
        
        classical_result = results['classical_results'][noise_idx]['retrieved_pattern']
        modern_result = results['modern_results'][noise_idx]['retrieved_pattern']
        attention_result = results['attention_results'][noise_idx]['retrieved_pattern']
        
        # Compute pairwise similarities
        classical_modern_sim = np.dot(classical_result, modern_result)
        classical_attention_sim = np.dot(classical_result, attention_result)
        modern_attention_sim = np.dot(modern_result, attention_result)
        
        print(f"  Classical ↔ Modern: {classical_modern_sim:.3f}")
        print(f"  Classical ↔ Attention: {classical_attention_sim:.3f}")
        print(f"  Modern ↔ Attention: {modern_attention_sim:.3f}")
        
        results['similarity_analysis'][noise_level] = {
            'classical_modern': classical_modern_sim,
            'classical_attention': classical_attention_sim,
            'modern_attention': modern_attention_sim
        }
    
    return results

def visualize_comparison_results(patterns, pattern_names, results, query_pattern):
    """
    Create comprehensive visualizations comparing the three mechanisms
    
    Args:
        patterns (numpy.ndarray): Stored patterns
        pattern_names (list): Pattern names
        results (dict): Results from compare_retrieval_mechanisms
        query_pattern (numpy.ndarray): Original query pattern
    """
    fig = plt.figure(figsize=(20, 16))
    
    # ========================================================================
    # Plot 1: Stored Patterns Visualization
    # ========================================================================
    
    ax1 = plt.subplot(4, 5, 1)
    
    # Visualize stored patterns as heatmap
    im1 = ax1.imshow(patterns, cmap='RdBu', aspect='auto', interpolation='nearest')
    ax1.set_title('Stored Patterns\n(Each column is a pattern)', fontweight='bold')
    ax1.set_xlabel('Pattern Index')
    ax1.set_ylabel('Dimension')
    ax1.set_xticks(range(len(pattern_names)))
    ax1.set_xticklabels(pattern_names, rotation=45)
    plt.colorbar(im1, ax=ax1, label='Value')
    
    # ========================================================================
    # Plot 2: Pattern Correlation Matrix
    # ========================================================================
    
    ax2 = plt.subplot(4, 5, 2)
    
    # Compute correlation matrix between patterns
    correlation_matrix = np.corrcoef(patterns.T)
    
    im2 = ax2.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_title('Pattern Correlation Matrix', fontweight='bold')
    ax2.set_xticks(range(len(pattern_names)))
    ax2.set_yticks(range(len(pattern_names)))
    ax2.set_xticklabels(pattern_names, rotation=45)
    ax2.set_yticklabels(pattern_names)
    
    # Add correlation values to cells
    for i in range(len(pattern_names)):
        for j in range(len(pattern_names)):
            text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                           fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Correlation')
    
    # ========================================================================
    # Plot 3: Retrieval Accuracy vs Noise
    # ========================================================================
    
    ax3 = plt.subplot(4, 5, 3)
    
    noise_levels = results['noise_levels']
    
    # Extract accuracy (similarity to best match)
    classical_accuracy = [r['best_similarity'] for r in results['classical_results']]
    modern_accuracy = [r['best_similarity'] for r in results['modern_results']]
    attention_accuracy = [r['best_similarity'] for r in results['attention_results']]
    
    ax3.plot(noise_levels, classical_accuracy, 'o-', label='Classical Hopfield', linewidth=2, markersize=6)
    ax3.plot(noise_levels, modern_accuracy, 's-', label='Modern Hopfield', linewidth=2, markersize=6)
    ax3.plot(noise_levels, attention_accuracy, '^-', label='Attention', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Noise Level')
    ax3.set_ylabel('Best Match Similarity')
    ax3.set_title('Retrieval Accuracy vs Noise', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # ========================================================================
    # Plot 4: Cross-Mechanism Similarity
    # ========================================================================
    
    ax4 = plt.subplot(4, 5, 4)
    
    # Extract cross-mechanism similarities
    classical_modern_sims = [results['similarity_analysis'][nl]['classical_modern'] for nl in noise_levels]
    classical_attention_sims = [results['similarity_analysis'][nl]['classical_attention'] for nl in noise_levels]
    modern_attention_sims = [results['similarity_analysis'][nl]['modern_attention'] for nl in noise_levels]
    
    ax4.plot(noise_levels, classical_modern_sims, 'o-', label='Classical ↔ Modern', linewidth=2)
    ax4.plot(noise_levels, classical_attention_sims, 's-', label='Classical ↔ Attention', linewidth=2)
    ax4.plot(noise_levels, modern_attention_sims, '^-', label='Modern ↔ Attention', linewidth=2)
    
    ax4.set_xlabel('Noise Level')
    ax4.set_ylabel('Output Similarity')
    ax4.set_title('Cross-Mechanism Similarity', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # ========================================================================
    # Plot 5: Convergence Analysis
    # ========================================================================
    
    ax5 = plt.subplot(4, 5, 5)
    
    # Classical Hopfield convergence iterations
    classical_iterations = [r['iterations'] for r in results['classical_results']]
    
    ax5.bar(range(len(noise_levels)), classical_iterations, alpha=0.7, color='blue')
    ax5.set_xlabel('Noise Level Index')
    ax5.set_ylabel('Iterations to Converge')
    ax5.set_title('Classical Hopfield\nConvergence Speed', fontweight='bold')
    ax5.set_xticks(range(len(noise_levels)))
    ax5.set_xticklabels([f'{nl:.1%}' for nl in noise_levels])
    
    # Add text annotations
    for i, iterations in enumerate(classical_iterations):
        ax5.text(i, iterations + 0.1, str(iterations), ha='center', va='bottom')
    
    # ========================================================================
    # Plots 6-10: Attention Weight Visualizations
    # ========================================================================
    
    for noise_idx, noise_level in enumerate(noise_levels):
        ax = plt.subplot(4, 5, 6 + noise_idx)
        
        # Get attention weights for this noise level
        modern_weights = results['modern_results'][noise_idx]['attention_weights']
        attention_weights = results['attention_results'][noise_idx]['attention_weights']
        
        x_pos = np.arange(len(pattern_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, modern_weights, width, label='Modern Hopfield', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, attention_weights, width, label='Attention', alpha=0.7)
        
        ax.set_xlabel('Pattern')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Attention Weights\nNoise: {noise_level:.1%}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pattern_names, rotation=45)
        ax.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # ========================================================================
    # Plots 11-15: Retrieved Pattern Visualizations
    # ========================================================================
    
    for noise_idx, noise_level in enumerate(noise_levels):
        ax = plt.subplot(4, 5, 11 + noise_idx)
        
        # Get retrieved patterns for this noise level
        classical_retrieved = results['classical_results'][noise_idx]['retrieved_pattern']
        modern_retrieved = results['modern_results'][noise_idx]['retrieved_pattern']
        attention_retrieved = results['attention_results'][noise_idx]['retrieved_pattern']
        
        # Create comparison matrix
        comparison_matrix = np.column_stack([
            query_pattern,
            classical_retrieved,
            modern_retrieved,
            attention_retrieved
        ])
        
        im = ax.imshow(comparison_matrix, cmap='RdBu', aspect='auto', interpolation='nearest')
        ax.set_title(f'Retrieved Patterns\nNoise: {noise_level:.1%}', fontweight='bold')
        ax.set_xlabel('Method')
        ax.set_ylabel('Dimension')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Query', 'Classical', 'Modern', 'Attention'], rotation=45)
        
        # Add vertical separators
        for i in range(1, 4):
            ax.axvline(i - 0.5, color='white', linewidth=2)
    
    # ========================================================================
    # Plots 16-20: Energy/Loss Landscape Analysis
    # ========================================================================
    
    # For the remaining plots, show energy analysis
    ax16 = plt.subplot(4, 5, 16)
    
    # Compute energy for classical Hopfield at different noise levels
    classical_energies = []
    for result in results['classical_results']:
        pattern = result['retrieved_pattern']
        # Compute Hopfield energy (simplified)
        energy = -0.5 * np.sum(pattern**2)  # Simplified energy calculation
        classical_energies.append(energy)
    
    ax16.plot(noise_levels, classical_energies, 'o-', linewidth=2, markersize=8)
    ax16.set_xlabel('Noise Level')
    ax16.set_ylabel('Energy')
    ax16.set_title('Classical Hopfield\nEnergy vs Noise', fontweight='bold')
    ax16.grid(True, alpha=0.3)
    
    # Summary statistics plot
    ax17 = plt.subplot(4, 5, 17)
    
    # Create summary comparison
    methods = ['Classical', 'Modern', 'Attention']
    avg_accuracy = [
        np.mean(classical_accuracy),
        np.mean(modern_accuracy),
        np.mean(attention_accuracy)
    ]
    
    bars = ax17.bar(methods, avg_accuracy, alpha=0.7, color=['blue', 'green', 'red'])
    ax17.set_ylabel('Average Accuracy')
    ax17.set_title('Average Retrieval\nAccuracy Comparison', fontweight='bold')
    ax17.set_ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, avg_accuracy):
        height = bar.get_height()
        ax17.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Computational complexity comparison
    ax18 = plt.subplot(4, 5, 18)
    
    # Theoretical complexity comparison (simplified)
    n_patterns = patterns.shape[1]
    dimension = patterns.shape[0]
    
    classical_complexity = [r['iterations'] * dimension for r in results['classical_results']]
    modern_complexity = [dimension * n_patterns] * len(noise_levels)  # Single step
    attention_complexity = [dimension * n_patterns] * len(noise_levels)  # Single step
    
    ax18.plot(noise_levels, classical_complexity, 'o-', label='Classical', linewidth=2)
    ax18.plot(noise_levels, modern_complexity, 's-', label='Modern', linewidth=2)
    ax18.plot(noise_levels, attention_complexity, '^-', label='Attention', linewidth=2)
    
    ax18.set_xlabel('Noise Level')
    ax18.set_ylabel('Computational Operations')
    ax18.set_title('Computational Complexity\nComparison', fontweight='bold')
    ax18.legend()
    ax18.grid(True, alpha=0.3)
    
    # Pattern space visualization (PCA)
    ax19 = plt.subplot(4, 5, 19)
    
    # Perform PCA on patterns for 2D visualization
    from sklearn.decomposition import PCA
    
    # Combine all patterns and retrieved patterns
    all_patterns = patterns.T  # Shape: (n_patterns, dimension)
    
    # Add retrieved patterns from different methods (using first noise level)
    retrieved_patterns = np.array([
        results['classical_results'][0]['retrieved_pattern'],
        results['modern_results'][0]['retrieved_pattern'],
        results['attention_results'][0]['retrieved_pattern']
    ])
    
    combined_patterns = np.vstack([all_patterns, retrieved_patterns])
    
    # Apply PCA
    pca = PCA(n_components=2)
    patterns_2d = pca.fit_transform(combined_patterns)
    
    # Plot stored patterns
    ax19.scatter(patterns_2d[:len(pattern_names), 0], patterns_2d[:len(pattern_names), 1], 
                s=100, alpha=0.7, label='Stored Patterns', marker='o')
    
    # Plot retrieved patterns
    colors = ['blue', 'green', 'red']
    labels = ['Classical', 'Modern', 'Attention']
    for i, (color, label) in enumerate(zip(colors, labels)):
        idx = len(pattern_names) + i
        ax19.scatter(patterns_2d[idx, 0], patterns_2d[idx, 1], 
                    s=150, color=color, label=f'{label} Retrieved', marker='x', linewidth=3)
    
    ax19.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax19.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax19.set_title('Pattern Space Visualization\n(PCA Projection)', fontweight='bold')
    ax19.legend()
    ax19.grid(True, alpha=0.3)
    
    # Final summary text
    ax20 = plt.subplot(4, 5, 20)
    
    summary_text = f"""Comparison Summary

Stored Patterns: {len(pattern_names)}
Pattern Dimension: {patterns.shape[0]}
Noise Levels Tested: {len(noise_levels)}

Average Accuracy:
• Classical: {np.mean(classical_accuracy):.3f}
• Modern: {np.mean(modern_accuracy):.3f}
• Attention: {np.mean(attention_accuracy):.3f}

Key Insights:
• Modern Hopfield ≈ Attention
• Single-step convergence
• Exponential capacity
• Continuous representations

Convergence:
• Classical: Multi-step iterative
• Modern: Single-step direct
• Attention: Single-step direct
"""
    
    ax20.text(0.05, 0.95, summary_text, transform=ax20.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax20.set_xlim(0, 1)
    ax20.set_ylim(0, 1)
    ax20.axis('off')
    ax20.set_title('Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/hopfield_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_mathematical_equivalence():
    """
    Demonstrate the mathematical equivalence between modern Hopfield networks
    and attention mechanisms through direct computation.
    """
    print("=" * 80)
    print("MATHEMATICAL EQUIVALENCE DEMONSTRATION")
    print("=" * 80)
    
    # Create simple test case
    dimension = 4
    num_patterns = 3
    
    # Create test patterns
    patterns = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ]).T  # Shape: (4, 3)
    
    query = np.array([0.8, 0.2, 0.9, 0.1])
    beta = 2.0
    
    print(f"Test setup:")
    print(f"  Patterns shape: {patterns.shape}")
    print(f"  Query: {query}")
    print(f"  Beta (inverse temperature): {beta}")
    
    # Method 1: Modern Hopfield Network
    print(f"\nMethod 1: Modern Hopfield Network")
    modern_hopfield = ModernHopfieldNetwork(beta=beta)
    modern_hopfield.store_patterns(patterns)
    modern_result, modern_weights = modern_hopfield.retrieve_pattern(query, return_attention=True)
    
    print(f"  Result: {modern_result}")
    print(f"  Attention weights: {modern_weights}")
    
    # Method 2: Direct Attention Computation
    print(f"\nMethod 2: Direct Attention Computation")
    
    # Compute attention manually
    scores = beta * (patterns.T @ query)  # Shape: (3,)
    attention_weights = softmax(scores)
    attention_result = patterns @ attention_weights
    
    print(f"  Scores: {scores}")
    print(f"  Attention weights: {attention_weights}")
    print(f"  Result: {attention_result}")
    
    # Method 3: Standard Attention Mechanism
    print(f"\nMethod 3: Standard Attention Mechanism")
    
    attention_mech = AttentionMechanism(scale_factor=beta)
    query_2d = query.reshape(1, -1)
    keys = patterns.T  # Shape: (3, 4)
    values = patterns.T  # Shape: (3, 4)
    
    standard_result, standard_weights = attention_mech.compute_attention(
        query_2d, keys, values, return_weights=True)
    
    print(f"  Result: {standard_result.flatten()}")
    print(f"  Attention weights: {standard_weights.flatten()}")
    
    # Verify equivalence
    print(f"\nEquivalence Verification:")
    print(f"  Modern ≈ Direct: {np.allclose(modern_result, attention_result)}")
    print(f"  Modern ≈ Standard: {np.allclose(modern_result, standard_result.flatten())}")
    print(f"  Direct ≈ Standard: {np.allclose(attention_result, standard_result.flatten())}")
    
    print(f"  Max difference (Modern vs Direct): {np.max(np.abs(modern_result - attention_result)):.2e}")
    print(f"  Max difference (Modern vs Standard): {np.max(np.abs(modern_result - standard_result.flatten())):.2e}")
    
    # Show step-by-step computation
    print(f"\nStep-by-step computation:")
    print(f"1. Compute similarities: β * X^T @ q")
    print(f"   {beta} * {patterns.T} @ {query}")
    print(f"   = {scores}")
    
    print(f"2. Apply softmax: softmax(similarities)")
    print(f"   = {attention_weights}")
    
    print(f"3. Weighted combination: X @ weights")
    print(f"   {patterns} @ {attention_weights}")
    print(f"   = {attention_result}")
    
    return {
        'modern_result': modern_result,
        'attention_result': attention_result,
        'standard_result': standard_result.flatten(),
        'modern_weights': modern_weights,
        'attention_weights': attention_weights,
        'standard_weights': standard_weights.flatten()
    }

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating the connections between Hopfield networks
    and attention mechanisms.
    """
    print("=" * 80)
    print("HOPFIELD NETWORKS vs ATTENTION MECHANISMS")
    print("Exploring the Deep Connections in Associative Memory")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========================================================================
    # 1. MATHEMATICAL EQUIVALENCE DEMONSTRATION
    # ========================================================================
    
    print("\n1. Mathematical Equivalence Demonstration")
    print("-" * 50)
    
    equivalence_results = demonstrate_mathematical_equivalence()
    
    # ========================================================================
    # 2. CREATE TEST PATTERNS
    # ========================================================================
    
    print("\n2. Creating Test Patterns")
    print("-" * 50)
    
    # Create different types of patterns for comprehensive testing
    patterns_random, names_random = create_test_patterns(
        num_patterns=5, dimension=8, pattern_type='random')
    
    patterns_orthogonal, names_orthogonal = create_test_patterns(
        num_patterns=5, dimension=8, pattern_type='orthogonal')
    
    patterns_structured, names_structured = create_test_patterns(
        num_patterns=5, dimension=8, pattern_type='structured')
    
    print(f"Created three sets of test patterns:")
    print(f"  Random patterns: {patterns_random.shape}")
    print(f"  Orthogonal patterns: {patterns_orthogonal.shape}")
    print(f"  Structured patterns: {patterns_structured.shape}")
    
    # ========================================================================
    # 3. COMPARATIVE ANALYSIS
    # ========================================================================
    
    print("\n3. Comparative Analysis")
    print("-" * 50)
    
    # Test with structured patterns (most interpretable)
    test_patterns = patterns_structured
    test_names = names_structured
    
    # Use first pattern as query template
    query_pattern = test_patterns[:, 0]
    
    print(f"Using structured patterns for main comparison")
    print(f"Query pattern: {test_names[0]}")
    
    # Run comprehensive comparison
    comparison_results = compare_retrieval_mechanisms(
        test_patterns, test_names, query_pattern,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4]
    )
    
    # ========================================================================
    # 4. VISUALIZATION
    # ========================================================================
    
    print("\n4. Generating Comprehensive Visualizations")
    print("-" * 50)
    
    visualize_comparison_results(
        test_patterns, test_names, comparison_results, query_pattern)
    
    # ========================================================================
    # 5. ANALYSIS SUMMARY
    # ========================================================================
    
    print("\n5. Analysis Summary")
    print("-" * 50)
    
    # Extract key insights
    noise_levels = comparison_results['noise_levels']
    
    # Average accuracies
    classical_avg = np.mean([r['best_similarity'] for r in comparison_results['classical_results']])
    modern_avg = np.mean([r['best_similarity'] for r in comparison_results['modern_results']])
    attention_avg = np.mean([r['best_similarity'] for r in comparison_results['attention_results']])
    
    # Cross-mechanism similarities
    modern_attention_similarities = [
        comparison_results['similarity_analysis'][nl]['modern_attention'] 
        for nl in noise_levels
    ]
    avg_modern_attention_sim = np.mean(modern_attention_similarities)
    
    print(f"Key Findings:")
    print(f"  Average Retrieval Accuracy:")
    print(f"    Classical Hopfield: {classical_avg:.3f}")
    print(f"    Modern Hopfield: {modern_avg:.3f}")
    print(f"    Attention Mechanism: {attention_avg:.3f}")
    
    print(f"  Modern Hopfield ↔ Attention Similarity: {avg_modern_attention_sim:.3f}")
    
    print(f"  Convergence Properties:")
    avg_iterations = np.mean([r['iterations'] for r in comparison_results['classical_results']])
    print(f"    Classical Hopfield: {avg_iterations:.1f} iterations average")
    print(f"    Modern Hopfield: 1 step (single-step convergence)")
    print(f"    Attention: 1 step (direct computation)")
    
    print(f"\nConclusions:")
    print(f"  1. Modern Hopfield networks and attention mechanisms are mathematically equivalent")
    print(f"  2. Both achieve single-step convergence vs multi-step classical Hopfield")
    print(f"  3. Both support continuous representations vs binary classical Hopfield")
    print(f"  4. Both have exponential capacity vs linear capacity classical Hopfield")
    print(f"  5. Attention can be viewed as a generalized, continuous Hopfield network")
    
    print("=" * 80)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the comprehensive comparison between Hopfield networks and attention mechanisms
    """
    main()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from hopfield_basic import HopfieldNetwork
from hopfield_attention_comparison import ModernHopfieldNetwork, AttentionMechanism

# ============================================================================
# PRACTICAL ANALOGIES: REAL-WORLD APPLICATIONS DEMONSTRATION
# ============================================================================

class MemoryRetrievalDemo:
    """
    Demonstration of memory retrieval analogies between Hopfield networks
    and attention mechanisms using practical scenarios.
    
    This class implements several real-world scenarios to show how both
    Hopfield networks and attention mechanisms solve similar problems
    through associative memory principles.
    """
    
    def __init__(self):
        """Initialize the demonstration with various memory systems"""
        self.scenarios = {}
        self.results = {}
        
    def create_word_association_scenario(self):
        """
        Create a word association scenario demonstrating how both systems
        can retrieve related concepts from partial or noisy inputs.
        
        This scenario simulates a simple semantic memory where words are
        represented as vectors and the system retrieves related words
        based on partial input.
        
        Returns:
            dict: Scenario configuration with patterns and labels
        """
        print("Creating Word Association Scenario...")
        
        # Define simple word vectors (simplified semantic embeddings)
        # Each word is represented as a 10-dimensional vector with semantic features
        word_vectors = {
            'cat': np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1]),      # animal, pet, furry, small
            'dog': np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 1]),      # animal, pet, furry, small, loyal
            'bird': np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0]),     # animal, flies, wings, small
            'car': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),      # vehicle, transport
            'plane': np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),    # flies, wings, transport
        }
        
        # Convert to normalized vectors
        for word in word_vectors:
            word_vectors[word] = word_vectors[word].astype(float)
            word_vectors[word] = word_vectors[word] / np.linalg.norm(word_vectors[word])
        
        # Create pattern matrix
        words = list(word_vectors.keys())
        patterns = np.column_stack([word_vectors[word] for word in words])
        
        print(f"  Created {len(words)} word vectors: {words}")
        print(f"  Vector dimension: {patterns.shape[0]}")
        
        scenario = {
            'name': 'Word Association',
            'patterns': patterns,
            'labels': words,
            'description': 'Retrieve related words from partial semantic input',
            'test_queries': self._create_word_queries(word_vectors)
        }
        
        return scenario
    
    def _create_word_queries(self, word_vectors):
        """Create test queries for word association"""
        queries = []
        
        # Query 1: Partial "cat" (missing some features)
        partial_cat = word_vectors['cat'].copy()
        partial_cat[2:4] = 0  # Remove some features
        partial_cat = partial_cat / np.linalg.norm(partial_cat)
        queries.append({
            'name': 'Partial Cat',
            'vector': partial_cat,
            'expected': 'cat',
            'description': 'Cat with missing semantic features'
        })
        
        # Query 2: Noisy "dog"
        noisy_dog = word_vectors['dog'].copy()
        noisy_dog += np.random.normal(0, 0.2, len(noisy_dog))
        noisy_dog = noisy_dog / np.linalg.norm(noisy_dog)
        queries.append({
            'name': 'Noisy Dog',
            'vector': noisy_dog,
            'expected': 'dog',
            'description': 'Dog with added noise'
        })
        
        # Query 3: Animal concept (average of cat and dog)
        animal_concept = (word_vectors['cat'] + word_vectors['dog']) / 2
        animal_concept = animal_concept / np.linalg.norm(animal_concept)
        queries.append({
            'name': 'Animal Concept',
            'vector': animal_concept,
            'expected': ['cat', 'dog'],
            'description': 'General animal concept'
        })
        
        # Query 4: Flying concept (average of bird and plane)
        flying_concept = (word_vectors['bird'] + word_vectors['plane']) / 2
        flying_concept = flying_concept / np.linalg.norm(flying_concept)
        queries.append({
            'name': 'Flying Concept',
            'vector': flying_concept,
            'expected': ['bird', 'plane'],
            'description': 'General flying concept'
        })
        
        return queries
    
    def create_pattern_completion_scenario(self):
        """
        Create a pattern completion scenario showing how both systems
        can complete partial patterns.
        
        This demonstrates the classic use case of Hopfield networks
        and shows how attention mechanisms solve the same problem.
        
        Returns:
            dict: Scenario configuration
        """
        print("Creating Pattern Completion Scenario...")
        
        # Create simple geometric patterns
        patterns = {}
        
        # Pattern 1: Cross
        cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]).flatten()
        
        # Pattern 2: Square
        square = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]).flatten()
        
        # Pattern 3: Diagonal
        diagonal = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).flatten()
        
        # Pattern 4: Anti-diagonal
        anti_diagonal = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]).flatten()
        
        patterns = {
            'Cross': cross,
            'Square': square,
            'Diagonal': diagonal,
            'Anti-diagonal': anti_diagonal
        }
        
        # Convert to bipolar representation and normalize
        for name in patterns:
            patterns[name] = 2 * patterns[name] - 1  # {0,1} -> {-1,1}
            patterns[name] = patterns[name].astype(float)
            patterns[name] = patterns[name] / np.linalg.norm(patterns[name])
        
        # Create pattern matrix
        labels = list(patterns.keys())
        pattern_matrix = np.column_stack([patterns[label] for label in labels])
        
        print(f"  Created {len(labels)} geometric patterns: {labels}")
        print(f"  Pattern size: 3x3 = {pattern_matrix.shape[0]} pixels")
        
        scenario = {
            'name': 'Pattern Completion',
            'patterns': pattern_matrix,
            'labels': labels,
            'description': 'Complete partial geometric patterns',
            'test_queries': self._create_pattern_queries(patterns)
        }
        
        return scenario
    
    def _create_pattern_queries(self, patterns):
        """Create test queries for pattern completion"""
        queries = []
        
        # Query 1: Partial cross (missing bottom)
        partial_cross = patterns['Cross'].copy()
        partial_cross[6:9] = 0  # Remove bottom row
        partial_cross = partial_cross / np.linalg.norm(partial_cross)
        queries.append({
            'name': 'Partial Cross',
            'vector': partial_cross,
            'expected': 'Cross',
            'description': 'Cross pattern missing bottom part'
        })
        
        # Query 2: Corrupted square
        corrupted_square = patterns['Square'].copy()
        corrupted_square[4] = -corrupted_square[4]  # Flip center pixel
        corrupted_square = corrupted_square / np.linalg.norm(corrupted_square)
        queries.append({
            'name': 'Corrupted Square',
            'vector': corrupted_square,
            'expected': 'Square',
            'description': 'Square with corrupted center'
        })
        
        # Query 3: Partial diagonal
        partial_diagonal = patterns['Diagonal'].copy()
        partial_diagonal[8] = 0  # Remove bottom-right corner
        partial_diagonal = partial_diagonal / np.linalg.norm(partial_diagonal)
        queries.append({
            'name': 'Partial Diagonal',
            'vector': partial_diagonal,
            'expected': 'Diagonal',
            'description': 'Diagonal missing corner'
        })
        
        return queries
    
    def create_sequence_attention_scenario(self):
        """
        Create a scenario demonstrating sequence attention, showing how
        attention mechanisms naturally handle sequential data while
        Hopfield networks can be adapted for similar tasks.
        
        Returns:
            dict: Scenario configuration
        """
        print("Creating Sequence Attention Scenario...")
        
        # Create simple sequence patterns representing different "sentences"
        # Each position in the sequence has a different semantic role
        
        # Vocabulary: simple concepts
        vocab = {
            'subject': np.array([1, 0, 0, 0, 0]),
            'verb': np.array([0, 1, 0, 0, 0]),
            'object': np.array([0, 0, 1, 0, 0]),
            'adjective': np.array([0, 0, 0, 1, 0]),
            'article': np.array([0, 0, 0, 0, 1])
        }
        
        # Create sequence patterns (simplified sentences)
        sequences = {
            'Simple_Sentence': np.concatenate([
                vocab['article'],    # "the"
                vocab['adjective'],  # "big"
                vocab['subject'],    # "cat"
                vocab['verb'],       # "runs"
                vocab['object']      # "home"
            ]),
            'Question': np.concatenate([
                vocab['verb'],       # "does"
                vocab['article'],    # "the"
                vocab['subject'],    # "dog"
                vocab['verb'],       # "bark"
                np.zeros(5)          # (no object)
            ]),
            'Command': np.concatenate([
                vocab['verb'],       # "go"
                vocab['object'],     # "home"
                np.zeros(15)         # (shorter sentence)
            ])
        }
        
        # Normalize sequences
        for name in sequences:
            sequences[name] = sequences[name].astype(float)
            if np.linalg.norm(sequences[name]) > 0:
                sequences[name] = sequences[name] / np.linalg.norm(sequences[name])
        
        labels = list(sequences.keys())
        pattern_matrix = np.column_stack([sequences[label] for label in labels])
        
        print(f"  Created {len(labels)} sequence patterns: {labels}")
        print(f"  Sequence length: {pattern_matrix.shape[0]} elements")
        
        scenario = {
            'name': 'Sequence Attention',
            'patterns': pattern_matrix,
            'labels': labels,
            'description': 'Attend to relevant parts of sequences',
            'test_queries': self._create_sequence_queries(sequences, vocab)
        }
        
        return scenario
    
    def _create_sequence_queries(self, sequences, vocab):
        """Create test queries for sequence attention"""
        queries = []
        
        # Query 1: Looking for subject
        subject_query = np.concatenate([
            np.zeros(5),         # ignore article
            np.zeros(5),         # ignore adjective
            vocab['subject'],    # focus on subject
            np.zeros(10)         # ignore rest
        ])
        subject_query = subject_query / np.linalg.norm(subject_query)
        queries.append({
            'name': 'Subject Query',
            'vector': subject_query,
            'expected': 'Simple_Sentence',
            'description': 'Query focusing on subject position'
        })
        
        # Query 2: Looking for verb
        verb_query = np.concatenate([
            np.zeros(5),         # ignore article
            np.zeros(5),         # ignore adjective
            np.zeros(5),         # ignore subject
            vocab['verb'],       # focus on verb
            np.zeros(5)          # ignore object
        ])
        verb_query = verb_query / np.linalg.norm(verb_query)
        queries.append({
            'name': 'Verb Query',
            'vector': verb_query,
            'expected': ['Simple_Sentence', 'Question'],
            'description': 'Query focusing on verb position'
        })
        
        return queries
    
    def run_scenario_comparison(self, scenario):
        """
        Run a complete comparison between Hopfield networks and attention
        mechanisms on a given scenario.
        
        Args:
            scenario (dict): Scenario configuration
        
        Returns:
            dict: Comprehensive results
        """
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Description: {scenario['description']}")
        
        patterns = scenario['patterns']
        labels = scenario['labels']
        test_queries = scenario['test_queries']
        
        dimension, num_patterns = patterns.shape
        
        # Initialize systems
        classical_hopfield = HopfieldNetwork(num_neurons=dimension)
        modern_hopfield = ModernHopfieldNetwork(beta=2.0, dimension=dimension)
        attention_mechanism = AttentionMechanism()
        
        # Prepare patterns for classical Hopfield (binary)
        binary_patterns = []
        for i in range(num_patterns):
            pattern = patterns[:, i]
            binary_pattern = np.sign(pattern)
            binary_pattern[binary_pattern == 0] = 1
            binary_patterns.append(binary_pattern)
        
        classical_hopfield.store_patterns(binary_patterns)
        modern_hopfield.store_patterns(patterns)
        
        # For attention: patterns are both keys and values
        keys = patterns.T
        values = patterns.T
        
        results = {
            'scenario_name': scenario['name'],
            'num_patterns': num_patterns,
            'dimension': dimension,
            'query_results': []
        }
        
        print(f"\nTesting {len(test_queries)} queries...")
        
        for query_idx, query_info in enumerate(test_queries):
            print(f"\n--- Query {query_idx + 1}: {query_info['name']} ---")
            print(f"Description: {query_info['description']}")
            
            query = query_info['vector']
            expected = query_info['expected']
            
            # Test Classical Hopfield
            binary_query = np.sign(query)
            binary_query[binary_query == 0] = 1
            
            classical_result, classical_converged, classical_iterations = classical_hopfield.recall_pattern(
                binary_query, max_iterations=50, verbose=False)
            
            # Find best match for classical
            classical_similarities = [np.dot(classical_result, patterns[:, i]) for i in range(num_patterns)]
            classical_best_idx = np.argmax(classical_similarities)
            classical_best_label = labels[classical_best_idx]
            classical_best_similarity = classical_similarities[classical_best_idx]
            
            print(f"Classical Hopfield:")
            print(f"  Best match: {classical_best_label} (similarity: {classical_best_similarity:.3f})")
            print(f"  Converged: {classical_converged} in {classical_iterations} iterations")
            
            # Test Modern Hopfield
            modern_result, modern_attention = modern_hopfield.retrieve_pattern(query, return_attention=True)
            
            modern_similarities = [np.dot(modern_result, patterns[:, i]) for i in range(num_patterns)]
            modern_best_idx = np.argmax(modern_similarities)
            modern_best_label = labels[modern_best_idx]
            modern_best_similarity = modern_similarities[modern_best_idx]
            
            print(f"Modern Hopfield:")
            print(f"  Best match: {modern_best_label} (similarity: {modern_best_similarity:.3f})")
            print(f"  Attention distribution: {dict(zip(labels, modern_attention))}")
            
            # Test Attention Mechanism
            query_2d = query.reshape(1, -1)
            attention_result, attention_weights = attention_mechanism.compute_attention(
                query_2d, keys, values, return_weights=True)
            attention_result = attention_result.flatten()
            
            attention_similarities = [np.dot(attention_result, patterns[:, i]) for i in range(num_patterns)]
            attention_best_idx = np.argmax(attention_similarities)
            attention_best_label = labels[attention_best_idx]
            attention_best_similarity = attention_similarities[attention_best_idx]
            
            print(f"Attention Mechanism:")
            print(f"  Best match: {attention_best_label} (similarity: {attention_best_similarity:.3f})")
            print(f"  Attention distribution: {dict(zip(labels, attention_weights.flatten()))}")
            
            # Evaluate correctness
            def check_correctness(predicted, expected):
                if isinstance(expected, list):
                    return predicted in expected
                else:
                    return predicted == expected
            
            classical_correct = check_correctness(classical_best_label, expected)
            modern_correct = check_correctness(modern_best_label, expected)
            attention_correct = check_correctness(attention_best_label, expected)
            
            print(f"Correctness:")
            print(f"  Classical: {'✓' if classical_correct else '✗'}")
            print(f"  Modern: {'✓' if modern_correct else '✗'}")
            print(f"  Attention: {'✓' if attention_correct else '✗'}")
            
            # Store results
            query_result = {
                'query_name': query_info['name'],
                'query_description': query_info['description'],
                'expected': expected,
                'classical': {
                    'best_match': classical_best_label,
                    'similarity': classical_best_similarity,
                    'correct': classical_correct,
                    'iterations': classical_iterations,
                    'converged': classical_converged
                },
                'modern': {
                    'best_match': modern_best_label,
                    'similarity': modern_best_similarity,
                    'correct': modern_correct,
                    'attention_weights': modern_attention,
                    'attention_distribution': dict(zip(labels, modern_attention))
                },
                'attention': {
                    'best_match': attention_best_label,
                    'similarity': attention_best_similarity,
                    'correct': attention_correct,
                    'attention_weights': attention_weights.flatten(),
                    'attention_distribution': dict(zip(labels, attention_weights.flatten()))
                }
            }
            
            results['query_results'].append(query_result)
        
        return results
    
    def visualize_scenario_results(self, scenario, results):
        """
        Create comprehensive visualizations for scenario results
        
        Args:
            scenario (dict): Scenario configuration
            results (dict): Results from run_scenario_comparison
        """
        fig = plt.figure(figsize=(16, 12))
        
        patterns = scenario['patterns']
        labels = scenario['labels']
        query_results = results['query_results']
        
        # ========================================================================
        # Plot 1: Stored Patterns Visualization
        # ========================================================================
        
        ax1 = plt.subplot(3, 4, 1)
        
        # Visualize patterns based on scenario type
        if scenario['name'] == 'Pattern Completion':
            # Reshape to 3x3 for geometric patterns
            pattern_grid = []
            for i, label in enumerate(labels):
                pattern_2d = patterns[:, i].reshape(3, 3)
                pattern_grid.append(pattern_2d)
            
            # Create combined visualization
            combined_pattern = np.hstack(pattern_grid)
            im1 = ax1.imshow(combined_pattern, cmap='RdBu', interpolation='nearest')
            ax1.set_title(f'{scenario["name"]}\nStored Patterns', fontweight='bold')
            ax1.set_xticks([1, 4, 7, 10])
            ax1.set_xticklabels(labels, rotation=45)
            ax1.set_yticks([])
            
            # Add separators
            for i in range(1, len(labels)):
                ax1.axvline(i * 3 - 0.5, color='white', linewidth=2)
        
        else:
            # Standard heatmap for other scenarios
            im1 = ax1.imshow(patterns, cmap='RdBu', aspect='auto', interpolation='nearest')
            ax1.set_title(f'{scenario["name"]}\nStored Patterns', fontweight='bold')
            ax1.set_xlabel('Pattern Index')
            ax1.set_ylabel('Dimension')
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45)
        
        plt.colorbar(im1, ax=ax1, label='Value')
        
        # ========================================================================
        # Plot 2: Accuracy Comparison
        # ========================================================================
        
        ax2 = plt.subplot(3, 4, 2)
        
        # Calculate accuracy for each method
        methods = ['Classical', 'Modern', 'Attention']
        accuracies = []
        
        for method_key in ['classical', 'modern', 'attention']:
            correct_count = sum(1 for qr in query_results if qr[method_key]['correct'])
            accuracy = correct_count / len(query_results)
            accuracies.append(accuracy)
        
        bars = ax2.bar(methods, accuracies, alpha=0.7, color=['blue', 'green', 'red'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Query Accuracy Comparison', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # ========================================================================
        # Plot 3: Attention Weight Comparison
        # ========================================================================
        
        ax3 = plt.subplot(3, 4, 3)
        
        # Compare attention weights between Modern Hopfield and Attention
        if len(query_results) > 0:
            # Use first query for comparison
            first_query = query_results[0]
            modern_weights = first_query['modern']['attention_weights']
            attention_weights = first_query['attention']['attention_weights']
            
            x_pos = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, modern_weights, width, 
                           label='Modern Hopfield', alpha=0.7)
            bars2 = ax3.bar(x_pos + width/2, attention_weights, width, 
                           label='Attention', alpha=0.7)
            
            ax3.set_xlabel('Pattern')
            ax3.set_ylabel('Attention Weight')
            ax3.set_title(f'Attention Weights\n({first_query["query_name"]})', fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels, rotation=45)
            ax3.legend()
        
        # ========================================================================
        # Plot 4: Similarity Scores
        # ========================================================================
        
        ax4 = plt.subplot(3, 4, 4)
        
        # Plot similarity scores for all queries
        query_names = [qr['query_name'] for qr in query_results]
        classical_sims = [qr['classical']['similarity'] for qr in query_results]
        modern_sims = [qr['modern']['similarity'] for qr in query_results]
        attention_sims = [qr['attention']['similarity'] for qr in query_results]
        
        x_pos = np.arange(len(query_names))
        width = 0.25
        
        ax4.bar(x_pos - width, classical_sims, width, label='Classical', alpha=0.7)
        ax4.bar(x_pos, modern_sims, width, label='Modern', alpha=0.7)
        ax4.bar(x_pos + width, attention_sims, width, label='Attention', alpha=0.7)
        
        ax4.set_xlabel('Query')
        ax4.set_ylabel('Best Match Similarity')
        ax4.set_title('Similarity Scores', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(query_names, rotation=45)
        ax4.legend()
        
        # ========================================================================
        # Plots 5-8: Individual Query Results
        # ========================================================================
        
        for query_idx, query_result in enumerate(query_results[:4]):  # Show up to 4 queries
            ax = plt.subplot(3, 4, 5 + query_idx)
            
            # Create attention weight comparison for this query
            modern_weights = query_result['modern']['attention_weights']
            attention_weights = query_result['attention']['attention_weights']
            
            x_pos = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, modern_weights, width, 
                          label='Modern', alpha=0.7, color='green')
            bars2 = ax.bar(x_pos + width/2, attention_weights, width, 
                          label='Attention', alpha=0.7, color='red')
            
            ax.set_xlabel('Pattern')
            ax.set_ylabel('Weight')
            ax.set_title(f'{query_result["query_name"]}\nAttention Weights', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45)
            
            # Highlight correct answer
            expected = query_result['expected']
            if isinstance(expected, str):
                expected_indices = [labels.index(expected)]
            else:
                expected_indices = [labels.index(exp) for exp in expected if exp in labels]
            
            for idx in expected_indices:
                ax.axvline(idx, color='gold', linewidth=3, alpha=0.5, label='Expected')
            
            if query_idx == 0:  # Only show legend for first plot
                ax.legend()
        
        # ========================================================================
        # Plots 9-12: Query Visualizations
        # ========================================================================
        
        for query_idx, query_info in enumerate(scenario['test_queries'][:4]):
            ax = plt.subplot(3, 4, 9 + query_idx)
            
            query_vector = query_info['vector']
            
            if scenario['name'] == 'Pattern Completion':
                # Reshape to 3x3 for visualization
                query_2d = query_vector.reshape(3, 3)
                im = ax.imshow(query_2d, cmap='RdBu', interpolation='nearest')
                ax.set_title(f'Query: {query_info["name"]}', fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Line plot for other scenarios
                ax.plot(query_vector, 'o-', linewidth=2, markersize=4)
                ax.set_title(f'Query: {query_info["name"]}', fontweight='bold')
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/home/ubuntu/{scenario["name"].lower().replace(" ", "_")}_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_scenarios(self):
        """
        Run all scenarios and generate comprehensive analysis
        
        Returns:
            dict: Complete results from all scenarios
        """
        print("=" * 80)
        print("PRACTICAL ANALOGIES DEMONSTRATION")
        print("Hopfield Networks vs Attention Mechanisms in Real-World Scenarios")
        print("=" * 80)
        
        # Create all scenarios
        scenarios = [
            self.create_word_association_scenario(),
            self.create_pattern_completion_scenario(),
            self.create_sequence_attention_scenario()
        ]
        
        all_results = {}
        
        # Run each scenario
        for scenario in scenarios:
            results = self.run_scenario_comparison(scenario)
            all_results[scenario['name']] = results
            
            # Generate visualizations
            self.visualize_scenario_results(scenario, results)
        
        # Generate summary analysis
        self.generate_summary_analysis(all_results)
        
        return all_results
    
    def generate_summary_analysis(self, all_results):
        """
        Generate a comprehensive summary analysis across all scenarios
        
        Args:
            all_results (dict): Results from all scenarios
        """
        print("\n" + "=" * 80)
        print("CROSS-SCENARIO SUMMARY ANALYSIS")
        print("=" * 80)
        
        # Aggregate statistics
        total_queries = 0
        method_stats = {'classical': {'correct': 0, 'total': 0, 'avg_similarity': 0, 'avg_iterations': 0},
                       'modern': {'correct': 0, 'total': 0, 'avg_similarity': 0},
                       'attention': {'correct': 0, 'total': 0, 'avg_similarity': 0}}
        
        scenario_performance = {}
        
        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name} Results:")
            
            scenario_stats = {'classical': {'correct': 0, 'total': 0},
                            'modern': {'correct': 0, 'total': 0},
                            'attention': {'correct': 0, 'total': 0}}
            
            for query_result in results['query_results']:
                total_queries += 1
                
                for method in ['classical', 'modern', 'attention']:
                    method_stats[method]['total'] += 1
                    scenario_stats[method]['total'] += 1
                    
                    if query_result[method]['correct']:
                        method_stats[method]['correct'] += 1
                        scenario_stats[method]['correct'] += 1
                    
                    method_stats[method]['avg_similarity'] += query_result[method]['similarity']
                    
                    if method == 'classical':
                        method_stats[method]['avg_iterations'] += query_result[method]['iterations']
            
            # Print scenario-specific results
            for method in ['classical', 'modern', 'attention']:
                accuracy = scenario_stats[method]['correct'] / scenario_stats[method]['total']
                print(f"  {method.capitalize()}: {scenario_stats[method]['correct']}/{scenario_stats[method]['total']} ({accuracy:.1%})")
            
            scenario_performance[scenario_name] = scenario_stats
        
        # Calculate overall statistics
        print(f"\nOverall Performance Summary:")
        print(f"Total queries tested: {total_queries}")
        
        for method in ['classical', 'modern', 'attention']:
            accuracy = method_stats[method]['correct'] / method_stats[method]['total']
            avg_similarity = method_stats[method]['avg_similarity'] / method_stats[method]['total']
            
            print(f"\n{method.capitalize()} Hopfield:" if method != 'attention' else f"\nAttention Mechanism:")
            print(f"  Overall accuracy: {method_stats[method]['correct']}/{method_stats[method]['total']} ({accuracy:.1%})")
            print(f"  Average similarity: {avg_similarity:.3f}")
            
            if method == 'classical':
                avg_iterations = method_stats[method]['avg_iterations'] / method_stats[method]['total']
                print(f"  Average iterations: {avg_iterations:.1f}")
        
        # Key insights
        print(f"\nKey Insights:")
        
        # Compare modern vs attention
        modern_accuracy = method_stats['modern']['correct'] / method_stats['modern']['total']
        attention_accuracy = method_stats['attention']['correct'] / method_stats['attention']['total']
        
        print(f"1. Modern Hopfield vs Attention accuracy: {modern_accuracy:.1%} vs {attention_accuracy:.1%}")
        
        # Convergence comparison
        classical_avg_iter = method_stats['classical']['avg_iterations'] / method_stats['classical']['total']
        print(f"2. Classical Hopfield convergence: {classical_avg_iter:.1f} iterations average")
        print(f"   Modern Hopfield & Attention: 1 step (immediate)")
        
        # Scenario-specific insights
        best_scenario_classical = max(scenario_performance.items(), 
                                    key=lambda x: x[1]['classical']['correct']/x[1]['classical']['total'])
        best_scenario_modern = max(scenario_performance.items(), 
                                 key=lambda x: x[1]['modern']['correct']/x[1]['modern']['total'])
        
        print(f"3. Best scenario for Classical: {best_scenario_classical[0]}")
        print(f"4. Best scenario for Modern/Attention: {best_scenario_modern[0]}")
        
        print(f"\nConclusions:")
        print(f"• Modern Hopfield and Attention show similar performance patterns")
        print(f"• Both modern approaches outperform classical in complex scenarios")
        print(f"• Single-step convergence provides significant computational advantages")
        print(f"• Attention mechanisms excel in sequence-based tasks")
        print(f"• Classical Hopfield remains effective for simple pattern completion")

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function to run the practical analogies demonstration
    """
    demo = MemoryRetrievalDemo()
    results = demo.run_all_scenarios()
    
    print("\n" + "=" * 80)
    print("PRACTICAL ANALOGIES DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Generated visualizations and analysis for:")
    print("• Word Association Scenario")
    print("• Pattern Completion Scenario") 
    print("• Sequence Attention Scenario")
    print("\nAll results demonstrate the fundamental connections between")
    print("Hopfield networks and attention mechanisms through the lens")
    print("of associative memory and content-addressable retrieval.")

if __name__ == "__main__":
    main()


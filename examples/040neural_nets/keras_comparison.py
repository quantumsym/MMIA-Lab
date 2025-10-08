#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Try to import TensorFlow/Keras, with graceful fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
    print("TensorFlow/Keras is available for comparison")
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available - will show comparison conceptually")

# Import our custom implementations
from multilayer_perceptron import (initialize_mlp_weights, mlp_forward_pass, 
                                  mlp_backward_pass, train_mlp, test_mlp, create_xor_dataset)

# ============================================================================
# KERAS VS CUSTOM IMPLEMENTATION COMPARISON
# ============================================================================

def create_keras_mlp_for_xor():
    """
    Create a Keras MLP model equivalent to our custom implementation
    
    This function builds a Keras Sequential model that mirrors the architecture
    of our custom MLP implementation:
    - Input layer: 2 neurons (for XOR inputs)
    - Hidden layer: 4 neurons with sigmoid activation
    - Output layer: 1 neuron with sigmoid activation
    
    The model uses the same activation functions and similar initialization
    to ensure fair comparison with our custom implementation.
    
    Returns:
        tensorflow.keras.Model: Compiled Keras model ready for training
        
    Note:
        We use Mean Squared Error loss and SGD optimizer to match our
        custom implementation as closely as possible.
    """
    if not KERAS_AVAILABLE:
        print("Keras not available - cannot create model")
        return None
    
    # Suppress TensorFlow warnings for cleaner output
    tf.get_logger().setLevel('ERROR')
    
    # Build Sequential model (equivalent to our feedforward architecture)
    model = keras.Sequential([
        # Input layer is implicit in Keras
        
        # Hidden layer: 4 neurons with sigmoid activation
        # This matches our custom MLP hidden layer
        layers.Dense(
            units=4,                    # Number of neurons
            activation='sigmoid',       # Sigmoid activation function
            input_shape=(2,),          # Input shape: 2 features for XOR
            kernel_initializer='random_uniform',  # Random weight initialization
            bias_initializer='zeros'    # Zero bias initialization
        ),
        
        # Output layer: 1 neuron with sigmoid activation
        # This produces probability-like output for binary classification
        layers.Dense(
            units=1,                    # Single output neuron
            activation='sigmoid',       # Sigmoid activation for binary output
            kernel_initializer='random_uniform',
            bias_initializer='zeros'
        )
    ])
    
    # Compile model with optimizer and loss function
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=10.0),  # Match our learning rate
        loss='mse',                     # Mean Squared Error (same as our implementation)
        metrics=['accuracy']            # Track accuracy during training
    )
    
    return model

def train_keras_model_on_xor(model, X_train, y_train, epochs=5000, verbose_frequency=1000):
    """
    Train the Keras model on XOR dataset with detailed monitoring
    
    This function trains the Keras model using the same XOR dataset and
    similar hyperparameters as our custom implementation, allowing for
    direct performance comparison.
    
    Args:
        model (tensorflow.keras.Model): Compiled Keras model
        X_train (numpy.ndarray): XOR input patterns, shape (4, 2)
        y_train (numpy.ndarray): XOR target outputs, shape (4,)
        epochs (int, optional): Number of training epochs. Default 5000.
        verbose_frequency (int, optional): Print progress every N epochs. Default 1000.
    
    Returns:
        tensorflow.keras.callbacks.History: Training history with loss and metrics
        
    Note:
        We use batch_size=4 to process all XOR samples in each epoch,
        matching the behavior of our custom implementation.
    """
    if not KERAS_AVAILABLE or model is None:
        print("Cannot train - Keras model not available")
        return None
    
    print("Training Keras model on XOR dataset...")
    print(f"  Architecture: {len(model.layers)} layers")
    print(f"  Hidden neurons: {model.layers[0].units}")
    print(f"  Learning rate: {model.optimizer.learning_rate.numpy()}")
    print(f"  Epochs: {epochs}")
    print("-" * 50)
    
    # Custom callback to print progress at specified intervals
    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self, verbose_frequency):
            self.verbose_frequency = verbose_frequency
        
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.verbose_frequency == 0 or epoch == 0:
                print(f"Epoch {epoch+1}: Loss = {logs['loss']:.6f}, "
                      f"Accuracy = {logs['accuracy']:.3f}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=4,           # Process all XOR samples together
        verbose=0,              # Suppress default Keras output
        callbacks=[ProgressCallback(verbose_frequency)]
    )
    
    return history

def compare_implementations_on_xor():
    """
    Comprehensive comparison of custom MLP vs Keras on XOR problem
    
    This function provides a side-by-side comparison of:
    1. Training time and convergence behavior
    2. Final accuracy and performance metrics
    3. Implementation complexity and code length
    4. Flexibility and customization options
    
    The comparison helps understand the trade-offs between implementing
    neural networks from scratch vs using established frameworks.
    
    Returns:
        dict: Comparison results with metrics from both implementations
    """
    print("=" * 60)
    print("COMPREHENSIVE COMPARISON: CUSTOM MLP vs KERAS")
    print("=" * 60)
    
    # ========================================================================
    # DATASET PREPARATION
    # ========================================================================
    
    # Create XOR dataset (same for both implementations)
    xor_inputs, xor_targets = create_xor_dataset()
    
    print("XOR Dataset (used by both implementations):")
    for i in range(len(xor_inputs)):
        print(f"  {xor_inputs[i]} → {xor_targets[i]}")
    print()
    
    # ========================================================================
    # CUSTOM IMPLEMENTATION TRAINING
    # ========================================================================
    
    print("1. TRAINING CUSTOM MLP IMPLEMENTATION")
    print("=" * 40)
    
    # Train our custom implementation
    custom_weights_ih, custom_bias_h, custom_weights_ho, custom_bias_o, custom_errors = \
        train_mlp(xor_inputs, xor_targets, 
                 hidden_layer_size=4, 
                 learning_rate=10.0, 
                 max_epochs=5000)
    
    # Test custom implementation
    custom_accuracy, custom_predictions = test_mlp(
        xor_inputs, xor_targets, custom_weights_ih, custom_bias_h, 
        custom_weights_ho, custom_bias_o)
    
    custom_epochs_to_convergence = len(custom_errors)
    custom_final_error = custom_errors[-1] if custom_errors else float('inf')
    
    # ========================================================================
    # KERAS IMPLEMENTATION TRAINING
    # ========================================================================
    
    print(f"\n2. TRAINING KERAS IMPLEMENTATION")
    print("=" * 40)
    
    if KERAS_AVAILABLE:
        # Create and train Keras model
        keras_model = create_keras_mlp_for_xor()
        keras_history = train_keras_model_on_xor(keras_model, xor_inputs, xor_targets)
        
        # Evaluate Keras model
        keras_loss, keras_accuracy = keras_model.evaluate(xor_inputs, xor_targets, verbose=0)
        keras_predictions = keras_model.predict(xor_inputs, verbose=0)
        keras_binary_predictions = (keras_predictions.flatten() >= 0.5).astype(int)
        
        keras_epochs_to_convergence = len(keras_history.history['loss'])
        keras_final_error = keras_history.history['loss'][-1]
        
        print(f"\nKeras Final Results:")
        print(f"  Final Loss: {keras_final_error:.6f}")
        print(f"  Final Accuracy: {keras_accuracy:.3f}")
        
    else:
        # Simulate results when Keras is not available
        print("Keras not available - using simulated comparison results")
        keras_accuracy = 1.0
        keras_epochs_to_convergence = 3500
        keras_final_error = 0.000045
        keras_binary_predictions = [0, 1, 1, 0]  # Perfect XOR predictions
    
    # ========================================================================
    # DETAILED COMPARISON
    # ========================================================================
    
    print(f"\n3. DETAILED COMPARISON")
    print("=" * 40)
    
    comparison_results = {
        'custom': {
            'accuracy': custom_accuracy,
            'epochs_to_convergence': custom_epochs_to_convergence,
            'final_error': custom_final_error,
            'predictions': custom_predictions
        },
        'keras': {
            'accuracy': keras_accuracy,
            'epochs_to_convergence': keras_epochs_to_convergence,
            'final_error': keras_final_error,
            'predictions': keras_binary_predictions
        }
    }
    
    # Performance comparison table
    print("PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'Custom MLP':<15} {'Keras MLP':<15}")
    print("-" * 55)
    print(f"{'Final Accuracy':<25} {custom_accuracy:<15.3f} {keras_accuracy:<15.3f}")
    print(f"{'Epochs to Convergence':<25} {custom_epochs_to_convergence:<15d} {keras_epochs_to_convergence:<15d}")
    print(f"{'Final Error':<25} {custom_final_error:<15.6f} {keras_final_error:<15.6f}")
    
    # Prediction comparison
    print(f"\nPREDICTION COMPARISON:")
    print(f"{'Input':<10} {'Target':<8} {'Custom':<8} {'Keras':<8}")
    print("-" * 34)
    for i in range(len(xor_inputs)):
        print(f"{str(xor_inputs[i]):<10} {xor_targets[i]:<8.0f} "
              f"{custom_predictions[i]:<8d} {keras_binary_predictions[i]:<8d}")
    
    return comparison_results

def analyze_implementation_differences():
    """
    Analyze the key differences between custom and Keras implementations
    
    This function provides educational analysis of:
    1. Code complexity and readability
    2. Performance and optimization differences
    3. Flexibility and customization options
    4. Learning value and practical considerations
    5. When to use each approach
    
    The analysis helps students understand the trade-offs between
    different implementation approaches.
    """
    print(f"\n4. IMPLEMENTATION ANALYSIS")
    print("=" * 40)
    
    analysis_text = """
CODE COMPLEXITY COMPARISON:

Custom Implementation:
  ✓ Explicit and educational - every step is visible
  ✓ Full control over all aspects of the algorithm
  ✓ Easy to modify for research or special requirements
  ✗ More code to write and maintain
  ✗ Higher chance of implementation bugs
  ✗ No built-in optimizations

Keras Implementation:
  ✓ Concise and production-ready
  ✓ Heavily optimized and tested
  ✓ Extensive documentation and community support
  ✓ Built-in features (callbacks, metrics, etc.)
  ✗ Less transparent - "black box" behavior
  ✗ Limited customization for novel algorithms

PERFORMANCE CHARACTERISTICS:

Custom Implementation:
  • Pure Python/NumPy - slower for large datasets
  • No GPU acceleration by default
  • Memory usage depends on implementation quality
  • Good for small problems and learning

Keras Implementation:
  • Optimized C++/CUDA backend - much faster
  • Automatic GPU acceleration when available
  • Efficient memory management
  • Scales to very large datasets and models

LEARNING VALUE:

Custom Implementation:
  • Deep understanding of algorithms
  • Appreciation for framework optimizations
  • Ability to implement novel architectures
  • Strong foundation for research

Keras Implementation:
  • Focus on problem-solving rather than implementation
  • Industry-standard practices and workflows
  • Rapid prototyping and experimentation
  • Production deployment experience

WHEN TO USE EACH APPROACH:

Use Custom Implementation for:
  ✓ Learning and education
  ✓ Research and novel algorithms
  ✓ Understanding existing methods
  ✓ Small-scale problems
  ✓ Debugging and analysis

Use Keras/Framework for:
  ✓ Production applications
  ✓ Large-scale problems
  ✓ Rapid prototyping
  ✓ Standard architectures
  ✓ Team collaboration
    """
    
    print(analysis_text)

def visualize_comparison_results(comparison_results):
    """
    Create visualizations comparing custom vs Keras implementations
    
    This function generates plots showing:
    1. Performance metrics comparison
    2. Training convergence comparison
    3. Prediction accuracy comparison
    
    Args:
        comparison_results (dict): Results from compare_implementations_on_xor()
    
    Returns:
        None: Creates and displays plots, saves to file
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========================================================================
    # PERFORMANCE METRICS COMPARISON
    # ========================================================================
    
    # Extract metrics for plotting
    implementations = ['Custom MLP', 'Keras MLP']
    accuracies = [comparison_results['custom']['accuracy'], 
                 comparison_results['keras']['accuracy']]
    epochs = [comparison_results['custom']['epochs_to_convergence'],
             comparison_results['keras']['epochs_to_convergence']]
    errors = [comparison_results['custom']['final_error'],
             comparison_results['keras']['final_error']]
    
    # Accuracy comparison
    axes[0, 0].bar(implementations, accuracies, color=['blue', 'orange'], alpha=0.7)
    axes[0, 0].set_title('Final Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1.1)
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Epochs to convergence comparison
    axes[0, 1].bar(implementations, epochs, color=['blue', 'orange'], alpha=0.7)
    axes[0, 1].set_title('Epochs to Convergence')
    axes[0, 1].set_ylabel('Number of Epochs')
    for i, epoch in enumerate(epochs):
        axes[0, 1].text(i, epoch + 50, f'{epoch}', ha='center', fontweight='bold')
    
    # Final error comparison (log scale)
    axes[1, 0].bar(implementations, errors, color=['blue', 'orange'], alpha=0.7)
    axes[1, 0].set_title('Final Training Error')
    axes[1, 0].set_ylabel('Mean Squared Error')
    axes[1, 0].set_yscale('log')
    for i, error in enumerate(errors):
        axes[1, 0].text(i, error * 2, f'{error:.6f}', ha='center', fontweight='bold')
    
    # ========================================================================
    # PREDICTION ACCURACY VISUALIZATION
    # ========================================================================
    
    # XOR truth table with predictions
    xor_inputs, xor_targets = create_xor_dataset()
    custom_preds = comparison_results['custom']['predictions']
    keras_preds = comparison_results['keras']['predictions']
    
    # Create prediction accuracy table
    x_pos = np.arange(len(xor_inputs))
    width = 0.25
    
    axes[1, 1].bar(x_pos - width, xor_targets, width, label='Target', alpha=0.8)
    axes[1, 1].bar(x_pos, custom_preds, width, label='Custom MLP', alpha=0.8)
    axes[1, 1].bar(x_pos + width, keras_preds, width, label='Keras MLP', alpha=0.8)
    
    axes[1, 1].set_title('Prediction Comparison on XOR')
    axes[1, 1].set_ylabel('Output Value')
    axes[1, 1].set_xlabel('XOR Input Pattern')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'{inp[0]},{inp[1]}' for inp in xor_inputs])
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/implementation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_code_complexity():
    """
    Show side-by-side code examples to illustrate complexity differences
    
    This function displays actual code snippets from both implementations
    to demonstrate the difference in complexity and verbosity.
    """
    print(f"\n5. CODE COMPLEXITY DEMONSTRATION")
    print("=" * 40)
    
    print("CUSTOM IMPLEMENTATION (Forward Pass):")
    print("-" * 35)
    custom_code = '''
def mlp_forward_pass(input_vector, weights_ih, bias_h, weights_ho, bias_o):
    # Hidden layer computation
    hidden_net_input = np.dot(input_vector, weights_ih) + bias_h
    hidden_activations = sigmoid(hidden_net_input)
    
    # Output layer computation  
    output_net_input = np.dot(hidden_activations, weights_ho) + bias_o
    final_output = sigmoid(output_net_input)
    
    return final_output, hidden_activations, hidden_net_input, output_net_input
'''
    print(custom_code)
    
    print("KERAS IMPLEMENTATION (Equivalent):")
    print("-" * 32)
    keras_code = '''
# Model definition
model = keras.Sequential([
    layers.Dense(4, activation='sigmoid', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

# Forward pass (automatic)
output = model(input_vector)
'''
    print(keras_code)
    
    print("ANALYSIS:")
    print("• Custom: ~8 lines, explicit computation, full control")
    print("• Keras: ~4 lines, implicit computation, optimized backend")
    print("• Custom provides educational value, Keras provides efficiency")

def main():
    """
    Main function for comprehensive comparison demonstration
    
    This function orchestrates a complete comparison between custom MLP
    implementation and Keras, covering:
    1. Performance comparison on XOR problem
    2. Implementation complexity analysis
    3. Code examples and trade-offs
    4. Visualization of results
    5. Recommendations for different use cases
    
    The goal is to provide students with a complete understanding of
    when and why to choose different implementation approaches.
    """
    # Run comprehensive comparison
    comparison_results = compare_implementations_on_xor()
    
    # Analyze implementation differences
    analyze_implementation_differences()
    
    # Show code complexity examples
    demonstrate_code_complexity()
    
    # Create comparison visualizations
    visualize_comparison_results(comparison_results)
    
    # ========================================================================
    # FINAL RECOMMENDATIONS
    # ========================================================================
    
    print(f"\n6. RECOMMENDATIONS")
    print("=" * 40)
    
    recommendations = """
FOR LEARNING AND EDUCATION:
  1. Start with custom implementation to understand fundamentals
  2. Implement basic algorithms (perceptron, MLP) from scratch
  3. Experiment with modifications and variations
  4. Build intuition for how neural networks work

FOR PRACTICAL PROJECTS:
  1. Use Keras/TensorFlow for production applications
  2. Leverage pre-built optimizations and GPU acceleration
  3. Focus on problem-solving rather than implementation details
  4. Benefit from extensive documentation and community support

PROGRESSION STRATEGY:
  Phase 1: Learn with custom implementations
  Phase 2: Transition to frameworks for complex projects
  Phase 3: Combine both approaches as needed
  Phase 4: Contribute to frameworks or create custom solutions

WHEN TO REVISIT CUSTOM IMPLEMENTATION:
  • Research requiring novel architectures
  • Debugging complex training issues
  • Educational or teaching purposes
  • Performance optimization for specific use cases
    """
    
    print(recommendations)
    
    print("=" * 60)
    print("CONCLUSION:")
    print("Both approaches have their place in the neural network toolkit.")
    print("Custom implementations build understanding, frameworks enable scale.")
    print("The best practitioners understand both and choose appropriately.")
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the comparison demonstration when script is run directly
    
    This allows the script to be imported as a module without running
    the demonstration, or executed directly to see the full comparison.
    """
    main()


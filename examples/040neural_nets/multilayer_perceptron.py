#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from activation_functions import sigmoid, sigmoid_derivative

# ============================================================================
# MULTI-LAYER PERCEPTRON (MLP) IMPLEMENTATION - NO CLASSES, FUNCTIONS ONLY
# ============================================================================

def initialize_mlp_weights(input_size, hidden_size, output_size):
    """
    Initialize weights and biases for a Multi-Layer Perceptron with one hidden layer
    
    Proper weight initialization is crucial for successful training:
    - Random initialization breaks symmetry between neurons
    - Xavier/Glorot initialization helps with gradient flow
    - Small values prevent saturation of activation functions
    
    Network architecture:
    Input Layer (input_size) → Hidden Layer (hidden_size) → Output Layer (output_size)
    
    Args:
        input_size (int): Number of input features (e.g., 2 for XOR problem)
        hidden_size (int): Number of neurons in hidden layer (e.g., 4 for XOR)
        output_size (int): Number of output neurons (typically 1 for binary classification)
    
    Returns:
        tuple: (weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output)
            weights_input_to_hidden (numpy.ndarray): Shape (input_size, hidden_size)
                Each column represents weights connecting all inputs to one hidden neuron
            bias_hidden (numpy.ndarray): Shape (hidden_size,)
                Bias term for each hidden neuron
            weights_hidden_to_output (numpy.ndarray): Shape (hidden_size, output_size)
                Each column represents weights connecting all hidden neurons to one output
            bias_output (numpy.ndarray): Shape (output_size,)
                Bias term for each output neuron
    
    Note:
        We use Xavier initialization: weights ~ Uniform(-sqrt(6/fan_in), sqrt(6/fan_in))
        This helps maintain similar variance of activations across layers.
    """
    # Initialize weights from input to hidden layer
    # Xavier initialization: scale by sqrt(2/fan_in) for better gradient flow
    # Shape: (input_size, hidden_size) - each column is weights for one hidden neuron
    weights_input_to_hidden = np.random.uniform(
        -1, 1, (input_size, hidden_size)) * np.sqrt(2.0 / input_size)
    
    # Initialize hidden layer biases to zero (common practice)
    # Shape: (hidden_size,) - one bias per hidden neuron
    bias_hidden = np.zeros(hidden_size)
    
    # Initialize weights from hidden to output layer
    # Shape: (hidden_size, output_size) - each column is weights for one output neuron
    weights_hidden_to_output = np.random.uniform(
        -1, 1, (hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
    
    # Initialize output layer biases to zero
    # Shape: (output_size,) - one bias per output neuron
    bias_output = np.zeros(output_size)
    
    return weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output

def mlp_forward_pass(input_vector, weights_input_to_hidden, bias_hidden, 
                     weights_hidden_to_output, bias_output):
    """
    Perform forward propagation through the Multi-Layer Perceptron
    
    Forward pass computes the network output by propagating input through all layers:
    1. Input → Hidden: Apply linear transformation + sigmoid activation
    2. Hidden → Output: Apply linear transformation + sigmoid activation
    
    Mathematical formulation:
    Hidden layer: h = sigmoid(X · W₁ + b₁)
    Output layer: y = sigmoid(h · W₂ + b₂)
    
    Where:
    - X: input vector
    - W₁, b₁: weights and bias for hidden layer
    - W₂, b₂: weights and bias for output layer
    - h: hidden layer activations
    - y: final output
    
    Args:
        input_vector (numpy.ndarray): Input features, shape (input_size,)
        weights_input_to_hidden (numpy.ndarray): Input-to-hidden weights, shape (input_size, hidden_size)
        bias_hidden (numpy.ndarray): Hidden layer biases, shape (hidden_size,)
        weights_hidden_to_output (numpy.ndarray): Hidden-to-output weights, shape (hidden_size, output_size)
        bias_output (numpy.ndarray): Output layer biases, shape (output_size,)
    
    Returns:
        tuple: (final_output, hidden_activations, hidden_net_input, output_net_input)
            final_output (numpy.ndarray): Network output after sigmoid, shape (output_size,)
            hidden_activations (numpy.ndarray): Hidden layer outputs after sigmoid, shape (hidden_size,)
            hidden_net_input (numpy.ndarray): Hidden layer inputs before sigmoid, shape (hidden_size,)
            output_net_input (numpy.ndarray): Output layer inputs before sigmoid, shape (output_size,)
    
    Note:
        We return intermediate values (hidden_activations, hidden_net_input, output_net_input)
        because they are needed for backpropagation. This avoids recomputing them.
    """
    # ========================================================================
    # HIDDEN LAYER COMPUTATION
    # ========================================================================
    
    # Compute net input to hidden layer: weighted sum of inputs plus bias
    # Matrix multiplication: input_vector (1 x input_size) · weights (input_size x hidden_size)
    # Result: (1 x hidden_size) vector of net inputs to each hidden neuron
    hidden_net_input = np.dot(input_vector, weights_input_to_hidden) + bias_hidden
    
    # Apply sigmoid activation function to get hidden layer outputs
    # This introduces non-linearity, allowing the network to learn complex patterns
    # Each hidden neuron output is in range (0, 1)
    hidden_activations = sigmoid(hidden_net_input)
    
    # ========================================================================
    # OUTPUT LAYER COMPUTATION
    # ========================================================================
    
    # Compute net input to output layer: weighted sum of hidden activations plus bias
    # Matrix multiplication: hidden_activations (1 x hidden_size) · weights (hidden_size x output_size)
    # Result: (1 x output_size) vector of net inputs to each output neuron
    output_net_input = np.dot(hidden_activations, weights_hidden_to_output) + bias_output
    
    # Apply sigmoid activation to get final network output
    # For binary classification, this gives probability-like output in (0, 1)
    final_output = sigmoid(output_net_input)
    
    return final_output, hidden_activations, hidden_net_input, output_net_input

def mlp_backward_pass(input_vector, target_output, network_output, hidden_activations, 
                      hidden_net_input, output_net_input, weights_input_to_hidden, 
                      weights_hidden_to_output):
    """
    Perform backward propagation (backpropagation) to compute gradients
    
    Backpropagation computes gradients of the loss function with respect to all
    network parameters by applying the chain rule of calculus. The algorithm
    propagates error signals backward through the network.
    
    Key steps:
    1. Compute output layer error: δ₂ = (target - output) · σ'(net_output)
    2. Compute hidden layer error: δ₁ = δ₂ · W₂ᵀ · σ'(net_hidden)
    3. Compute weight gradients: ∇W = activations · δ
    4. Compute bias gradients: ∇b = δ
    
    Mathematical derivation:
    Loss = ½(target - output)²
    ∂Loss/∂W₂ = ∂Loss/∂output · ∂output/∂net_output · ∂net_output/∂W₂
    ∂Loss/∂W₁ = ∂Loss/∂output · ∂output/∂net_output · ∂net_output/∂hidden · ∂hidden/∂net_hidden · ∂net_hidden/∂W₁
    
    Args:
        input_vector (numpy.ndarray): Original input to the network, shape (input_size,)
        target_output (float or numpy.ndarray): Desired output, shape (output_size,)
        network_output (numpy.ndarray): Actual network output, shape (output_size,)
        hidden_activations (numpy.ndarray): Hidden layer outputs, shape (hidden_size,)
        hidden_net_input (numpy.ndarray): Hidden layer net inputs, shape (hidden_size,)
        output_net_input (numpy.ndarray): Output layer net inputs, shape (output_size,)
        weights_input_to_hidden (numpy.ndarray): Current input-to-hidden weights
        weights_hidden_to_output (numpy.ndarray): Current hidden-to-output weights
    
    Returns:
        tuple: (grad_weights_input_hidden, grad_bias_hidden, grad_weights_hidden_output, grad_bias_output)
            grad_weights_input_hidden (numpy.ndarray): Gradients for input-to-hidden weights
            grad_bias_hidden (numpy.ndarray): Gradients for hidden layer biases
            grad_weights_hidden_output (numpy.ndarray): Gradients for hidden-to-output weights
            grad_bias_output (numpy.ndarray): Gradients for output layer biases
    
    Note:
        All gradients have the same shape as their corresponding parameters.
        Positive gradients indicate the direction to increase the parameter to reduce loss.
    """
    # ========================================================================
    # OUTPUT LAYER ERROR COMPUTATION
    # ========================================================================
    
    # Compute output layer error signal
    # Error = (target - actual) gives the direction and magnitude of correction needed
    output_error = target_output - network_output
    
    # Apply chain rule: multiply error by derivative of activation function
    # This gives the error signal that needs to be propagated backward
    # δ₂ = error · σ'(net_output) where σ' is sigmoid derivative
    output_delta = output_error * sigmoid_derivative(output_net_input)
    
    # ========================================================================
    # OUTPUT LAYER GRADIENT COMPUTATION
    # ========================================================================
    
    # Compute gradients for hidden-to-output weights
    # ∇W₂ = hidden_activations^T · δ₂
    # Each gradient tells us how much to change each weight to reduce error
    # Shape: (hidden_size, output_size) - same as weights_hidden_to_output
    grad_weights_hidden_output = np.outer(hidden_activations, output_delta)
    
    # Compute gradients for output layer biases
    # ∇b₂ = δ₂ (bias gradient equals the error signal)
    # Shape: (output_size,) - same as bias_output
    grad_bias_output = output_delta
    
    # ========================================================================
    # HIDDEN LAYER ERROR COMPUTATION
    # ========================================================================
    
    # Propagate error from output layer back to hidden layer
    # This is the key insight of backpropagation: errors flow backward through weights
    # hidden_error = δ₂ · W₂^T (transpose of weights matrix)
    # Shape: (hidden_size,) - one error signal per hidden neuron
    hidden_error = np.dot(output_delta, weights_hidden_to_output.T)
    
    # Apply chain rule: multiply by derivative of hidden layer activation
    # δ₁ = hidden_error · σ'(net_hidden)
    # This gives the error signal for the hidden layer
    hidden_delta = hidden_error * sigmoid_derivative(hidden_net_input)
    
    # ========================================================================
    # HIDDEN LAYER GRADIENT COMPUTATION
    # ========================================================================
    
    # Compute gradients for input-to-hidden weights
    # ∇W₁ = input_vector^T · δ₁
    # Shape: (input_size, hidden_size) - same as weights_input_to_hidden
    grad_weights_input_hidden = np.outer(input_vector, hidden_delta)
    
    # Compute gradients for hidden layer biases
    # ∇b₁ = δ₁
    # Shape: (hidden_size,) - same as bias_hidden
    grad_bias_hidden = hidden_delta
    
    return (grad_weights_input_hidden, grad_bias_hidden, 
            grad_weights_hidden_output, grad_bias_output)

def train_mlp(training_inputs, training_targets, hidden_layer_size=4, 
              learning_rate=1.0, max_epochs=10000, convergence_tolerance=1e-6):
    """
    Train the Multi-Layer Perceptron using backpropagation algorithm
    
    This function implements the complete training loop:
    1. Initialize network weights randomly
    2. For each epoch:
       a. For each training sample:
          - Forward pass: compute output
          - Compute error
          - Backward pass: compute gradients
          - Update weights using gradients
    3. Check for convergence
    4. Return trained network
    
    Args:
        training_inputs (numpy.ndarray): Training data, shape (num_samples, num_features)
        training_targets (numpy.ndarray): Target outputs, shape (num_samples,)
        hidden_layer_size (int, optional): Number of hidden neurons. Default 4.
        learning_rate (float, optional): Step size for weight updates. Default 1.0.
        max_epochs (int, optional): Maximum training iterations. Default 10000.
        convergence_tolerance (float, optional): Stop when error below this. Default 1e-6.
    
    Returns:
        tuple: (weights_ih, bias_h, weights_ho, bias_o, error_history)
            weights_ih: Trained input-to-hidden weights
            bias_h: Trained hidden layer biases
            weights_ho: Trained hidden-to-output weights
            bias_o: Trained output layer biases
            error_history: List of total errors per epoch (for plotting)
    
    Note:
        Learning rate is crucial: too high causes oscillation, too low causes slow learning.
        Hidden layer size affects capacity: too few neurons = underfitting, too many = overfitting.
    """
    # Get dataset dimensions
    num_samples, num_features = training_inputs.shape
    input_layer_size = num_features
    output_layer_size = 1  # Binary classification
    
    # Initialize network weights and biases
    weights_ih, bias_h, weights_ho, bias_o = initialize_mlp_weights(
        input_layer_size, hidden_layer_size, output_layer_size)
    
    # Track training progress
    error_history = []
    
    # Print training configuration
    print(f"MLP Training Configuration:")
    print(f"  Input layer size: {input_layer_size}")
    print(f"  Hidden layer size: {hidden_layer_size}")
    print(f"  Output layer size: {output_layer_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Convergence tolerance: {convergence_tolerance}")
    print("-" * 50)
    
    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    
    for epoch in range(max_epochs):
        total_epoch_error = 0
        
        # Process each training sample
        for sample_idx in range(num_samples):
            # Get current training sample and target
            current_input = training_inputs[sample_idx]
            current_target = training_targets[sample_idx]
            
            # ================================================================
            # FORWARD PASS
            # ================================================================
            
            # Compute network output for current input
            network_output, hidden_activations, hidden_net_input, output_net_input = \
                mlp_forward_pass(current_input, weights_ih, bias_h, weights_ho, bias_o)
            
            # Compute squared error for this sample
            # Using Mean Squared Error: ½(target - output)²
            sample_error = 0.5 * (current_target - network_output[0]) ** 2
            total_epoch_error += sample_error
            
            # ================================================================
            # BACKWARD PASS
            # ================================================================
            
            # Compute gradients using backpropagation
            grad_w_ih, grad_b_h, grad_w_ho, grad_b_o = mlp_backward_pass(
                current_input, current_target, network_output, hidden_activations,
                hidden_net_input, output_net_input, weights_ih, weights_ho)
            
            # ================================================================
            # WEIGHT UPDATES
            # ================================================================
            
            # Update all weights and biases using computed gradients
            # Gradient ascent: move in direction that reduces error
            weights_ih += learning_rate * grad_w_ih
            bias_h += learning_rate * grad_b_h
            weights_ho += learning_rate * grad_w_ho
            bias_o += learning_rate * grad_b_o
        
        # Record total error for this epoch
        error_history.append(total_epoch_error)
        
        # Print progress every 1000 epochs or when converged
        if epoch % 1000 == 0 or total_epoch_error < convergence_tolerance:
            print(f"Epoch {epoch+1}: Total error = {total_epoch_error:.6f}")
        
        # Check for convergence
        if total_epoch_error < convergence_tolerance:
            print(f"Convergence achieved in {epoch+1} epochs!")
            break
    
    return weights_ih, bias_h, weights_ho, bias_o, error_history

def test_mlp(test_inputs, test_targets, weights_ih, bias_h, weights_ho, bias_o):
    """
    Test the trained MLP on a dataset and compute performance metrics
    
    This function evaluates the trained network by:
    1. Computing predictions for all test samples
    2. Converting continuous outputs to binary predictions
    3. Computing accuracy and detailed results
    
    Args:
        test_inputs (numpy.ndarray): Test data, shape (num_samples, num_features)
        test_targets (numpy.ndarray): True labels, shape (num_samples,)
        weights_ih (numpy.ndarray): Trained input-to-hidden weights
        bias_h (numpy.ndarray): Trained hidden biases
        weights_ho (numpy.ndarray): Trained hidden-to-output weights
        bias_o (numpy.ndarray): Trained output biases
    
    Returns:
        tuple: (accuracy, predictions)
            accuracy (float): Fraction of correct predictions (0.0 to 1.0)
            predictions (list): Binary predictions for each test sample
    
    Note:
        We use 0.5 as the threshold for converting continuous outputs to binary.
        This is appropriate for sigmoid outputs trained on binary targets.
    """
    num_test_samples = len(test_inputs)
    predictions = []
    correct_count = 0
    
    print("=== MLP TEST RESULTS ===")
    
    # Test each sample
    for sample_idx in range(num_test_samples):
        # Get current test sample and true target
        current_input = test_inputs[sample_idx]
        true_target = test_targets[sample_idx]
        
        # Compute network output
        network_output, _, _, _ = mlp_forward_pass(
            current_input, weights_ih, bias_h, weights_ho, bias_o)
        
        # Convert continuous output to binary prediction
        # Threshold at 0.5: output >= 0.5 → prediction = 1, else prediction = 0
        binary_prediction = 1 if network_output[0] >= 0.5 else 0
        predictions.append(binary_prediction)
        
        # Check correctness
        is_correct = (binary_prediction == true_target)
        if is_correct:
            correct_count += 1
        
        # Print detailed results for analysis
        print(f"Input: {current_input} → Output: {network_output[0]:.4f}, "
              f"Prediction: {binary_prediction}, Target: {true_target}, "
              f"Correct: {is_correct}")
    
    # Calculate and display accuracy
    accuracy = correct_count / num_test_samples
    print(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{num_test_samples})")
    
    return accuracy, predictions

def create_xor_dataset():
    """
    Create the XOR dataset for testing MLP capabilities
    
    XOR is the classic test case for multi-layer networks because:
    1. It's not linearly separable (single perceptron fails)
    2. It's simple enough to understand and visualize
    3. It demonstrates the power of hidden layers
    
    Returns:
        tuple: (inputs, targets)
            inputs (numpy.ndarray): XOR input patterns, shape (4, 2)
            targets (numpy.ndarray): XOR target outputs, shape (4,)
    """
    # XOR input patterns: all combinations of 2 binary inputs
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]], dtype=np.float32)
    
    # XOR outputs: 1 when inputs differ, 0 when inputs are same
    targets = np.array([0, 1, 1, 0], dtype=np.float32)
    
    return inputs, targets

def visualize_mlp_decision_surface(weights_ih, bias_h, weights_ho, bias_o):
    """
    Visualize the decision surface learned by the MLP
    
    This function creates a comprehensive visualization showing:
    1. The non-linear decision boundary learned by the MLP
    2. The XOR data points colored by their true class
    3. The continuous output surface as a color map
    4. The decision boundary (where output = 0.5)
    
    This visualization demonstrates how MLPs can learn complex, non-linear
    decision boundaries that are impossible for single-layer perceptrons.
    
    Args:
        weights_ih (numpy.ndarray): Trained input-to-hidden weights
        bias_h (numpy.ndarray): Trained hidden biases
        weights_ho (numpy.ndarray): Trained hidden-to-output weights
        bias_o (numpy.ndarray): Trained output biases
    
    Returns:
        None: Creates and displays plot, saves to file
    
    Note:
        The visualization uses a grid of points to show the decision surface.
        Each point is colored based on the MLP's output value.
    """
    # Create a fine grid of points to evaluate the decision surface
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    grid_resolution = 100
    
    # Create coordinate matrices for the grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    
    # Flatten grid to create list of points to evaluate
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Evaluate MLP output at each grid point
    grid_outputs = []
    for point in grid_points:
        output, _, _, _ = mlp_forward_pass(point, weights_ih, bias_h, weights_ho, bias_o)
        grid_outputs.append(output[0])
    
    # Reshape outputs back to grid shape for plotting
    output_surface = np.array(grid_outputs).reshape(xx.shape)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the decision surface as a color map
    # Blue regions: MLP outputs close to 0 (predicts class 0)
    # Red regions: MLP outputs close to 1 (predicts class 1)
    contour_plot = plt.contourf(xx, yy, output_surface, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(contour_plot, label='MLP Output')
    
    # Draw the decision boundary (where output = 0.5)
    # This line separates regions where MLP predicts class 0 vs class 1
    plt.contour(xx, yy, output_surface, levels=[0.5], colors='black', linewidths=2, linestyles='--', label='Decision Boundary')
    
    # Plot the XOR data points
    xor_inputs, xor_targets = create_xor_dataset()
    colors = ['red', 'blue']
    class_labels = ['Class 0', 'Class 1']
    
    for sample_idx in range(len(xor_inputs)):
        current_class = int(xor_targets[sample_idx])
        plt.scatter(xor_inputs[sample_idx, 0], xor_inputs[sample_idx, 1], 
                   c=colors[current_class], s=200, edgecolors='black', linewidth=2,
                   label=class_labels[current_class] if sample_idx == 0 or 
                   (sample_idx == 1 and xor_targets[sample_idx] != xor_targets[0]) else "")
        
        # Add labels showing input and output
        plt.annotate(f'({xor_inputs[sample_idx, 0]:.0f},{xor_inputs[sample_idx, 1]:.0f})→{xor_targets[sample_idx]:.0f}', 
                    (xor_inputs[sample_idx, 0], xor_inputs[sample_idx, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=12, fontweight='bold')
    
    # Set plot properties
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('MLP Decision Surface for XOR Problem')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('mlp_decision_surface.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_decision_surface.svg')
    plt.show()

def main():
    """
    Main function demonstrating MLP solving the XOR problem
    
    This comprehensive demonstration shows:
    1. Creating the XOR dataset
    2. Training an MLP to solve XOR
    3. Testing the trained network
    4. Visualizing the learned decision surface
    5. Plotting the training error curve
    6. Comparing with perceptron limitations
    
    The goal is to show how MLPs overcome the fundamental limitations
    of single-layer perceptrons through hidden layers and non-linear activations.
    """
    print("=" * 60)
    print("MULTI-LAYER PERCEPTRON - SOLVING THE XOR PROBLEM")
    print("=" * 60)
    
    # ========================================================================
    # DATASET PREPARATION
    # ========================================================================
    
    # Create XOR dataset
    xor_inputs, xor_targets = create_xor_dataset()
    
    print("XOR Dataset:")
    for i in range(len(xor_inputs)):
        print(f"  {xor_inputs[i]} → {xor_targets[i]}")
    
    # ========================================================================
    # NETWORK TRAINING
    # ========================================================================
    
    print(f"\nTraining MLP on XOR problem...")
    
    # Train the MLP with appropriate hyperparameters for XOR
    trained_weights_ih, trained_bias_h, trained_weights_ho, trained_bias_o, training_errors = \
        train_mlp(xor_inputs, xor_targets, 
                 hidden_layer_size=4,    # 4 hidden neurons sufficient for XOR
                 learning_rate=10.0,     # Higher learning rate for faster convergence
                 max_epochs=10000)       # Enough epochs to ensure convergence
    
    # ========================================================================
    # NETWORK TESTING
    # ========================================================================
    
    print(f"\nTesting trained MLP...")
    
    # Test the trained network on the same data (XOR is small enough)
    test_accuracy, test_predictions = test_mlp(
        xor_inputs, xor_targets, trained_weights_ih, trained_bias_h, 
        trained_weights_ho, trained_bias_o)
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print(f"\nGenerating visualizations...")
    
    # Visualize the decision surface
    visualize_mlp_decision_surface(trained_weights_ih, trained_bias_h, 
                                  trained_weights_ho, trained_bias_o)
    
    # Plot training error curve
    plt.figure(figsize=(10, 6))
    plt.plot(training_errors, 'b-', linewidth=2)
    plt.title('MLP Training Error - XOR Problem')
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')
    plt.yscale('log')  # Log scale to show exponential decay
    plt.grid(True, alpha=0.3)
    plt.savefig('mlp_training_error.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlp_training_error.svg')
    plt.show()
    
    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================
    
    print(f"\n" + "=" * 60)
    print("RESULTS ANALYSIS:")
    print("=" * 60)
    
    if test_accuracy >= 0.99:
        print("✓ SUCCESS: MLP successfully solved the XOR problem!")
        print("  The multi-layer architecture overcame the linear separability limitation.")
        print("\nKey factors for success:")
        print("  1. Hidden layer creates intermediate representations")
        print("  2. Non-linear activation functions enable complex boundaries")
        print("  3. Backpropagation enables training of multi-layer networks")
        
    else:
        print("? Unexpected result - check hyperparameters and implementation")
    
    print(f"\nFinal Performance:")
    print(f"  Test Accuracy: {test_accuracy:.2%}")
    print(f"  Training Epochs: {len(training_errors)}")
    print(f"  Final Error: {training_errors[-1]:.6f}")
    
    print("\nComparison with Simple Perceptron:")
    print("  Simple Perceptron on XOR: ~50% accuracy (random guessing)")
    print("  Multi-Layer Perceptron on XOR: 100% accuracy (perfect solution)")
    
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the MLP demonstration when script is run directly
    
    This allows the script to be imported as a module without running
    the demonstration, or executed directly to see the full demo.
    """
    main()


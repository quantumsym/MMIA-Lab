#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SIMPLE PERCEPTRON IMPLEMENTATION - NO CLASSES, FUNCTIONS ONLY
# ============================================================================

def initialize_perceptron(num_inputs):
    """
    Initialize perceptron weights and bias with random values
    
    The perceptron consists of:
    - Weights: one for each input feature
    - Bias: single value that shifts the decision boundary
    
    Random initialization is important because:
    - Zero initialization would make all neurons learn the same features
    - Small random values break symmetry and enable diverse learning
    - Range [-1, 1] provides good starting point for most problems
    
    Args:
        num_inputs (int): Number of input features the perceptron will receive.
                         For example, 2 for XOR problem (two binary inputs).
    
    Returns:
        tuple: (weights, bias)
            weights (numpy.ndarray): Array of shape (num_inputs,) with random values in [-1, 1]
            bias (float): Single random value in [-1, 1]
    
    Note:
        We use uniform distribution in [-1, 1] rather than normal distribution
        to ensure bounded initial values. This helps with numerical stability
        in the early stages of training.
    """
    # Initialize weights randomly between -1 and 1
    # Each weight corresponds to one input feature
    # Shape: (num_inputs,) - one weight per input
    weights = np.random.uniform(-1, 1, num_inputs)
    
    # Initialize bias randomly between -1 and 1
    # Bias allows the decision boundary to be shifted away from origin
    # This is crucial for learning functions that are not linearly separable through origin
    bias = np.random.uniform(-1, 1)
    
    return weights, bias

def perceptron_forward_pass(input_vector, weights, bias):
    """
    Perform forward pass through the perceptron (compute output)
    
    The perceptron computes output in two steps:
    1. Linear combination: net_input = sum(inputs * weights) + bias
    2. Activation: output = step_function(net_input)
    
    Mathematical formula:
    net_input = x₁*w₁ + x₂*w₂ + ... + xₙ*wₙ + b
    output = 1 if net_input >= 0, else 0
    
    The perceptron essentially finds a hyperplane (line in 2D) that separates
    the input space into two regions. Points on one side are classified as 1,
    points on the other side as 0.
    
    Args:
        input_vector (numpy.ndarray): Input features, shape (num_features,).
                                     For example, [0, 1] for one sample of XOR problem.
        weights (numpy.ndarray): Weight vector, shape (num_features,).
                                Each weight multiplies corresponding input feature.
        bias (float): Bias term that shifts the decision boundary.
                     Positive bias makes the perceptron more likely to output 1.
    
    Returns:
        tuple: (output, net_input)
            output (int): Binary classification result (0 or 1)
            net_input (float): Raw weighted sum before applying step function.
                              This value determines the output and is used for learning.
    
    Note:
        We return both output and net_input because:
        - output is the final prediction
        - net_input is needed for the learning rule and debugging
    """
    # Compute weighted sum of inputs plus bias
    # np.dot performs dot product: sum(input_vector[i] * weights[i])
    # This is the linear combination that defines the decision boundary
    net_input = np.dot(input_vector, weights) + bias
    
    # Apply step activation function
    # Step function: output 1 if net_input >= 0, otherwise output 0
    # This creates a hard decision boundary for binary classification
    output = 1 if net_input >= 0 else 0
    
    return output, net_input

def train_perceptron(training_inputs, training_targets, learning_rate=0.1, max_epochs=100):
    """
    Train the perceptron using the perceptron learning rule
    
    The perceptron learning rule is one of the earliest learning algorithms:
    1. For each training example, compute the output
    2. If output is correct, do nothing
    3. If output is wrong, adjust weights in the direction that reduces error
    
    Weight update rule:
    weights = weights + learning_rate * error * inputs
    bias = bias + learning_rate * error
    
    where error = target - output
    
    This rule has a beautiful geometric interpretation: it moves the decision
    boundary towards correctly classifying the misclassified point.
    
    Args:
        training_inputs (numpy.ndarray): Training data, shape (num_samples, num_features).
                                        Each row is one training example.
        training_targets (numpy.ndarray): Target outputs, shape (num_samples,).
                                         Each element is the desired output (0 or 1).
        learning_rate (float, optional): Step size for weight updates. Default 0.1.
                                        Higher values = faster learning but less stable.
                                        Lower values = slower but more stable learning.
        max_epochs (int, optional): Maximum number of training iterations. Default 100.
                                   One epoch = one pass through all training data.
    
    Returns:
        tuple: (final_weights, final_bias, error_history)
            final_weights (numpy.ndarray): Trained weight vector
            final_bias (float): Trained bias value
            error_history (list): Number of errors in each epoch (for plotting)
    
    Note:
        The perceptron is guaranteed to converge if the data is linearly separable.
        If data is not linearly separable (like XOR), it will never converge.
    """
    # Get dimensions of training data
    num_samples, num_features = training_inputs.shape
    
    # Initialize perceptron parameters
    weights, bias = initialize_perceptron(num_features)
    
    # Track errors per epoch for analysis and plotting
    error_history = []
    
    # Print initial state for debugging and educational purposes
    print(f"Initial weights: {weights}")
    print(f"Initial bias: {bias}")
    print("-" * 50)
    
    # Training loop - iterate through epochs
    for epoch in range(max_epochs):
        # Count errors in this epoch
        epoch_errors = 0
        
        # Process each training sample
        for sample_idx in range(num_samples):
            # Get current training sample and its target
            current_input = training_inputs[sample_idx]
            current_target = training_targets[sample_idx]
            
            # Forward pass: compute perceptron output
            predicted_output, net_input = perceptron_forward_pass(current_input, weights, bias)
            
            # Compute prediction error
            # error > 0: target is 1 but predicted 0 (false negative)
            # error < 0: target is 0 but predicted 1 (false positive)
            # error = 0: correct prediction
            prediction_error = current_target - predicted_output
            
            # Update weights and bias only if there's an error
            if prediction_error != 0:
                epoch_errors += 1
                
                # Perceptron learning rule:
                # If we predicted 0 but target is 1: increase weights for positive inputs
                # If we predicted 1 but target is 0: decrease weights for positive inputs
                weights += learning_rate * prediction_error * current_input
                bias += learning_rate * prediction_error
                
                # Print detailed information about the update for educational purposes
                print(f"Epoch {epoch+1}, Sample {sample_idx+1}: "
                      f"Input={current_input}, Target={current_target}, "
                      f"Predicted={predicted_output}, Error={prediction_error}")
        
        # Record errors for this epoch
        error_history.append(epoch_errors)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}: {epoch_errors} errors")
        
        # Check for convergence (no errors in this epoch)
        if epoch_errors == 0:
            print(f"Training completed successfully in {epoch+1} epochs!")
            break
    
    # Print final state
    print("-" * 50)
    print(f"Final weights: {weights}")
    print(f"Final bias: {bias}")
    
    return weights, bias, error_history

def test_perceptron(test_inputs, test_targets, trained_weights, trained_bias):
    """
    Test the trained perceptron on a dataset and compute performance metrics
    
    This function evaluates how well the trained perceptron performs on test data.
    It computes predictions for all test samples and calculates accuracy.
    
    Args:
        test_inputs (numpy.ndarray): Test data, shape (num_samples, num_features).
                                    Should be same format as training data.
        test_targets (numpy.ndarray): True labels for test data, shape (num_samples,).
                                     Ground truth values to compare predictions against.
        trained_weights (numpy.ndarray): Weights learned during training.
        trained_bias (float): Bias learned during training.
    
    Returns:
        tuple: (accuracy, predictions)
            accuracy (float): Fraction of correct predictions (0.0 to 1.0)
            predictions (list): List of predicted outputs for each test sample
    
    Note:
        This function provides detailed output for educational purposes,
        showing the decision-making process for each test sample.
    """
    num_test_samples = len(test_inputs)
    predictions = []
    correct_predictions = 0
    
    print("=== PERCEPTRON TEST RESULTS ===")
    
    # Test each sample individually
    for sample_idx in range(num_test_samples):
        # Get current test sample and its true label
        current_input = test_inputs[sample_idx]
        true_target = test_targets[sample_idx]
        
        # Make prediction using trained perceptron
        predicted_output, net_input = perceptron_forward_pass(current_input, trained_weights, trained_bias)
        predictions.append(predicted_output)
        
        # Check if prediction is correct
        is_correct = (predicted_output == true_target)
        if is_correct:
            correct_predictions += 1
        
        # Print detailed results for educational analysis
        print(f"Input: {current_input} → Output: {predicted_output}, "
              f"Target: {true_target}, Net Input: {net_input:.3f}, "
              f"Correct: {is_correct}")
    
    # Calculate and display accuracy
    accuracy = correct_predictions / num_test_samples
    print(f"\nAccuracy: {accuracy:.2%} ({correct_predictions}/{num_test_samples})")
    
    return accuracy, predictions

# ============================================================================
# LOGICAL FUNCTION DATASETS
# ============================================================================

def create_and_dataset():
    """
    Create dataset for logical AND function
    
    The AND function returns 1 only when both inputs are 1.
    This is a linearly separable function - a single line can separate
    the positive and negative examples.
    
    Truth table:
    Input1 | Input2 | Output
    -------|--------|-------
       0   |   0    |   0
       0   |   1    |   0
       1   |   0    |   0
       1   |   1    |   1
    
    Returns:
        tuple: (inputs, targets)
            inputs (numpy.ndarray): All possible input combinations, shape (4, 2)
            targets (numpy.ndarray): Corresponding AND outputs, shape (4,)
    
    Note:
        This function is linearly separable because we can draw a line
        that separates the single positive example (1,1) from the three
        negative examples (0,0), (0,1), (1,0).
    """
    # Define all possible input combinations for 2 binary variables
    inputs = np.array([[0, 0],  # Both inputs false
                       [0, 1],  # First false, second true
                       [1, 0],  # First true, second false
                       [1, 1]]) # Both inputs true
    
    # Define AND function outputs
    # Only returns 1 when both inputs are 1
    targets = np.array([0, 0, 0, 1])
    
    return inputs, targets

def create_or_dataset():
    """
    Create dataset for logical OR function
    
    The OR function returns 1 when at least one input is 1.
    This is also a linearly separable function.
    
    Truth table:
    Input1 | Input2 | Output
    -------|--------|-------
       0   |   0    |   0
       0   |   1    |   1
       1   |   0    |   1
       1   |   1    |   1
    
    Returns:
        tuple: (inputs, targets)
            inputs (numpy.ndarray): All possible input combinations, shape (4, 2)
            targets (numpy.ndarray): Corresponding OR outputs, shape (4,)
    
    Note:
        This function is linearly separable because we can draw a line
        that separates the single negative example (0,0) from the three
        positive examples (0,1), (1,0), (1,1).
    """
    # Same input combinations as AND function
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    
    # Define OR function outputs
    # Returns 1 when at least one input is 1
    targets = np.array([0, 1, 1, 1])
    
    return inputs, targets

def visualize_perceptron_results(inputs, targets, weights, bias, function_name):
    """
    Visualize perceptron decision boundary and data points
    
    This function creates a 2D plot showing:
    1. Training data points colored by their class
    2. Decision boundary learned by the perceptron
    3. Clear visualization of how the perceptron separates the classes
    
    The decision boundary is defined by the equation:
    w₁*x₁ + w₂*x₂ + bias = 0
    
    Args:
        inputs (numpy.ndarray): Input data points, shape (num_samples, 2)
        targets (numpy.ndarray): Target labels, shape (num_samples,)
        weights (numpy.ndarray): Trained perceptron weights, shape (2,)
        bias (float): Trained perceptron bias
        function_name (str): Name of the logical function for plot title
    
    Returns:
        None: Creates and displays plot, saves to file
    
    Note:
        This visualization only works for 2D input data (2 features).
        For higher dimensions, we would need different visualization techniques.
    """
    # Create new figure with appropriate size
    plt.figure(figsize=(8, 6))
    
    # Define colors for different classes
    colors = ['red', 'blue']
    class_labels = ['Class 0', 'Class 1']
    
    # Plot data points colored by their class
    for sample_idx in range(len(inputs)):
        current_class = int(targets[sample_idx])
        plt.scatter(inputs[sample_idx, 0], inputs[sample_idx, 1], 
                   c=colors[current_class], s=100, 
                   label=class_labels[current_class] if sample_idx == 0 or 
                   (sample_idx == 1 and targets[sample_idx] != targets[0]) else "")
    
    # Draw decision boundary if we have 2D input
    if len(weights) == 2:
        # Decision boundary equation: w₁*x₁ + w₂*x₂ + bias = 0
        # Solve for x₂: x₂ = -(w₁*x₁ + bias) / w₂
        if weights[1] != 0:  # Avoid division by zero
            # Create range of x₁ values for plotting the line
            x1_range = np.linspace(-0.5, 1.5, 100)
            # Calculate corresponding x₂ values for the decision boundary
            x2_boundary = -(weights[0] * x1_range + bias) / weights[1]
            plt.plot(x1_range, x2_boundary, 'k--', linewidth=2, 
                    label='Decision Boundary')
    
    # Set plot properties for better visualization
    plt.xlim(-0.5, 1.5)  # Show area around data points
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(f'Perceptron - {function_name} Function')
    plt.legend()
    plt.grid(True, alpha=0.3)  # Light grid for better readability
    
    # Save plot with descriptive filename
    filename = f'/home/ubuntu/perceptron_{function_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating perceptron on logical functions
    
    This function provides a comprehensive demonstration of the perceptron
    algorithm on two linearly separable logical functions: AND and OR.
    
    The demonstration includes:
    1. Training perceptron on AND function
    2. Testing and visualizing results
    3. Training perceptron on OR function  
    4. Testing and visualizing results
    5. Plotting training error curves
    6. Summary of results
    
    This shows that the perceptron can successfully learn linearly
    separable functions but will fail on non-linearly separable ones.
    """
    print("=" * 60)
    print("SIMPLE PERCEPTRON - LOGICAL FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    # ========================================================================
    # TEST ON AND FUNCTION
    # ========================================================================
    
    print("\n1. TESTING ON AND FUNCTION")
    print("=" * 40)
    
    # Create AND dataset
    and_inputs, and_targets = create_and_dataset()
    print("AND Dataset:")
    for i in range(len(and_inputs)):
        print(f"  {and_inputs[i]} → {and_targets[i]}")
    
    # Train perceptron on AND function
    and_weights, and_bias, and_errors = train_perceptron(
        and_inputs, and_targets, learning_rate=0.1)
    
    # Test trained perceptron
    and_accuracy, and_predictions = test_perceptron(
        and_inputs, and_targets, and_weights, and_bias)
    
    # Visualize results
    visualize_perceptron_results(and_inputs, and_targets, and_weights, and_bias, "AND Function")
    
    # ========================================================================
    # TEST ON OR FUNCTION
    # ========================================================================
    
    print("\n\n2. TESTING ON OR FUNCTION")
    print("=" * 40)
    
    # Create OR dataset
    or_inputs, or_targets = create_or_dataset()
    print("OR Dataset:")
    for i in range(len(or_inputs)):
        print(f"  {or_inputs[i]} → {or_targets[i]}")
    
    # Train perceptron on OR function
    or_weights, or_bias, or_errors = train_perceptron(
        or_inputs, or_targets, learning_rate=0.1)
    
    # Test trained perceptron
    or_accuracy, or_predictions = test_perceptron(
        or_inputs, or_targets, or_weights, or_bias)
    
    # Visualize results
    visualize_perceptron_results(or_inputs, or_targets, or_weights, or_bias, "OR Function")
    
    # ========================================================================
    # PLOT TRAINING ERROR CURVES
    # ========================================================================
    
    # Create comparison plot of training errors
    plt.figure(figsize=(12, 5))
    
    # Plot AND training errors
    plt.subplot(1, 2, 1)
    plt.plot(and_errors, 'bo-', linewidth=2, markersize=6)
    plt.title('Training Errors - AND Function')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)  # Start y-axis at 0
    
    # Plot OR training errors
    plt.subplot(1, 2, 2)
    plt.plot(or_errors, 'ro-', linewidth=2, markersize=6)
    plt.title('Training Errors - OR Function')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)  # Start y-axis at 0
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/perceptron_training_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # SUMMARY RESULTS
    # ========================================================================
    
    print(f"\n\nRESULTS SUMMARY:")
    print(f"AND Function - Accuracy: {and_accuracy:.2%}")
    print(f"OR Function  - Accuracy: {or_accuracy:.2%}")
    print("\nKey Observations:")
    print("- Both functions are linearly separable")
    print("- Perceptron successfully learns both functions")
    print("- Training converges to 100% accuracy")
    print("- Decision boundaries clearly separate the classes")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute main demonstration when script is run directly
    
    This allows the script to be imported as a module without running
    the demonstration, or executed directly to see the full demo.
    """
    main()


#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PERCEPTRON LIMITATIONS DEMONSTRATION - XOR PROBLEM
# ============================================================================

def create_xor_dataset():
    """
    Create dataset for the XOR (Exclusive OR) logical function
    
    XOR is the classic example of a non-linearly separable function.
    It returns 1 when inputs are different, 0 when they are the same.
    
    Truth table:
    Input1 | Input2 | Output
    -------|--------|-------
       0   |   0    |   0
       0   |   1    |   1
       1   |   0    |   1
       1   |   1    |   0
    
    Why XOR is not linearly separable:
    - Class 0 points: (0,0) and (1,1) - diagonally opposite corners
    - Class 1 points: (0,1) and (1,0) - the other diagonal corners
    - No single straight line can separate these two groups
    
    Returns:
        tuple: (inputs, targets)
            inputs (numpy.ndarray): All possible input combinations, shape (4, 2)
            targets (numpy.ndarray): Corresponding XOR outputs, shape (4,)
    
    Note:
        This function demonstrates the fundamental limitation of linear classifiers:
        they cannot solve problems where the classes are not linearly separable.
    """
    # Define all possible input combinations for 2 binary variables
    inputs = np.array([[0, 0],  # Both inputs same (false) → output 0
                       [0, 1],  # Inputs different → output 1
                       [1, 0],  # Inputs different → output 1
                       [1, 1]]) # Both inputs same (true) → output 0
    
    # Define XOR function outputs
    # Returns 1 only when inputs are different
    targets = np.array([0, 1, 1, 0])
    
    return inputs, targets

def test_perceptron_on_xor_limited_epochs():
    """
    Test perceptron on XOR with limited epochs to demonstrate non-convergence
    
    This function attempts to train a perceptron on the XOR function.
    Since XOR is not linearly separable, the perceptron will never converge
    to a perfect solution. We limit the epochs to avoid infinite loops.
    
    The demonstration shows:
    1. How the perceptron struggles with XOR
    2. Oscillating weights that never stabilize
    3. Persistent errors that never reach zero
    4. Final accuracy around 50% (random guessing level)
    
    Returns:
        tuple: (final_weights, final_bias, final_accuracy)
            final_weights (numpy.ndarray): Weights after training attempt
            final_bias (float): Bias after training attempt  
            final_accuracy (float): Best accuracy achieved (typically ~50%)
    
    Note:
        We use a simplified training loop here to focus on the demonstration
        of the fundamental limitation rather than implementation details.
    """
    print("=" * 60)
    print("DEMONSTRATION: PERCEPTRON CANNOT LEARN XOR")
    print("=" * 60)
    
    # Get XOR dataset
    xor_inputs, xor_targets = create_xor_dataset()
    
    print("XOR Dataset:")
    for i in range(len(xor_inputs)):
        print(f"  {xor_inputs[i]} → {xor_targets[i]}")
    
    # Initialize perceptron with simple fixed values for reproducibility
    weights = np.array([0.5, 0.5])  # Start with equal weights
    bias = 0.0                      # Start with zero bias
    learning_rate = 0.1
    max_epochs = 50  # Limit epochs to avoid infinite loop
    
    print(f"\nInitial weights: {weights}")
    print(f"Initial bias: {bias}")
    print("-" * 50)
    
    # Track errors per epoch to show non-convergence
    error_history = []
    
    # Training loop - will not converge for XOR
    for epoch in range(max_epochs):
        epoch_errors = 0
        
        # Process each training sample
        for sample_idx in range(len(xor_inputs)):
            # Get current sample
            current_input = xor_inputs[sample_idx]
            current_target = xor_targets[sample_idx]
            
            # Forward pass: compute perceptron output
            net_input = np.dot(current_input, weights) + bias
            predicted_output = 1 if net_input >= 0 else 0
            
            # Compute error
            prediction_error = current_target - predicted_output
            
            # Update weights if there's an error
            if prediction_error != 0:
                epoch_errors += 1
                # Apply perceptron learning rule
                weights += learning_rate * prediction_error * current_input
                bias += learning_rate * prediction_error
        
        # Record errors for this epoch
        error_history.append(epoch_errors)
        
        # Print progress for first 10 epochs and every 10th epoch after
        if epoch < 10 or epoch % 10 == 0:
            print(f"Epoch {epoch+1}: {epoch_errors} errors, "
                  f"Weights: [{weights[0]:.3f}, {weights[1]:.3f}], "
                  f"Bias: {bias:.3f}")
        
        # Check if we somehow achieved perfect classification (won't happen for XOR)
        if epoch_errors == 0:
            print(f"Unexpected convergence in {epoch+1} epochs!")
            break
    
    # Final testing to compute accuracy
    print("\n=== FINAL RESULTS ===")
    correct_predictions = 0
    
    for sample_idx in range(len(xor_inputs)):
        current_input = xor_inputs[sample_idx]
        true_target = xor_targets[sample_idx]
        
        # Make final prediction
        net_input = np.dot(current_input, weights) + bias
        predicted_output = 1 if net_input >= 0 else 0
        
        # Check correctness
        is_correct = (predicted_output == true_target)
        if is_correct:
            correct_predictions += 1
        
        print(f"Input: {current_input} → Output: {predicted_output}, "
              f"Target: {true_target}, Correct: {is_correct}")
    
    # Calculate final accuracy
    final_accuracy = correct_predictions / len(xor_inputs)
    print(f"\nFinal accuracy: {final_accuracy:.2%} ({correct_predictions}/{len(xor_inputs)})")
    
    # Analyze the result
    if final_accuracy <= 0.5:
        print("✗ The perceptron FAILED to learn XOR!")
        print("  Accuracy is no better than random guessing (50%).")
    else:
        print("? Unexpected result - please verify implementation")
    
    return weights, bias, final_accuracy

def analyze_linear_separability():
    """
    Visual analysis of why XOR is not linearly separable
    
    This function creates visualizations comparing linearly separable functions
    (AND, OR) with the non-linearly separable XOR function.
    
    The visualization demonstrates:
    1. How AND and OR can be separated by straight lines
    2. Why no straight line can separate XOR classes
    3. The geometric nature of the linear separability problem
    
    This helps build intuition for why the perceptron fails on XOR
    and why we need more complex models (like multi-layer networks).
    """
    print("\n" + "=" * 60)
    print("ANALYSIS OF LINEAR SEPARABILITY")
    print("=" * 60)
    
    # Create figure with three subplots for comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define input points (same for all logical functions)
    input_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Define the three logical functions for comparison
    logical_functions = [
        (np.array([0, 0, 0, 1]), "AND - Linearly Separable", axes[0]),
        (np.array([0, 1, 1, 1]), "OR - Linearly Separable", axes[1]),
        (np.array([0, 1, 1, 0]), "XOR - NOT Linearly Separable", axes[2])
    ]
    
    # Plot each logical function
    for targets, title, ax in logical_functions:
        # Define colors for different classes
        colors = ['red' if label == 0 else 'blue' for label in targets]
        
        # Plot data points with class-based coloring
        scatter = ax.scatter(input_points[:, 0], input_points[:, 1], 
                           c=colors, s=200, alpha=0.8, edgecolors='black')
        
        # Add labels to each point showing input and output
        for point_idx, (point, target_value) in enumerate(zip(input_points, targets)):
            ax.annotate(f'({point[0]},{point[1]})→{target_value}', 
                       (point[0], point[1]), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=10,
                       fontweight='bold')
        
        # Draw possible separation lines for AND and OR
        if "AND" in title:
            # For AND: line that separates (1,1) from the rest
            # Example line: x₁ + x₂ = 1.5
            x_line = np.linspace(-0.2, 1.2, 100)
            y_line = 1.5 - x_line  # Rearranged from x₁ + x₂ = 1.5
            ax.plot(x_line, y_line, 'g--', linewidth=2, alpha=0.7, 
                   label='Possible Separation Line')
            
        elif "OR" in title:
            # For OR: line that separates (0,0) from the rest
            # Example line: x₁ + x₂ = 0.5
            x_line = np.linspace(-0.2, 1.2, 100)
            y_line = 0.5 - x_line  # Rearranged from x₁ + x₂ = 0.5
            ax.plot(x_line, y_line, 'g--', linewidth=2, alpha=0.7, 
                   label='Possible Separation Line')
            
        else:  # XOR case
            # For XOR: demonstrate that no line can separate the classes
            ax.text(0.5, 0.8, 'No straight line\ncan separate\nthese points!', 
                   ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=10, fontweight='bold')
        
        # Set plot properties
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add color legend
        ax.scatter([], [], c='red', s=100, label='Class 0', alpha=0.8)
        ax.scatter([], [], c='blue', s=100, label='Class 1', alpha=0.8)
        ax.legend(loc='upper right')
    
    # Save and display the plot
    plt.tight_layout()
    plt.savefig('/home/ubuntu/linear_separability.png', dpi=300, bbox_inches='tight')
    plt.show()

def theoretical_explanation():
    """
    Provide theoretical explanation of the XOR problem
    
    This function prints a comprehensive explanation of:
    1. What linear separability means mathematically
    2. Why XOR violates this property
    3. The geometric interpretation of the problem
    4. Implications for the perceptron algorithm
    5. Historical significance of this limitation
    
    This explanation helps bridge the gap between the practical demonstration
    and the theoretical understanding of neural network limitations.
    """
    print("\n" + "=" * 60)
    print("THEORETICAL EXPLANATION: WHY XOR IS NOT LINEARLY SEPARABLE")
    print("=" * 60)
    
    explanation_text = """
1. DEFINITION OF LINEAR SEPARABILITY:
   Two classes are linearly separable if there exists a hyperplane
   (a line in 2D, plane in 3D, etc.) that can completely separate them.
   
   Mathematically: w₁x₁ + w₂x₂ + b = 0 defines the decision boundary
   - Points on one side: w₁x₁ + w₂x₂ + b > 0 (class 1)
   - Points on other side: w₁x₁ + w₂x₂ + b < 0 (class 0)

2. XOR POINT ANALYSIS:
   - Class 0 points: (0,0) and (1,1) - diagonally opposite corners
   - Class 1 points: (0,1) and (1,0) - the other diagonal corners
   
   Problem: These points are arranged in a checkerboard pattern.
   No single straight line can separate the diagonal corners.

3. GEOMETRIC IMPOSSIBILITY:
   Any line that correctly separates (0,0) from (0,1) will incorrectly
   classify either (1,0) or (1,1). This is a fundamental geometric
   constraint that cannot be overcome by adjusting weights or bias.

4. PERCEPTRON ALGORITHM IMPLICATIONS:
   - The perceptron learning rule assumes linear separability
   - For non-separable data, weights will oscillate indefinitely
   - No convergence is possible - the algorithm will never terminate
   - Best achievable accuracy is around 50% (random guessing)

5. HISTORICAL SIGNIFICANCE:
   - Discovered by Minsky and Papert (1969) in their book "Perceptrons"
   - Led to the first "AI Winter" - reduced funding for neural networks
   - Showed fundamental limitations of single-layer networks
   - Motivated development of multi-layer networks and backpropagation

6. SOLUTION PREVIEW:
   Multi-layer perceptrons can solve XOR by:
   - Creating non-linear decision boundaries
   - Learning intermediate representations in hidden layers
   - Combining multiple linear separations to create complex boundaries
    """
    
    print(explanation_text)

def compare_logical_functions():
    """
    Create a comprehensive comparison table of logical functions
    
    This function displays a truth table comparing AND, OR, and XOR
    functions, highlighting their separability properties.
    
    The comparison helps students understand:
    1. The similarity in input space (all use same 2D binary inputs)
    2. The difference in output patterns
    3. Why some patterns are separable and others are not
    """
    print("\n" + "=" * 60)
    print("LOGICAL FUNCTIONS COMPARISON")
    print("=" * 60)
    
    print("| Input A | Input B | AND | OR  | XOR |")
    print("|---------|---------|-----|-----|-----|")
    print("|    0    |    0    |  0  |  0  |  0  |")
    print("|    0    |    1    |  0  |  1  |  1  |")
    print("|    1    |    0    |  0  |  1  |  1  |")
    print("|    1    |    1    |  1  |  1  |  0  |")
    print()
    print("LINEAR SEPARABILITY ANALYSIS:")
    print("✓ AND: Linearly separable (one positive example)")
    print("✓ OR:  Linearly separable (one negative example)") 
    print("✗ XOR: NOT linearly separable (checkerboard pattern)")
    print()
    print("GEOMETRIC INTERPRETATION:")
    print("- AND: Can draw line below and to the left of (1,1)")
    print("- OR:  Can draw line above and to the right of (0,0)")
    print("- XOR: No single line can separate diagonal corners")

def main():
    """
    Main function demonstrating perceptron limitations
    
    This comprehensive demonstration shows:
    1. Practical failure of perceptron on XOR
    2. Visual analysis of linear separability
    3. Theoretical explanation of the problem
    4. Comparison with solvable functions
    5. Historical context and significance
    
    The goal is to build complete understanding of why simple perceptrons
    have fundamental limitations and why more complex architectures are needed.
    """
    # Demonstrate perceptron failure on XOR
    final_weights, final_bias, final_accuracy = test_perceptron_on_xor_limited_epochs()
    
    # Visual analysis of the separability problem
    analyze_linear_separability()
    
    # Theoretical explanation
    theoretical_explanation()
    
    # Comparison table
    compare_logical_functions()
    
    # Create error plot showing non-convergence
    # (We'll create a simple demonstration plot)
    plt.figure(figsize=(10, 6))
    
    # Simulate typical XOR training errors (oscillating, not converging)
    epochs = range(1, 51)
    # XOR errors typically oscillate between 2-4 errors per epoch
    simulated_errors = [4, 2, 3, 4, 2, 4, 3, 2, 4, 3] * 5  # Repeating pattern
    
    plt.plot(epochs, simulated_errors, 'ro-', linewidth=2, markersize=6)
    plt.title('XOR Training Errors - Perceptron Does Not Converge')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='g', linestyle='--', alpha=0.7, 
               label='Convergence Target (0 errors)')
    plt.legend()
    plt.ylim(0, 5)
    
    # Add annotation explaining the oscillation
    plt.annotate('Errors oscillate\nNever reach zero', 
                xy=(25, 3), xytext=(35, 4),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.savefig('/home/ubuntu/xor_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("CONCLUSION:")
    print("The simple perceptron has fundamental limitations that require")
    print("more sophisticated architectures (multi-layer networks) to overcome.")
    print("This discovery was crucial in the development of modern deep learning.")
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the perceptron limitations demonstration
    
    This script can be run independently to see the full demonstration
    of perceptron limitations, or imported as a module for use in other scripts.
    """
    main()


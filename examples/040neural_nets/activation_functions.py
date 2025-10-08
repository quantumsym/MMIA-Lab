#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# ============================================================================

def sigmoid(x):
    """
    Sigmoid activation function implementation
    
    The sigmoid function maps any real-valued number to a value between 0 and 1.
    It has an S-shaped curve and is differentiable everywhere.
    
    Mathematical formula: f(x) = 1 / (1 + e^(-x))
    
    Advantages:
    - Output is always positive and bounded between 0 and 1
    - Differentiable everywhere with smooth gradient
    - Can be interpreted as probability values
    
    Disadvantages:
    - Vanishing gradient problem for very large or small input values
    - Output is not zero-centered (always positive)
    - Computationally more expensive than ReLU
    
    Args:
        x (numpy.ndarray or float): Input value(s) to the sigmoid function.
                                   Can be a single number or array of numbers.
    
    Returns:
        numpy.ndarray or float: Output value(s) after applying sigmoid function.
                               Values will be in range (0, 1).
    
    Note:
        We clip the input values to prevent numerical overflow when computing
        the exponential function. Values beyond [-500, 500] would cause
        overflow/underflow issues.
    """
    # Clip input values to prevent numerical overflow in exp() function
    # This ensures numerical stability for very large positive or negative values
    x_clipped = np.clip(x, -500, 500)
    
    # Apply sigmoid formula: 1 / (1 + exp(-x))
    # The exponential function exp(-x) approaches 0 for large positive x
    # and approaches infinity for large negative x
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function
    
    The derivative of sigmoid has a very useful mathematical property:
    if s = sigmoid(x), then ds/dx = s * (1 - s)
    
    This property makes backpropagation computation very efficient because
    we can compute the derivative using the sigmoid output directly,
    without needing to recompute the original input.
    
    Mathematical derivation:
    f(x) = 1 / (1 + e^(-x))
    f'(x) = e^(-x) / (1 + e^(-x))^2
    f'(x) = (1 / (1 + e^(-x))) * (e^(-x) / (1 + e^(-x)))
    f'(x) = f(x) * (1 - f(x))
    
    Args:
        x (numpy.ndarray or float): Input value(s) for which to compute derivative.
                                   These are the same values passed to sigmoid function.
    
    Returns:
        numpy.ndarray or float: Derivative values at the given input points.
                               Maximum value is 0.25 (at x=0), approaches 0 for large |x|.
    
    Note:
        The derivative has maximum value of 0.25 when x=0 (sigmoid output = 0.5).
        For large positive or negative x values, the derivative approaches 0,
        which can cause vanishing gradient problems in deep networks.
    """
    # First compute sigmoid of input
    sigmoid_output = sigmoid(x)
    
    # Apply derivative formula: sigmoid(x) * (1 - sigmoid(x))
    # This is more efficient than computing the derivative from scratch
    return sigmoid_output * (1 - sigmoid_output)

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function
    
    ReLU is one of the most popular activation functions in modern neural networks.
    It's computationally efficient and helps mitigate the vanishing gradient problem.
    
    Mathematical formula: f(x) = max(0, x)
    
    Advantages:
    - Computationally very efficient (simple max operation)
    - Does not suffer from vanishing gradient problem for positive inputs
    - Introduces sparsity (many neurons can be inactive with output = 0)
    - Empirically works well in practice for deep networks
    
    Disadvantages:
    - Not differentiable at x = 0 (though we can define it by convention)
    - "Dying ReLU" problem: neurons can become permanently inactive
    - Output is not bounded (can grow arbitrarily large)
    
    Args:
        x (numpy.ndarray or float): Input value(s) to the ReLU function.
                                   Can be any real number or array of numbers.
    
    Returns:
        numpy.ndarray or float: Output value(s) after applying ReLU.
                               All negative inputs become 0, positive inputs unchanged.
    
    Note:
        The numpy.maximum function performs element-wise comparison between
        0 and each element in x, returning the larger value.
    """
    # Apply ReLU formula: max(0, x) for each element
    # numpy.maximum performs element-wise maximum operation
    # All negative values become 0, positive values remain unchanged
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU activation function
    
    The derivative of ReLU is a step function:
    - 1 for x > 0 (positive inputs)
    - 0 for x < 0 (negative inputs)
    - Undefined at x = 0 (but we conventionally use 0)
    
    This simple derivative is one reason why ReLU is computationally efficient
    and helps with the vanishing gradient problem.
    
    Mathematical definition:
    f'(x) = 1 if x > 0
    f'(x) = 0 if x <= 0
    
    Args:
        x (numpy.ndarray or float): Input value(s) for which to compute derivative.
                                   These are the same values passed to ReLU function.
    
    Returns:
        numpy.ndarray or float: Derivative values (0 or 1) at the given input points.
                               Returns 1.0 for positive inputs, 0.0 for non-positive inputs.
    
    Note:
        We use the convention that the derivative at x=0 is 0, though
        mathematically the derivative is undefined at this point.
        The .astype(float) ensures the output is floating-point numbers.
    """
    # Create boolean mask: True where x > 0, False elsewhere
    # Convert boolean to float: True becomes 1.0, False becomes 0.0
    return (x > 0).astype(float)

def step_function(x):
    """
    Step (Heaviside) activation function
    
    This is the simplest activation function, used in the original perceptron.
    It produces binary outputs, making it suitable for binary classification
    but unsuitable for gradient-based learning algorithms.
    
    Mathematical formula:
    f(x) = 1 if x >= 0
    f(x) = 0 if x < 0
    
    Advantages:
    - Very simple to compute and understand
    - Produces clear binary outputs
    - Historically important (original perceptron)
    
    Disadvantages:
    - Not differentiable (derivative is 0 everywhere except at x=0)
    - Cannot be used with gradient-based learning (backpropagation)
    - Discontinuous function creates optimization difficulties
    
    Args:
        x (numpy.ndarray or float): Input value(s) to the step function.
                                   Can be any real number or array of numbers.
    
    Returns:
        numpy.ndarray or float: Binary output value(s) (0.0 or 1.0).
                               Returns 1.0 for non-negative inputs, 0.0 for negative inputs.
    
    Note:
        We use the convention that step(0) = 1, though sometimes it's defined as 0.
        The .astype(float) ensures the output is floating-point numbers.
    """
    # Create boolean mask: True where x >= 0, False elsewhere
    # Convert boolean to float: True becomes 1.0, False becomes 0.0
    return (x >= 0).astype(float)

def step_derivative(x):
    """
    Derivative of the step activation function
    
    Mathematically, the derivative of the step function is:
    - 0 everywhere except at x = 0
    - Undefined (infinite) at x = 0
    
    For practical purposes in neural networks, we return 0 everywhere.
    This is why the step function cannot be used with backpropagation:
    the gradient is always 0, so no learning can occur.
    
    Args:
        x (numpy.ndarray or float): Input value(s) for which to compute derivative.
    
    Returns:
        numpy.ndarray or float: Always returns 0 (same shape as input).
                               This represents the fact that the step function
                               has zero gradient almost everywhere.
    
    Note:
        In the original perceptron learning rule, the derivative is not used.
        Instead, the perceptron uses a different update rule based on
        classification errors rather than gradients.
    """
    # Return array of zeros with same shape as input
    # This represents the zero gradient of the step function
    return np.zeros_like(x)

# ============================================================================
# VISUALIZATION FUNCTION FOR ACTIVATION FUNCTIONS
# ============================================================================

def visualize_activation_functions():
    """
    Create comprehensive visualization of activation functions and their derivatives
    
    This function generates a 3x2 subplot showing:
    - Top row: Sigmoid function and its derivative
    - Middle row: ReLU function and its derivative  
    - Bottom row: Step function and its derivative
    
    The visualization helps understand:
    1. The shape and range of each activation function
    2. How the derivatives behave (important for backpropagation)
    3. The differences between continuous and discontinuous functions
    4. The vanishing gradient problem in sigmoid vs. ReLU
    
    The plot is saved as 'activation_functions.png' for documentation purposes.
    
    Returns:
        None: Function creates and displays plot, saves to file.
    
    Note:
        The x-axis range [-5, 5] is chosen to show the characteristic behavior
        of each function. Sigmoid saturates at the extremes, ReLU is linear
        for positive values, and step function shows the discontinuity.
    """
    # Create input values for plotting
    # Range [-5, 5] with 1000 points gives smooth curves
    x_values = np.linspace(-5, 5, 1000)
    
    # Create figure with 3 rows and 2 columns of subplots
    # figsize=(12, 10) provides good aspect ratio for detailed viewing
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # ========================================================================
    # SIGMOID FUNCTION AND DERIVATIVE
    # ========================================================================
    
    # Plot sigmoid function (top-left subplot)
    axes[0, 0].plot(x_values, sigmoid(x_values), 'b-', linewidth=2, label='Sigmoid')
    axes[0, 0].set_title('Sigmoid Activation Function')
    axes[0, 0].set_xlabel('Input (x)')
    axes[0, 0].set_ylabel('Output f(x)')
    axes[0, 0].grid(True, alpha=0.3)  # Light grid for better readability
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.1, 1.1)  # Set y-limits to show full sigmoid range
    
    # Plot sigmoid derivative (top-right subplot)
    axes[0, 1].plot(x_values, sigmoid_derivative(x_values), 'b--', linewidth=2, 
                    label='Sigmoid Derivative')
    axes[0, 1].set_title('Sigmoid Derivative')
    axes[0, 1].set_xlabel('Input (x)')
    axes[0, 1].set_ylabel("Derivative f'(x)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.05, 0.3)  # Show derivative range [0, 0.25]
    
    # ========================================================================
    # RELU FUNCTION AND DERIVATIVE
    # ========================================================================
    
    # Plot ReLU function (middle-left subplot)
    axes[1, 0].plot(x_values, relu(x_values), 'r-', linewidth=2, label='ReLU')
    axes[1, 0].set_title('ReLU Activation Function')
    axes[1, 0].set_xlabel('Input (x)')
    axes[1, 0].set_ylabel('Output f(x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-1, 6)  # Show ReLU behavior for negative and positive inputs
    
    # Plot ReLU derivative (middle-right subplot)
    axes[1, 1].plot(x_values, relu_derivative(x_values), 'r--', linewidth=2, 
                    label='ReLU Derivative')
    axes[1, 1].set_title('ReLU Derivative')
    axes[1, 1].set_xlabel('Input (x)')
    axes[1, 1].set_ylabel("Derivative f'(x)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(-0.1, 1.1)  # Show step-like derivative behavior
    
    # ========================================================================
    # STEP FUNCTION AND DERIVATIVE
    # ========================================================================
    
    # Plot step function (bottom-left subplot)
    axes[2, 0].plot(x_values, step_function(x_values), 'g-', linewidth=2, label='Step Function')
    axes[2, 0].set_title('Step Activation Function')
    axes[2, 0].set_xlabel('Input (x)')
    axes[2, 0].set_ylabel('Output f(x)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    axes[2, 0].set_ylim(-0.1, 1.1)  # Show binary output range
    
    # Plot step derivative (bottom-right subplot)
    axes[2, 1].plot(x_values, step_derivative(x_values), 'g--', linewidth=2, 
                    label='Step Derivative')
    axes[2, 1].set_title('Step Function Derivative')
    axes[2, 1].set_xlabel('Input (x)')
    axes[2, 1].set_ylabel("Derivative f'(x)")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    axes[2, 1].set_ylim(-0.1, 0.1)  # Show that derivative is always 0
    
    # ========================================================================
    # FINALIZE PLOT
    # ========================================================================
    
    # Adjust spacing between subplots for better readability
    plt.tight_layout()
    
    # Save plot to file with high resolution for documentation
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

# ============================================================================
# MAIN EXECUTION AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block for testing activation functions
    
    This section runs when the script is executed directly (not imported).
    It performs comprehensive testing of all activation functions with
    sample input values and generates visualization.
    """
    
    print("=" * 60)
    print("ACTIVATION FUNCTIONS TESTING AND DEMONSTRATION")
    print("=" * 60)
    
    # Define test input values to demonstrate function behavior
    # These values are chosen to show different regions of each function:
    # -2: large negative (sigmoid ≈ 0, ReLU = 0, step = 0)
    # -1: moderate negative
    # 0: boundary point (important for step function and ReLU)
    # 1: moderate positive
    # 2: large positive (sigmoid ≈ 1, ReLU = 2, step = 1)
    test_input_values = np.array([-2, -1, 0, 1, 2])
    
    print(f"Test input values: {test_input_values}")
    print("-" * 60)
    
    # ========================================================================
    # TEST SIGMOID FUNCTION
    # ========================================================================
    
    print("SIGMOID FUNCTION RESULTS:")
    sigmoid_outputs = sigmoid(test_input_values)
    sigmoid_derivatives = sigmoid_derivative(test_input_values)
    
    print(f"Sigmoid outputs:     {sigmoid_outputs}")
    print(f"Sigmoid derivatives: {sigmoid_derivatives}")
    print()
    
    # Verify sigmoid properties
    print("Sigmoid properties verification:")
    print(f"- All outputs in (0,1): {np.all((sigmoid_outputs > 0) & (sigmoid_outputs < 1))}")
    print(f"- Maximum derivative: {np.max(sigmoid_derivatives):.4f} (should be ≤ 0.25)")
    print()
    
    # ========================================================================
    # TEST RELU FUNCTION
    # ========================================================================
    
    print("RELU FUNCTION RESULTS:")
    relu_outputs = relu(test_input_values)
    relu_derivatives = relu_derivative(test_input_values)
    
    print(f"ReLU outputs:        {relu_outputs}")
    print(f"ReLU derivatives:    {relu_derivatives}")
    print()
    
    # Verify ReLU properties
    print("ReLU properties verification:")
    print(f"- No negative outputs: {np.all(relu_outputs >= 0)}")
    print(f"- Derivatives are 0 or 1: {np.all(np.isin(relu_derivatives, [0, 1]))}")
    print()
    
    # ========================================================================
    # TEST STEP FUNCTION
    # ========================================================================
    
    print("STEP FUNCTION RESULTS:")
    step_outputs = step_function(test_input_values)
    step_derivatives = step_derivative(test_input_values)
    
    print(f"Step outputs:        {step_outputs}")
    print(f"Step derivatives:    {step_derivatives}")
    print()
    
    # Verify step function properties
    print("Step function properties verification:")
    print(f"- Only binary outputs: {np.all(np.isin(step_outputs, [0, 1]))}")
    print(f"- All derivatives zero: {np.all(step_derivatives == 0)}")
    print()
    
    # ========================================================================
    # GENERATE VISUALIZATION
    # ========================================================================
    
    print("Generating comprehensive visualization...")
    visualize_activation_functions()
    
    print("=" * 60)
    print("SUMMARY:")
    print("- Sigmoid: Smooth, bounded, but suffers from vanishing gradients")
    print("- ReLU: Simple, efficient, helps with vanishing gradients")
    print("- Step: Binary output, not suitable for gradient-based learning")
    print("- Visualization saved as 'activation_functions.png'")
    print("=" * 60)


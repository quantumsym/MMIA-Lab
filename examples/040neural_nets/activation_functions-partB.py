#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def linear_saturation(x):
    """
    Linear saturation activation function.

    f(x) = 0   if x < 0
    f(x) = x   if 0 <= x <= 1
    f(x) = 1   if x > 1

    Args:
        x: Input array or scalar

    Returns:
        Output after applying linear saturation function
    """
    # Use numpy.clip to efficiently implement the piecewise function
    # clip(x, min_val, max_val) constrains values between min_val and max_val
    return np.clip(x, 0, 1)

def linear_saturation_derivative(x):
    """
    Derivative of linear saturation function.

    f'(x) = 0   if x < 0
    f'(x) = 1   if 0 <= x <= 1
    f'(x) = 0   if x > 1

    Args:
        x: Input array or scalar

    Returns:
        Derivative values
    """
    # Initialize derivative array with zeros
    derivative = np.zeros_like(x)

    # Set derivative to 1 where function has slope (linear region)
    # Use logical indexing to find values in the linear region
    linear_region = (x >= 0) & (x <= 1)
    derivative[linear_region] = 1.0

    return derivative

def hyperbolic_tangent(x):
    """
    Hyperbolic tangent activation function.

    f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Properties:
    - Output range: (-1, 1)
    - Smooth and differentiable everywhere
    - Odd function: tanh(-x) = -tanh(x)

    Args:
        x: Input array or scalar

    Returns:
        Output after applying hyperbolic tangent
    """
    return np.tanh(x)

def hyperbolic_tangent_derivative(x):
    """
    Derivative of hyperbolic tangent function.

    f'(x) = 1 - tanh²(x) = sech²(x)

    This is a key property: the derivative can be computed from the function value itself.
    If y = tanh(x), then dy/dx = 1 - y²

    Args:
        x: Input array or scalar

    Returns:
        Derivative values
    """
    # Method 1: Direct calculation using tanh
    tanh_x = np.tanh(x)
    return 1.0 - tanh_x**2

    # Alternative method 2: Using sech² formula
    # return 1.0 / np.cosh(x)**2

def demonstrate_activation_functions():
    """
    Demonstrate both activation functions and their derivatives with plots.
    """
    # Create input range for visualization
    x = np.linspace(-3, 3, 1000)

    # Calculate function values
    linear_sat_values = linear_saturation(x)
    linear_sat_derivatives = linear_saturation_derivative(x)

    tanh_values = hyperbolic_tangent(x)
    tanh_derivatives = hyperbolic_tangent_derivative(x)

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Linear Saturation Function
    axes[0, 0].plot(x, linear_sat_values, 'b-', linewidth=2, label='Linear Saturation')
    axes[0, 0].set_title('Linear Saturation Function f(x)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Linear Saturation Derivative
    axes[0, 1].plot(x, linear_sat_derivatives, 'r-', linewidth=2, label="Linear Saturation f'(x)")
    axes[0, 1].set_title("Linear Saturation Derivative f'(x)")
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel("f'(x)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Hyperbolic Tangent Function
    axes[1, 0].plot(x, tanh_values, 'g-', linewidth=2, label='Hyperbolic Tangent')
    axes[1, 0].set_title('Hyperbolic Tangent Function f(x) = tanh(x)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=1, color='k', linestyle=':', alpha=0.5, label='y=1 asymptote')
    axes[1, 0].axhline(y=-1, color='k', linestyle=':', alpha=0.5, label='y=-1 asymptote')
    axes[1, 0].legend()

    # Plot 4: Hyperbolic Tangent Derivative
    axes[1, 1].plot(x, tanh_derivatives, 'm-', linewidth=2, label="Tanh f'(x)")
    axes[1, 1].set_title("Hyperbolic Tangent Derivative f'(x) = 1 - tanh²(x)")
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel("f'(x)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return x, linear_sat_values, linear_sat_derivatives, tanh_values, tanh_derivatives

def test_activation_functions():
    """
    Test the activation functions with specific input values.
    """
    print("=== ACTIVATION FUNCTIONS TEST ===")
    print()

    # Test values covering different regions
    test_inputs = np.array([-2, -0.5, 0, 0.5, 1, 1.5, 2])

    print("Linear Saturation Function:")
    print("x\t\tf(x)\t\tf'(x)")
    print("-" * 40)
    for x_val in test_inputs:
        f_val = linear_saturation(x_val)
        df_val = linear_saturation_derivative(x_val)
        print(f"{x_val:8.1f}\t{f_val:8.3f}\t{df_val:8.3f}")

    print()
    print("Hyperbolic Tangent Function:")
    print("x\t\tf(x)\t\tf'(x)")
    print("-" * 40)
    for x_val in test_inputs:
        f_val = hyperbolic_tangent(x_val)
        df_val = hyperbolic_tangent_derivative(x_val)
        print(f"{x_val:8.1f}\t{f_val:8.3f}\t{df_val:8.3f}")

def compare_activation_properties():
    """
    Compare key properties of both activation functions.
    """
    print("\n=== ACTIVATION FUNCTION PROPERTIES COMPARISON ===")
    print()

    properties = {
        'Linear Saturation': {
            'Range': '[0, 1]',
            'Domain': '(-∞, ∞)',
            'Differentiable': 'No (at x=0, x=1)',
            'Monotonic': 'Yes (non-decreasing)',
            'Zero-centered': 'No',
            'Vanishing gradient': 'Yes (outside [0,1])'
        },
        'Hyperbolic Tangent': {
            'Range': '(-1, 1)',
            'Domain': '(-∞, ∞)',
            'Differentiable': 'Yes (everywhere)',
            'Monotonic': 'Yes (strictly increasing)',
            'Zero-centered': 'Yes',
            'Vanishing gradient': 'Yes (for large |x|)'
        }
    }

    for func_name, props in properties.items():
        print(f"{func_name}:")
        for prop, value in props.items():
            print(f"  {prop:20s}: {value}")
        print()

# Main execution
if __name__ == "__main__":
    print("Neural Network Activation Functions Implementation")
    print("=" * 55)

    # Test the functions
    test_activation_functions()

    # Compare properties
    compare_activation_properties()

    # Create visualization
    print("Generating plots...")
    x_vals, lin_vals, lin_derivs, tanh_vals, tanh_derivs = demonstrate_activation_functions()

    print("Script execution completed!")


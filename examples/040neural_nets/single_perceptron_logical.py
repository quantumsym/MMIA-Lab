#!/usr/bin/env python

# ==========================================
# SINGLE PERCEPTRON - LOGICAL FUNCTIONS
# Demonstrates perceptron capabilities and limitations
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def step_function(x):
    """
    Step activation function for perceptron
    Returns 1 if x >= 0, otherwise 0
    """
    return np.where(x >= 0, 1, 0)

# ==========================================
# PERCEPTRON IMPLEMENTATION
# ==========================================

def create_perceptron(input_size):
    """
    Creates a simple perceptron with random weights
    
    Parameters:
    - input_size: number of input features
    
    Returns:
    - weights: array of weights for each input
    - bias: bias term (threshold)
    """
    # Initialize weights randomly (small values)
    weights = np.random.randn(input_size) * 0.1
    bias = np.random.randn() * 0.1
    
    return weights, bias

def perceptron_predict(inputs, weights, bias):
    """
    Makes a prediction using perceptron
    
    Parameters:
    - inputs: input features (array)
    - weights: learned weights
    - bias: learned bias
    
    Returns:
    - prediction: 0 or 1
    """
    # Calculate weighted sum
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Apply step function
    prediction = step_function(weighted_sum)
    
    return prediction

def train_perceptron(X, y, weights, bias, learning_rate=0.1, epochs=100):
    """
    Trains a perceptron using the perceptron learning rule
    
    Parameters:
    - X: input data (samples x features)
    - y: target labels (0 or 1)
    - weights: initial weights
    - bias: initial bias
    - learning_rate: learning rate
    - epochs: number of training iterations
    
    Returns:
    - trained weights and bias
    """
    print(f"Training perceptron for {epochs} epochs...")
    
    for epoch in range(epochs):
        errors = 0
        
        # Train on each sample
        for i in range(len(X)):
            # Make prediction
            prediction = perceptron_predict(X[i], weights, bias)
            
            # Calculate error
            error = y[i] - prediction
            
            # Update weights and bias if there's an error
            if error != 0:
                weights += learning_rate * error * X[i]
                bias += learning_rate * error
                errors += 1
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Errors: {errors}")
        
        # Stop early if no errors
        if errors == 0:
            print(f"Converged at epoch {epoch + 1}")
            break
    
    return weights, bias

# ==========================================
# LOGICAL FUNCTIONS TESTING
# ==========================================

def test_and_function():
    """
    Tests perceptron on logical AND function
    Shows that perceptron can learn linearly separable problems
    """
    print("\n" + "="*40)
    print("TESTING AND FUNCTION")
    print("="*40)
    
    # Define input patterns for AND function
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])  # Only (1,1) gives 1
    
    print("AND Truth Table:")
    print("Input A | Input B | Output")
    print("--------|---------|-------")
    for i in range(len(X)):
        print(f"   {X[i][0]}    |    {X[i][1]}    |   {y_and[i]}")
    
    # Create and train perceptron
    weights, bias = create_perceptron(2)
    weights, bias = train_perceptron(X, y_and, weights, bias, epochs=100)
    
    print("\nResults for AND function:")
    print("Input -> Expected -> Predicted -> Correct?")
    correct_count = 0
    
    for i, inputs in enumerate(X):
        prediction = perceptron_predict(inputs, weights, bias)
        is_correct = prediction == y_and[i]
        correct_count += is_correct
        print(f"{inputs} -> {y_and[i]} -> {prediction} -> {'✓' if is_correct else '✗'}")
    
    accuracy = correct_count / len(X) * 100
    print(f"Accuracy: {accuracy:.1f}%")
    
    return weights, bias

def test_or_function():
    """
    Tests perceptron on logical OR function
    """
    print("\n" + "="*40)
    print("TESTING OR FUNCTION")
    print("="*40)
    
    # Define input patterns for OR function
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])  # Only (0,0) gives 0
    
    print("OR Truth Table:")
    print("Input A | Input B | Output")
    print("--------|---------|-------")
    for i in range(len(X)):
        print(f"   {X[i][0]}    |    {X[i][1]}    |   {y_or[i]}")
    
    # Create and train perceptron
    weights, bias = create_perceptron(2)
    weights, bias = train_perceptron(X, y_or, weights, bias, epochs=100)
    
    print("\nResults for OR function:")
    print("Input -> Expected -> Predicted -> Correct?")
    correct_count = 0
    
    for i, inputs in enumerate(X):
        prediction = perceptron_predict(inputs, weights, bias)
        is_correct = prediction == y_or[i]
        correct_count += is_correct
        print(f"{inputs} -> {y_or[i]} -> {prediction} -> {'✓' if is_correct else '✗'}")
    
    accuracy = correct_count / len(X) * 100
    print(f"Accuracy: {accuracy:.1f}%")
    
    return weights, bias

def test_xor_failure():
    """
    Demonstrates that single perceptron CANNOT learn XOR
    This shows the limitation of single-layer networks
    """
    print("\n" + "="*40)
    print("TESTING XOR - SHOWING LIMITATION")
    print("="*40)
    
    # XOR truth table
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR: different inputs give 1
    
    print("XOR Truth Table:")
    print("Input A | Input B | Output")
    print("--------|---------|-------")
    for i in range(len(X)):
        print(f"   {X[i][0]}    |    {X[i][1]}    |   {y_xor[i]}")
    
    print("\nAttempting to learn XOR function...")
    print("(This will fail - demonstrating perceptron limitation)")
    
    # Create and train perceptron
    weights, bias = create_perceptron(2)
    weights, bias = train_perceptron(X, y_xor, weights, bias, epochs=100)
    
    print("\nResults for XOR function:")
    print("Input -> Expected -> Predicted -> Correct?")
    correct_count = 0
    
    for i, inputs in enumerate(X):
        prediction = perceptron_predict(inputs, weights, bias)
        is_correct = prediction == y_xor[i]
        correct_count += is_correct
        print(f"{inputs} -> {y_xor[i]} -> {prediction} -> {'✓' if is_correct else '✗'}")
    
    accuracy = correct_count / len(X) * 100
    print(f"Accuracy: {accuracy:.1f}%")
    
    print("\n" + "="*40)
    print("EXPLANATION:")
    print("The perceptron can only learn linearly separable problems.")
    print("XOR is NOT linearly separable - you cannot draw a single")
    print("straight line to separate the classes in 2D space.")
    print("This is why we need multi-layer networks!")
    print("="*40)

def visualize_decision_boundary(X, y, weights, bias, title="Perceptron Decision Boundary"):
    """
    Visualizes the decision boundary learned by the perceptron
    """
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    colors = ['red', 'blue']
    for i in range(2):
        mask = (y == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   s=100, label=f'Class {i}', edgecolors='black')
    
    # Plot decision boundary
    if weights[1] != 0:  # Avoid division by zero
        x_boundary = np.linspace(-0.5, 1.5, 100)
        y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
        plt.plot(x_boundary, y_boundary, 'g--', linewidth=2, label='Decision Boundary')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Input A')
    plt.ylabel('Input B')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """
    Main function to run all logical function tests
    """
    print("SINGLE PERCEPTRON - LOGICAL FUNCTIONS")
    print("=" * 50)
    
    # Test AND function
    and_weights, and_bias = test_and_function()
    
    # Test OR function  
    or_weights, or_bias = test_or_function()
    
    # Test XOR function (will fail)
    test_xor_failure()
    
    # Visualize decision boundaries (if matplotlib is available)
    try:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        
        # Visualize AND function
        y_and = np.array([0, 0, 0, 1])
        visualize_decision_boundary(X, y_and, and_weights, and_bias, "AND Function - Decision Boundary")
        
        # Visualize OR function
        y_or = np.array([0, 1, 1, 1])
        visualize_decision_boundary(X, y_or, or_weights, or_bias, "OR Function - Decision Boundary")
        
    except Exception as e:
        print(f"\nVisualization not available: {e}")
        print("Install matplotlib to see decision boundary plots")
    
    print("\n" + "="*50)
    print("KEY TAKEAWAYS:")
    print("1. Single perceptron can learn AND and OR (linearly separable)")
    print("2. Single perceptron CANNOT learn XOR (not linearly separable)")
    print("3. This limitation led to the development of multi-layer networks")
    print("4. Decision boundary is always a straight line for single perceptron")
    print("="*50)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()

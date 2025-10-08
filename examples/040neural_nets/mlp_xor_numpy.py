#!/usr/bin/env python

# ==========================================
# MULTI-LAYER PERCEPTRON - XOR SOLVER
# Demonstrates how MLP solves non-linearly separable problems
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def sigmoid(x):
    """
    Sigmoid activation function: transforms any real number to (0,1) range
    Used for binary classification and hidden layers
    """
    # Clip x to prevent overflow in exp function
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function - needed for backpropagation
    """
    return sigmoid(x) * (1 - sigmoid(x))

# ==========================================
# MULTI-LAYER PERCEPTRON FUNCTIONS
# ==========================================

def initialize_mlp_weights(input_size, hidden_size, output_size):
    """
    Initialize weights for a simple MLP with one hidden layer
    
    Parameters:
    - input_size: number of input features
    - hidden_size: number of neurons in hidden layer
    - output_size: number of output neurons
    
    Returns:
    - Dictionary containing all weights and biases
    """
    # Small random weights to break symmetry
    weights = {
        'W1': np.random.randn(input_size, hidden_size) * 0.5,  # Input to hidden
        'b1': np.zeros((1, hidden_size)),                      # Hidden layer bias
        'W2': np.random.randn(hidden_size, output_size) * 0.5, # Hidden to output
        'b2': np.zeros((1, output_size))                       # Output layer bias
    }
    
    return weights

def mlp_forward_pass(X, weights):
    """
    Forward pass through the MLP
    
    Parameters:
    - X: input data (can be single sample or batch)
    - weights: dictionary containing weights and biases
    
    Returns:
    - Dictionary containing activations and pre-activations
    """
    # Ensure X is 2D array (add batch dimension if needed)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Forward pass calculations
    cache = {}
    
    # Input to hidden layer
    cache['z1'] = np.dot(X, weights['W1']) + weights['b1']  # Linear transformation
    cache['a1'] = sigmoid(cache['z1'])                      # Apply activation
    
    # Hidden to output layer
    cache['z2'] = np.dot(cache['a1'], weights['W2']) + weights['b2']  # Linear transformation
    cache['a2'] = sigmoid(cache['z2'])                                # Apply activation
    
    # Store input for backpropagation
    cache['X'] = X
    
    return cache

def compute_cost(y_true, y_pred):
    """
    Compute binary cross-entropy cost
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted probabilities
    
    Returns:
    - cost: average loss
    """
    # Ensure y_true is same shape as y_pred
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    
    m = y_true.shape[0]  # Number of samples
    
    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
    
    # Binary cross-entropy formula
    cost = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    return cost

def mlp_backward_pass(cache, weights, y_true, learning_rate):
    """
    Backward pass (backpropagation) to compute gradients and update weights
    
    Parameters:
    - cache: forward pass results
    - weights: current weights and biases
    - y_true: true labels
    - learning_rate: step size for gradient descent
    
    Returns:
    - Updated weights
    """
    # Ensure y_true has correct shape
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    
    m = y_true.shape[0]  # Number of samples
    
    # Backward pass - compute gradients
    
    # Output layer gradients
    dz2 = cache['a2'] - y_true  # Derivative of cost w.r.t. z2
    dW2 = np.dot(cache['a1'].T, dz2) / m  # Gradient for W2
    db2 = np.mean(dz2, axis=0, keepdims=True)  # Gradient for b2
    
    # Hidden layer gradients
    da1 = np.dot(dz2, weights['W2'].T)  # Backpropagate error to hidden layer
    dz1 = da1 * sigmoid_derivative(cache['z1'])  # Apply derivative of activation
    dW1 = np.dot(cache['X'].T, dz1) / m  # Gradient for W1
    db1 = np.mean(dz1, axis=0, keepdims=True)  # Gradient for b1
    
    # Update weights using gradient descent
    weights['W1'] -= learning_rate * dW1
    weights['b1'] -= learning_rate * db1
    weights['W2'] -= learning_rate * dW2
    weights['b2'] -= learning_rate * db2
    
    return weights

def train_mlp(X, y, hidden_size=4, learning_rate=1.0, epochs=1000, print_progress=True):
    """
    Train a multi-layer perceptron
    
    Parameters:
    - X: input data
    - y: target labels
    - hidden_size: number of neurons in hidden layer
    - learning_rate: learning rate for gradient descent
    - epochs: number of training iterations
    - print_progress: whether to print cost during training
    
    Returns:
    - Trained weights
    - Cost history
    """
    # Initialize network
    input_size = X.shape[1]
    output_size = 1
    weights = initialize_mlp_weights(input_size, hidden_size, output_size)
    
    cost_history = []
    
    if print_progress:
        print(f"Training MLP: {input_size} inputs -> {hidden_size} hidden -> {output_size} output")
        print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        cache = mlp_forward_pass(X, weights)
        
        # Compute cost
        cost = compute_cost(y, cache['a2'])
        cost_history.append(cost)
        
        # Backward pass
        weights = mlp_backward_pass(cache, weights, y, learning_rate)
        
        # Print progress
        if print_progress and (epoch % 400 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}, Cost: {cost:.6f}")
    
    return weights, cost_history

def mlp_predict(X, weights, threshold=0.5):
    """
    Make predictions using trained MLP
    
    Parameters:
    - X: input data
    - weights: trained weights
    - threshold: decision threshold (default 0.5)
    
    Returns:
    - Binary predictions (0 or 1) and probabilities
    """
    cache = mlp_forward_pass(X, weights)
    probabilities = cache['a2']
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities

# ==========================================
# XOR PROBLEM SOLVER
# ==========================================

def solve_xor_problem():
    """
    Demonstrates how MLP can solve the XOR problem
    that single perceptron cannot handle
    """
    print("=" * 50)
    print("SOLVING XOR PROBLEM WITH MULTI-LAYER PERCEPTRON")
    print("=" * 50)
    
    # XOR dataset - the classic non-linearly separable problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR truth table
    
    print("XOR Truth Table:")
    print("Input A | Input B | Expected Output")
    print("--------|---------|----------------")
    for i in range(len(X)):
        print(f"   {int(X[i][0])}    |    {int(X[i][1])}    |        {int(y[i][0])}")
    
    print("\nWhy XOR is challenging:")
    print("- XOR is NOT linearly separable")
    print("- Cannot draw a single straight line to separate the classes")
    print("- Requires non-linear decision boundary")
    print("- This is where multi-layer networks excel!")
    
    # Train MLP to solve XOR
    print("\\nTraining Multi-Layer Perceptron...")
    weights, cost_history = train_mlp(X, y, hidden_size=4, learning_rate=5.0, epochs=2000)
    
    # Test the trained network
    predictions, probabilities = mlp_predict(X, weights)
    
    print("\\nResults after training:")
    print("Input -> Expected -> Predicted -> Probability -> Correct?")
    print("-" * 55)
    
    correct_count = 0
    for i in range(len(X)):
        expected = int(y[i][0])
        predicted = predictions[i][0]
        probability = probabilities[i][0]
        is_correct = predicted == expected
        correct_count += is_correct
        
        print(f"[{int(X[i][0])}, {int(X[i][1])}] ->    {expected}     ->     {predicted}      ->   {probability:.4f}    -> {'✓' if is_correct else '✗'}")
    
    accuracy = correct_count / len(X) * 100
    print(f"\\nFinal Accuracy: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("SUCCESS! The MLP has learned the XOR function perfectly!")
    else:
        print("The MLP is still learning. Try increasing epochs or adjusting learning rate.")
    
    return weights, cost_history, X, y

def analyze_hidden_layer(X, weights):
    """
    Analyze what the hidden layer has learned
    """
    print("\\n" + "="*50)
    print("ANALYZING HIDDEN LAYER REPRESENTATIONS")
    print("="*50)
    
    print("Hidden layer transforms the input space to make it linearly separable")
    print("\\nHidden layer activations for each input:")
    print("Input -> Hidden Layer Activations")
    print("-" * 35)
    
    for i, x in enumerate(X):
        cache = mlp_forward_pass(x, weights)
        hidden_activations = cache['a1'][0]  # Get first (and only) sample
        print(f"[{int(x[0])}, {int(x[1])}] -> [{hidden_activations[0]:.3f}, {hidden_activations[1]:.3f}, {hidden_activations[2]:.3f}, {hidden_activations[3]:.3f}]")
    
    print("\\nKey insight: The hidden layer creates new feature representations")
    print("that make the XOR problem linearly separable in the hidden space!")

def plot_training_progress(cost_history):
    """
    Plot the cost function during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title('MLP Training Progress on XOR Problem', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(cost_history))
    
    # Add annotations
    if len(cost_history) > 100:
        plt.annotate('Learning starts', 
                    xy=(0, cost_history[0]), xytext=(len(cost_history)*0.2, cost_history[0]*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.annotate(f'Final cost: {cost_history[-1]:.4f}', 
                    xy=(len(cost_history)-1, cost_history[-1]), 
                    xytext=(len(cost_history)*0.7, cost_history[-1]*2),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
    
    plt.tight_layout()
    plt.show()

def visualize_decision_boundary_2d(X, y, weights):
    """
    Visualize the decision boundary learned by the MLP
    """
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    _, mesh_probs = mlp_predict(mesh_points, weights)
    mesh_probs = mesh_probs.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, mesh_probs, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    colors = ['red', 'blue']
    labels = ['XOR = 0', 'XOR = 1']
    for i in range(2):
        mask = (y.flatten() == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=200, 
                   label=labels[i], edgecolors='black', linewidth=2)
    
    # Add input labels
    for i, (x_coord, y_coord) in enumerate(X):
        plt.annotate(f'({int(x_coord)},{int(y_coord)})', 
                    (x_coord, y_coord), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input A', fontsize=12)
    plt.ylabel('Input B', fontsize=12)
    plt.title('MLP Decision Boundary for XOR Problem\\n(Non-linear boundary separates the classes)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """
    Main function to demonstrate XOR solution with MLP
    """
    print("MULTI-LAYER PERCEPTRON - XOR PROBLEM SOLVER")
    print("=" * 60)
    
    # Solve XOR problem
    weights, cost_history, X, y = solve_xor_problem()
    
    # Analyze what the hidden layer learned
    analyze_hidden_layer(X, weights)
    
    # Plot training progress and decision boundary
    try:
        print("\\nGenerating visualizations...")
        
        # Plot training progress
        plot_training_progress(cost_history)
        
        # Plot decision boundary
        visualize_decision_boundary_2d(X, y, weights)
        
    except Exception as e:
        print(f"\\nVisualization not available: {e}")
        print("Install matplotlib to see training progress and decision boundary plots")
    
    # Summary
    print("\\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("1. MLP can solve non-linearly separable problems like XOR")
    print("2. Hidden layer creates new feature representations")
    print("3. Non-linear activation functions enable complex mappings")
    print("4. Backpropagation efficiently trains the network")
    print("5. Decision boundary is non-linear (curved)")
    print("="*60)
    
    print("\\nNext steps:")
    print("- Try different hidden layer sizes")
    print("- Experiment with different learning rates")
    print("- Test on other non-linear problems")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()

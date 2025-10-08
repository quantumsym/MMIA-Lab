#!/usr/bin/env python

# ==========================================
# BINARY CLASSIFICATION WITH MLP
# Demonstrates practical machine learning workflow
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def sigmoid(x):
    """
    Sigmoid activation function: transforms any real number to (0,1) range
    Perfect for binary classification output layer
    """
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function - needed for backpropagation
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """
    ReLU activation function: max(0, x)
    Popular choice for hidden layers
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU function
    """
    return np.where(x > 0, 1, 0)

# ==========================================
# DATA GENERATION
# ==========================================

def generate_classification_dataset(n_samples=400, noise=0.1, random_seed=42):
    """
    Generate a 2D binary classification dataset
    Creates two classes that are separable but not linearly
    
    Parameters:
    - n_samples: total number of samples
    - noise: amount of noise to add
    - random_seed: for reproducible results
    
    Returns:
    - X: feature matrix (n_samples, 2)
    - y: labels (n_samples, 1)
    """
    np.random.seed(random_seed)
    
    # Generate two distinct clusters
    n_per_class = n_samples // 2
    
    # Class 0: two small clusters
    X_class0_1 = np.random.randn(n_per_class//2, 2) * 0.5 + [1, 1]
    X_class0_2 = np.random.randn(n_per_class//2, 2) * 0.5 + [3, 3]
    X_class0 = np.vstack([X_class0_1, X_class0_2])
    y_class0 = np.zeros((n_per_class, 1))
    
    # Class 1: crescent shape around class 0
    angles = np.random.uniform(0, 2*np.pi, n_per_class)
    radii = np.random.uniform(2.5, 3.5, n_per_class)
    X_class1 = np.column_stack([
        radii * np.cos(angles) + 2,
        radii * np.sin(angles) + 2
    ])
    y_class1 = np.ones((n_per_class, 1))
    
    # Combine classes
    X = np.vstack([X_class0, X_class1])
    y = np.vstack([y_class0, y_class1])
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Shuffle the dataset
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

def split_dataset(X, y, train_ratio=0.8):
    """
    Split dataset into training and testing sets
    
    Parameters:
    - X: features
    - y: labels
    - train_ratio: fraction of data for training
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # Simple split (data is already shuffled)
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    return X_train, X_test, y_train, y_test

# ==========================================
# MLP IMPLEMENTATION
# ==========================================

def initialize_network(input_size, hidden_sizes, output_size):
    """
    Initialize a multi-layer network with variable hidden layers
    
    Parameters:
    - input_size: number of input features
    - hidden_sizes: list of hidden layer sizes [e.g., [8, 4]]
    - output_size: number of output neurons
    
    Returns:
    - weights: dictionary with all network parameters
    """
    weights = {}
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Initialize weights for each layer
    for i in range(len(layer_sizes) - 1):
        # Xavier initialization for better training
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
            weights[f'W{i+1}'] = np.random.uniform(-limit, limit, (fan_in, fan_out))
        weights[f'b{i+1}'] = np.zeros((1, fan_out))
    
    return weights

def forward_pass(X, weights, use_relu=True):
    """
    Forward pass through the network
    
    Parameters:
    - X: input data
    - weights: network parameters
    - use_relu: whether to use ReLU in hidden layers
    
    Returns:
    - cache: dictionary with all intermediate values
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    cache = {'a0': X}  # Input layer
    
    # Determine number of layers
    n_layers = len([k for k in weights.keys() if k.startswith('W')])
    
    # Forward pass through each layer
    for i in range(1, n_layers + 1):
        # Linear transformation
        z = np.dot(cache[f'a{i-1}'], weights[f'W{i}']) + weights[f'b{i}']
        cache[f'z{i}'] = z
        
        # Apply activation function
        if i == n_layers:  # Output layer
            cache[f'a{i}'] = sigmoid(z)
        else:  # Hidden layers
            if use_relu:
                cache[f'a{i}'] = relu(z)
            else:
                cache[f'a{i}'] = sigmoid(z)
    
    return cache

def compute_loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary cross-entropy
    loss = -np.mean(y_true * np.log(y_pred_clipped) + 
                    (1 - y_true) * np.log(1 - y_pred_clipped))
    
    return loss

def backward_pass(cache, weights, y_true, use_relu=True):
    """
    Backward pass (backpropagation)
    
    Parameters:
    - cache: forward pass results
    - weights: current network parameters
    - y_true: true labels
    - use_relu: whether ReLU was used in hidden layers
    
    Returns:
    - gradients: dictionary with gradients for all parameters
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    
    m = y_true.shape[0]  # batch size
    gradients = {}
    
    # Determine number of layers
    n_layers = len([k for k in weights.keys() if k.startswith('W')])
    
    # Output layer gradients
    dz = cache[f'a{n_layers}'] - y_true
    gradients[f'dW{n_layers}'] = np.dot(cache[f'a{n_layers-1}'].T, dz) / m
    gradients[f'db{n_layers}'] = np.mean(dz, axis=0, keepdims=True)
    
    # Hidden layer gradients (backpropagate)
    for i in range(n_layers - 1, 0, -1):
        # Compute da (derivative w.r.t. activation)
        da = np.dot(dz, weights[f'W{i+1}'].T)
        
        # Compute dz (derivative w.r.t. pre-activation)
        if use_relu:
            dz = da * relu_derivative(cache[f'z{i}'])
        else:
            dz = da * sigmoid_derivative(cache[f'z{i}'])
        
        # Compute parameter gradients
        gradients[f'dW{i}'] = np.dot(cache[f'a{i-1}'].T, dz) / m
        gradients[f'db{i}'] = np.mean(dz, axis=0, keepdims=True)
    
    return gradients

def update_parameters(weights, gradients, learning_rate):
    """
    Update network parameters using gradient descent
    """
    for key in weights.keys():
        if key.startswith('W'):
            weights[key] -= learning_rate * gradients[f'd{key}']
        elif key.startswith('b'):
            weights[key] -= learning_rate * gradients[f'd{key}']
    
    return weights

def train_network(X_train, y_train, X_test, y_test, 
                 hidden_sizes=[8, 4], learning_rate=0.1, epochs=1000, 
                 use_relu=True, print_progress=True):
    """
    Complete training loop for the neural network
    
    Parameters:
    - X_train, y_train: training data
    - X_test, y_test: test data for validation
    - hidden_sizes: list of hidden layer sizes
    - learning_rate: learning rate for gradient descent
    - epochs: number of training epochs
    - use_relu: whether to use ReLU activation
    - print_progress: whether to print training progress
    
    Returns:
    - weights: trained network parameters
    - history: training history (loss, accuracy)
    """
    # Initialize network
    input_size = X_train.shape[1]
    output_size = 1
    weights = initialize_network(input_size, hidden_sizes, output_size)
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    activation_name = "ReLU" if use_relu else "Sigmoid"
    
    if print_progress:
        print(f"Training Neural Network:")
        print(f"Architecture: {input_size} -> {' -> '.join(map(str, hidden_sizes))} -> {output_size}")
        print(f"Hidden activation: {activation_name}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print("-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        cache = forward_pass(X_train, weights, use_relu)
        train_predictions = cache[f'a{len(hidden_sizes) + 1}']
        
        # Compute training loss
        train_loss = compute_loss(y_train, train_predictions)
        
        # Backward pass
        gradients = backward_pass(cache, weights, y_train, use_relu)
        
        # Update parameters
        weights = update_parameters(weights, gradients, learning_rate)
        
        # Evaluate on test set
        test_cache = forward_pass(X_test, weights, use_relu)
        test_predictions = test_cache[f'a{len(hidden_sizes) + 1}']
        test_loss = compute_loss(y_test, test_predictions)
        
        # Compute accuracies
        train_accuracy = compute_accuracy(y_train, train_predictions)
        test_accuracy = compute_accuracy(y_test, test_predictions)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)
        
        # Print progress
        if print_progress and (epoch % 200 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")
    
    return weights, history

def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
    Compute classification accuracy
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    
    predictions = (y_pred > threshold).astype(int)
    accuracy = np.mean(predictions == y_true) * 100
    
    return accuracy

def make_predictions(X, weights, use_relu=True, threshold=0.5):
    """
    Make predictions on new data
    """
    cache = forward_pass(X, weights, use_relu)
    n_layers = len([k for k in weights.keys() if k.startswith('W')])
    probabilities = cache[f'a{n_layers}']
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def plot_dataset(X, y, title="Binary Classification Dataset"):
    """
    Plot the 2D dataset
    """
    plt.figure(figsize=(10, 8))
    
    # Plot points by class
    colors = ['red', 'blue']
    labels = ['Class 0', 'Class 1']
    
    for i in range(2):
        mask = (y.flatten() == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=50, 
                   alpha=0.7, label=labels[i], edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training history (loss and accuracy)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(history['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, weights, use_relu=True, title="Decision Boundary"):
    """
    Plot the decision boundary learned by the network
    """
    plt.figure(figsize=(12, 10))
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    _, mesh_probs = make_predictions(mesh_points, weights, use_relu)
    mesh_probs = mesh_probs.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, mesh_probs, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    colors = ['red', 'blue']
    labels = ['Class 0', 'Class 1']
    
    for i in range(2):
        mask = (y.flatten() == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, 
                   alpha=0.9, label=labels[i], edgecolors='black', linewidth=1)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN DEMONSTRATION
# ==========================================

def main():
    """
    Complete binary classification demonstration
    """
    print("BINARY CLASSIFICATION WITH MULTI-LAYER PERCEPTRON")
    print("=" * 60)
    
    # Generate dataset
    print("1. Generating dataset...")
    X, y = generate_classification_dataset(n_samples=400, noise=0.1)
    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    print(f"   Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    
    # Split dataset
    print("\\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, train_ratio=0.8)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Visualize dataset
    try:
        print("\\n3. Visualizing dataset...")
        plot_dataset(X, y, "Binary Classification Dataset")
    except:
        print("   (Visualization skipped - matplotlib not available)")
    
    # Train network with ReLU
    print("\\n4. Training neural network with ReLU activation...")
    weights_relu, history_relu = train_network(
        X_train, y_train, X_test, y_test,
        hidden_sizes=[12, 8, 4],
        learning_rate=0.01,
        epochs=1000,
        use_relu=True,
        print_progress=True
    )
    
    # Train network with Sigmoid (for comparison)
    print("\\n5. Training neural network with Sigmoid activation...")
    weights_sigmoid, history_sigmoid = train_network(
        X_train, y_train, X_test, y_test,
        hidden_sizes=[12, 8, 4],
        learning_rate=0.1,
        epochs=1000,
        use_relu=False,
        print_progress=True
    )
    
    # Final evaluation
    print("\\n6. Final Evaluation:")
    print("-" * 30)
    
    # ReLU network
    _, train_probs_relu = make_predictions(X_train, weights_relu, use_relu=True)
    _, test_probs_relu = make_predictions(X_test, weights_relu, use_relu=True)
    train_acc_relu = compute_accuracy(y_train, train_probs_relu)
    test_acc_relu = compute_accuracy(y_test, test_probs_relu)
    
    print(f"ReLU Network:")
    print(f"  Training Accuracy: {train_acc_relu:.2f}%")
    print(f"  Test Accuracy: {test_acc_relu:.2f}%")
    
    # Sigmoid network
    _, train_probs_sigmoid = make_predictions(X_train, weights_sigmoid, use_relu=False)
    _, test_probs_sigmoid = make_predictions(X_test, weights_sigmoid, use_relu=False)
    train_acc_sigmoid = compute_accuracy(y_train, train_probs_sigmoid)
    test_acc_sigmoid = compute_accuracy(y_test, test_probs_sigmoid)
    
    print(f"\\nSigmoid Network:")
    print(f"  Training Accuracy: {train_acc_sigmoid:.2f}%")
    print(f"  Test Accuracy: {test_acc_sigmoid:.2f}%")
    
    # Visualizations
    try:
        print("\\n7. Generating visualizations...")
        
        # Plot training history comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ReLU training history
        axes[0,0].plot(history_relu['train_loss'], 'b-', label='Train')
        axes[0,0].plot(history_relu['test_loss'], 'r-', label='Test')
        axes[0,0].set_title('ReLU Network - Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(history_relu['train_accuracy'], 'b-', label='Train')
        axes[0,1].plot(history_relu['test_accuracy'], 'r-', label='Test')
        axes[0,1].set_title('ReLU Network - Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Sigmoid training history
        axes[1,0].plot(history_sigmoid['train_loss'], 'b-', label='Train')
        axes[1,0].plot(history_sigmoid['test_loss'], 'r-', label='Test')
        axes[1,0].set_title('Sigmoid Network - Loss')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(history_sigmoid['train_accuracy'], 'b-', label='Train')
        axes[1,1].plot(history_sigmoid['test_accuracy'], 'r-', label='Test')
        axes[1,1].set_title('Sigmoid Network - Accuracy')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Accuracy (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot decision boundaries
        plot_decision_boundary(X, y, weights_relu, use_relu=True, 
                             title="Decision Boundary - ReLU Network")
        plot_decision_boundary(X, y, weights_sigmoid, use_relu=False, 
                             title="Decision Boundary - Sigmoid Network")
        
    except Exception as e:
        print(f"   Visualization error: {e}")
    
    # Summary and insights
    print("\\n" + "="*60)
    print("KEY INSIGHTS:")
    print("1. Neural networks can learn complex non-linear decision boundaries")
    print("2. ReLU activation often trains faster than sigmoid")
    print("3. Multiple hidden layers enable more complex representations")
    print("4. Proper train/test split helps evaluate generalization")
    print("5. Monitoring both loss and accuracy reveals training dynamics")
    print("="*60)
    
    print("\\nExperiment suggestions:")
    print("- Try different network architectures")
    print("- Vary the learning rate")
    print("- Add more training data")
    print("- Experiment with different activation functions")

if __name__ == "__main__":
    main()

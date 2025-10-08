#!/usr/bin/env python

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simple Perceptron Implementation for Logical Functions
def perceptron_train(X, y, learning_rate=0.1, epochs=100):
    """
    Train a simple perceptron using the perceptron learning rule.
    
    Args:
        X: Input features (n_samples x n_features)
        y: Target labels (n_samples,)
        learning_rate: Learning rate for weight updates
        epochs: Number of training epochs
    
    Returns:
        weights: Final weights including bias
        errors: List of errors per epoch
    """
    # Initialize weights randomly (including bias weight)
    n_features = X.shape[1]
    weights = np.random.randn(n_features + 1) * 0.1
    errors = []
    
    for epoch in range(epochs):
        epoch_errors = 0
        for i in range(len(X)):
            # Add bias term (x0 = 1) to input
            x_with_bias = np.concatenate([[1], X[i]])
            
            # Forward pass: compute weighted sum and activation
            net_input = np.dot(weights, x_with_bias)
            prediction = 1 if net_input >= 0 else 0
            
            # Calculate error and update weights if needed
            error = y[i] - prediction
            if error != 0:
                weights += learning_rate * error * x_with_bias
                epoch_errors += abs(error)
        
        errors.append(epoch_errors)
        
        # Early stopping if no errors
        if epoch_errors == 0:
            break
    
    return weights, errors

def predict_perceptron(X, weights):
    """Make predictions using trained perceptron weights."""
    predictions = []
    for i in range(len(X)):
        x_with_bias = np.concatenate([[1], X[i]])
        net_input = np.dot(weights, x_with_bias)
        prediction = 1 if net_input >= 0 else 0
        predictions.append(prediction)
    return np.array(predictions)

# Define logical function datasets
def create_logical_data():
    """Create datasets for AND, OR, and XOR logical functions."""
    # Input patterns for 2-bit logical functions
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Target outputs for each function
    y_and = np.array([0, 0, 0, 1])  # AND function
    y_or = np.array([0, 1, 1, 1])   # OR function  
    y_xor = np.array([0, 1, 1, 0])  # XOR function (not linearly separable)
    
    return X, y_and, y_or, y_xor

# Train perceptrons on logical functions
X, y_and, y_or, y_xor = create_logical_data()

print("Training Perceptron on Logical Functions")
print("=" * 50)

# Train on AND function
print("\n1. AND Function Training:")
weights_and, errors_and = perceptron_train(X, y_and, epochs=50)
pred_and = predict_perceptron(X, weights_and)
print(f"Final weights: {weights_and}")
print(f"Predictions: {pred_and}")
print(f"Targets:     {y_and}")
print(f"Accuracy: {np.mean(pred_and == y_and):.2%}")

# Train on OR function  
print("\n2. OR Function Training:")
weights_or, errors_or = perceptron_train(X, y_or, epochs=50)
pred_or = predict_perceptron(X, weights_or)
print(f"Final weights: {weights_or}")
print(f"Predictions: {pred_or}")
print(f"Targets:     {y_or}")
print(f"Accuracy: {np.mean(pred_or == y_or):.2%}")

# Train on XOR function (will fail!)
print("\n3. XOR Function Training (Expected to Fail):")
weights_xor, errors_xor = perceptron_train(X, y_xor, epochs=50)
pred_xor = predict_perceptron(X, weights_xor)
print(f"Final weights: {weights_xor}")
print(f"Predictions: {pred_xor}")
print(f"Targets:     {y_xor}")
print(f"Accuracy: {np.mean(pred_xor == y_xor):.2%}")

# Create visualization showing training errors
fig = go.Figure()

# Add error curves for each function
fig.add_trace(go.Scatter(
    x=list(range(len(errors_and))),
    y=errors_and,
    mode='lines+markers',
    name='AND Function',
    line=dict(color='#1FB8CD', width=3),
    marker=dict(size=6)
))

fig.add_trace(go.Scatter(
    x=list(range(len(errors_or))),
    y=errors_or,
    mode='lines+markers',
    name='OR Function',
    line=dict(color='#2E8B57', width=3),
    marker=dict(size=6)
))

fig.add_trace(go.Scatter(
    x=list(range(len(errors_xor))),
    y=errors_xor,
    mode='lines+markers',
    name='XOR (Fails)',
    line=dict(color='#DB4545', width=3),
    marker=dict(size=6)
))

# Update layout
fig.update_layout(
    title="Perceptron Learning: Logic Functions",
    xaxis_title="Epoch",
    yaxis_title="Training Errors",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

# Save the chart
fig.write_image("perceptron_logic_training.png")

print(f"\n" + "=" * 50)
print("KEY INSIGHTS:")
print("- Simple perceptron can learn linearly separable functions (AND, OR)")
print("- XOR function is NOT linearly separable - perceptron cannot solve it")
print("- This limitation led to the development of multi-layer perceptrons (MLPs)")
print("- Chart saved as 'perceptron_logic_training.png'")

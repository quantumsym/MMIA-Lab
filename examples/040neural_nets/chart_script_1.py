#!/usr/bin/env python

import numpy as np
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    """Initialize weights and biases for the neural network"""
    # Xavier initialization for better convergence
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2

def forward_pass(X, W1, b1, W2, b2):
    """Perform forward propagation through the network"""
    # Hidden layer calculation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Output layer calculation
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    return z1, a1, z2, a2

def backward_pass(X, y, z1, a1, z2, a2, W1, W2):
    """Perform backpropagation to calculate gradients"""
    m = X.shape[0]  # Number of training examples
    
    # Calculate output layer error
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    # Calculate hidden layer error
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

def calculate_loss(y_true, y_pred):
    """Calculate binary cross-entropy loss"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Define XOR dataset
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1], 
              [0]])

# Network parameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 5.0
epochs = 5000

# Initialize weights and biases
W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

# Training loop with loss tracking
losses = []
epoch_list = []

for epoch in range(epochs):
    # Forward pass
    z1, a1, z2, a2 = forward_pass(X, W1, b1, W2, b2)
    
    # Calculate loss
    loss = calculate_loss(y, a2)
    
    # Store loss every 50 epochs for plotting
    if epoch % 50 == 0:
        losses.append(loss)
        epoch_list.append(epoch)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(X, y, z1, a1, z2, a2, W1, W2)
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Final predictions
_, _, _, final_predictions = forward_pass(X, W1, b1, W2, b2)

# Create the training loss visualization
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=epoch_list,
    y=losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='#1FB8CD', width=3),
    hovertemplate='<b>Epoch</b>: %{x}<br><b>Loss</b>: %{y:.4f}<extra></extra>'
))

# Update layout
fig.update_layout(
    title='MLP Training on XOR Problem',
    xaxis_title='Epoch',
    yaxis_title='Loss'
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

# Save the chart
fig.write_image('xor_mlp_training.png')

# Print final results for verification
print("Final XOR Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} | Target: {y[i][0]} | Predicted: {final_predictions[i][0]:.4f}")

#!/usr/bin/env python3
"""
MLP for Linear Regression using Flax NNX (Modern API)

This script demonstrates a Multi-Layer Perceptron for regression using the
modern Flax NNX API, which simplifies state management compared to Linen.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import math

# 1. Data Preparation
# Generate training data: sine function over [-π, π]
key = jax.random.key(0)
x_train = jnp.linspace(-math.pi, math.pi, 2000).reshape(-1, 1)
y_train = jnp.sin(x_train)

# 2. Model Definition using Flax NNX
# NNX uses a stateful approach where the model directly contains its parameters
class SimpleMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        """
        Initialize the MLP with two dense layers.
        
        Args:
            rngs: Random number generator state for parameter initialization
        """
        # NNX modules are stateful: parameters are stored as attributes
        self.dense1 = nnx.Linear(in_features=1, out_features=64, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=64, out_features=1, rngs=rngs)
    
    def __call__(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1)
        
        Returns:
            Output predictions of shape (batch_size, 1)
        """
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        return x

# 3. Initialization
# Create model instance (parameters are initialized automatically)
model = SimpleMLP(rngs=nnx.Rngs(0))

# Create optimizer with its state
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))

# 4. Define the training step
# NNX allows direct mutation of model parameters, simplifying the code
@nnx.jit  # JIT-compile for high performance
def train_step(model, optimizer, batch_x, batch_y):
    """
    Perform one training step: compute loss, gradients, and update parameters.
    
    Args:
        model: The neural network model
        optimizer: The optimizer managing parameter updates
        batch_x: Input batch
        batch_y: Target batch
    
    Returns:
        loss: The computed loss value
    """
    def loss_fn(model):
        # Forward pass
        preds = model(batch_x)
        # Mean squared error loss
        loss = jnp.mean((preds - batch_y) ** 2)
        return loss
    
    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Update parameters (in-place mutation)
    optimizer.update(grads)
    
    return loss

# 5. Training Loop
print("Training MLP with Flax NNX...")
print("=" * 60)

for epoch in range(2000):
    loss = train_step(model, optimizer, x_train, y_train)
    
    if epoch % 100 == 99:
        print(f'Epoch {epoch+1:4d}, Loss: {loss:.6f}')

print("=" * 60)
print("Training completed!")

# 6. Evaluation (optional)
# Make predictions on test data
x_test = jnp.linspace(-math.pi, math.pi, 100).reshape(-1, 1)
y_pred = model(x_test)
y_true = jnp.sin(x_test)
test_loss = jnp.mean((y_pred - y_true) ** 2)
print(f"Test Loss: {test_loss:.6f}")

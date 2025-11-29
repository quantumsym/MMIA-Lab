# jax_example.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import math
from flax.training import train_state

# 1. Data Preparation
key = jax.random.PRNGKey(0)
x_train = jnp.linspace(-math.pi, math.pi, 2000).reshape(-1, 1)
y_train = jnp.sin(x_train)

# 2. Model Definition using Flax
class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # The __call__ method defines the model logic
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# 3. Initialization
model = SimpleMLP()
params = model.init(key, x_train)['params']
optimizer = optax.adam(learning_rate=1e-3)

# Create a TrainState to manage parameters and optimizer state
# This is a common pattern in Flax for handling state functionally
class TrainState(train_state.TrainState):
    pass

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# 4. Define the training step as a pure function
@jax.jit # JIT-compile the training step for high performance
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_x)
        loss = jnp.mean((preds - batch_y) ** 2)
        return loss

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)
    return state, loss

# 5. Training Loop
for epoch in range(2000):
    state, loss = train_step(state, x_train, y_train)
    if epoch % 100 == 99:
        print(f'Epoch {epoch+1}, Loss: {loss}')


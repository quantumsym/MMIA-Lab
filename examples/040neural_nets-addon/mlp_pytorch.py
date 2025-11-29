# pytorch_example.py
import torch
import torch.nn as nn
import numpy as np
import math

# 1. Data Preparation
x_train = torch.linspace(-math.pi, math.pi, 2000).unsqueeze(1)
y_train = torch.sin(x_train)

# 2. Model Definition using the nn.Module class
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):
        # The forward pass defines the computation graph dynamically
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# 3. Model Initialization, Loss, and Optimizer
model = SimpleMLP()
loss_function = nn.MSELoss() # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
for epoch in range(2000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_train)

    # Compute loss.
    loss = loss_function(y_pred, y_train)

    if epoch % 100 == 99:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable parameters.
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters.
    optimizer.step()


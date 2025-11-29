# tensorflow_example.py
import tensorflow as tf
import numpy as np
import math

# 1. Data Preparation
x_train = tf.linspace(-math.pi, math.pi, 2000)
x_train = tf.reshape(x_train, [-1, 1])
y_train = tf.sin(x_train)

# 2. Model Definition using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# 3. Model Compilation - specify optimizer, loss, and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error for monitoring
)

# 4. Training using the high-level fit() method
history = model.fit(
    x_train, y_train,
    epochs=2000,
    verbose=0,  # Suppress output for cleaner display
    validation_split=0.1  # Use 10% of data for validation
)

# Print final training loss
print(f'Final training loss: {history.history["loss"][-1]:.6f}')

# Alternative: Custom training loop with tf.function for optimization
@tf.function  # JIT-compile the training step for better performance
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = tf.keras.losses.mse(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Custom training loop (alternative to model.fit())
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
for epoch in range(100):
    loss = train_step(x_train, y_train)
    if epoch % 20 == 19:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.6f}')

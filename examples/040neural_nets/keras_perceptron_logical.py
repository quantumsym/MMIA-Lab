#!/usr/bin/env python

# keras_perceptron_logical.py
# ==========================================
# SINGLE PERCEPTRON WITH KERAS
# Tests learning of logical AND, OR, and failure on XOR
# ==========================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_perceptron():
    """
    Builds a single-layer perceptron model with sigmoid activation.
    Input shape: 2, Output shape: 1
    """
    model = keras.Sequential([
        layers.Dense(1, input_shape=(2,), activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def test_logical_function(X, y, function_name, epochs=100):
    """
    Trains the perceptron on a logical function and reports results.
    """
    print(f"\nLogical function: {function_name}")
    print(f"Truth table:\n{np.hstack([X, y])}")

    model = build_perceptron()
    history = model.fit(X, y, epochs=epochs, verbose=0)
    predictions = (model.predict(X) > 0.5).astype(int)

    print("Input | Target | Predicted | Correct")
    for i in range(len(X)):
        print(f"{X[i]} | {int(y[i][0])}      | {predictions[i][0]}        | {'✓' if predictions[i][0]==y[i][0] else '✗'}")

    accuracy = np.mean(predictions == y)
    print(f"Accuracy on {function_name}: {accuracy*100:.1f}%")
    return accuracy

def main():
    # Input patterns for binary functions
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    # Targets for each logic gate
    y_and = np.array([[0],[0],[0],[1]], dtype=np.float32)
    y_or  = np.array([[0],[1],[1],[1]], dtype=np.float32)
    y_xor = np.array([[0],[1],[1],[0]], dtype=np.float32)

    print("== KERAS PERCEPTRON ON LOGICAL FUNCTIONS ==")
    _ = test_logical_function(X, y_and, "AND")
    _ = test_logical_function(X, y_or, "OR")
    print("\nNow testing XOR (should fail)...")
    _ = test_logical_function(X, y_xor, "XOR")

    print("\nKey takeaway:")
    print("A single-layer perceptron can learn linearly separable functions (AND, OR),")
    print("but CANNOT learn XOR which is not linearly separable.\n")

if __name__ == "__main__":
    main()


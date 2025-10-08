import numpy as np

# Simple procedural perceptron implementation for AND function

def step_activation(x):
    """
    Step activation function (Heaviside function)
    Returns 1 if x >= 0, else 0
    
    Args:
        x (float): Input to activation function
        
    Returns:
        int: 0 or 1
    """
    return 1 if x >= 0 else 0

def predict(inputs, weights, bias):
    """
    Make prediction for given inputs
    
    Args:
        inputs (array): Input features
        weights (array): Network weights
        bias (float): Network bias
        
    Returns:
        int: Predicted output (0 or 1)
    """
    # Calculate weighted sum plus bias
    weighted_sum = np.dot(inputs, weights) + bias
    # Apply activation function
    return step_activation(weighted_sum)

def train_perceptron(training_inputs, training_outputs, learning_rate=0.1, epochs=100, show_every=10):
    """
    Train the perceptron using the perceptron learning rule
    
    Args:
        training_inputs (array): Training input data
        training_outputs (array): Training target outputs
        learning_rate (float): Learning rate for weight updates
        epochs (int): Number of training epochs
        show_every (int): Show error every N epochs
        
    Returns:
        tuple: (final_weights, final_bias, error_history)
    """
    # Convert to numpy arrays
    training_inputs = np.array(training_inputs)
    training_outputs = np.array(training_outputs).flatten()
    
    # Initialize weights and bias randomly
    num_inputs = training_inputs.shape[1]
    weights = np.random.uniform(-0.5, 0.5, num_inputs)
    bias = np.random.uniform(-0.5, 0.5)
    
    print(f"Initial weights: {weights}")
    print(f"Initial bias: {bias}")
    
    error_history = []
    
    for epoch in range(epochs):
        total_error = 0
        
        # Train on each sample
        for i in range(len(training_inputs)):
            # Forward pass: make prediction
            prediction = predict(training_inputs[i], weights, bias)
            
            # Calculate error
            error = training_outputs[i] - prediction
            total_error += abs(error)
            
            # Update weights and bias using perceptron learning rule
            # w = w + learning_rate * error * input
            weights += learning_rate * error * training_inputs[i]
            bias += learning_rate * error
        
        error_history.append(total_error)
        
        # Show progress
        if (epoch + 1) % show_every == 0:
            print(f"Epoch {epoch + 1}: Total Error = {total_error}")
        
        # Early stopping if error is 0 (perfect classification)
        if total_error == 0:
            print(f"Perfect classification achieved at epoch {epoch + 1}")
            break
    
    return weights, bias, error_history

# Training data for AND function
# AND truth table: 0 AND 0 = 0, 0 AND 1 = 0, 1 AND 0 = 0, 1 AND 1 = 1
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_data = [0, 0, 0, 1]  # Expected outputs for AND function

print("=== Simple Procedural Perceptron for AND Function ===")
print("Training data:")
for i, (inp, target) in enumerate(zip(input_data, target_data)):
    print(f"Input: {inp} -> Target: {target}")

print("\nTraining the perceptron...")

# Train the network
final_weights, final_bias, error_history = train_perceptron(
    input_data, target_data, 
    learning_rate=0.1, 
    epochs=100, 
    show_every=10
)

print(f"\nFinal weights: {final_weights}")
print(f"Final bias: {final_bias}")

# Test the trained perceptron
print("\nTesting the trained perceptron:")
print("Input -> Prediction (Target)")
correct_predictions = 0

for i, (inp, target) in enumerate(zip(input_data, target_data)):
    prediction = predict(inp, final_weights, final_bias)
    is_correct = prediction == target
    if is_correct:
        correct_predictions += 1
    print(f"{inp} -> {prediction} ({target}) {'✓' if is_correct else '✗'}")

# Calculate final accuracy
accuracy = correct_predictions / len(input_data) * 100
print(f"\nFinal accuracy: {accuracy:.1f}%")

# Show decision boundary information
print(f"\nDecision boundary equation:")
print(f"{final_weights[0]:.3f} * x1 + {final_weights[1]:.3f} * x2 + {final_bias:.3f} = 0")
print(f"Output = 1 if {final_weights[0]:.3f} * x1 + {final_weights[1]:.3f} * x2 + {final_bias:.3f} >= 0")

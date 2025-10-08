#!/usr/bin/env python
"""
Functional NumPy implementation of a Perceptron for AND logic function
Non-object-oriented version equivalent to the original neurolab code
"""
import numpy as np

def step_function(x):
    """
    Step activation function (equivalent to neurolab's HardLim)
    
    Parameters:
    -----------
    x : float
        Input value to the activation function
    
    Returns:
    --------
    int : 1 if x >= 0, otherwise 0
    """
    return 1 if x >= 0 else 0

def forward_pass(weights, bias, input_vector):
    """
    Forward propagation: computes output for a single input
    
    Parameters:
    -----------
    weights : numpy array
        Weight vector [w1, w2]
    bias : float
        Bias term
    input_vector : numpy array or list
        Input vector [x1, x2]
    
    Returns:
    --------
    int : Perceptron output (0 or 1)
    """
    # Compute weighted sum: w1*x1 + w2*x2 + bias
    linear_output = np.dot(weights, input_vector) + bias
    
    # Apply step activation function
    return step_function(linear_output)

def train_perceptron(X, y, learning_rate=0.1, max_epochs=100, verbose=True):
    """
    Train perceptron using delta rule (equivalent to neurolab's net.train)
    
    This function implements the same delta learning algorithm used by neurolab.
    The perceptron learns by adjusting weights proportionally to the prediction error.
    
    Parameters:
    -----------
    X : list or numpy array, shape (n_samples, n_features)
        Training input data - each row is a sample, each column is a feature
    y : list or numpy array, shape (n_samples,)
        Target labels (0 or 1 for binary classification)
    learning_rate : float, default=0.1
        Learning rate - controls how much weights are adjusted per error
        (equivalent to neurolab's 'lr' parameter)
    max_epochs : int, default=100
        Maximum number of training epochs
        (equivalent to neurolab's 'epochs' parameter)
    verbose : bool, default=True
        If True, display training progress every 10 epochs
        (equivalent to neurolab's 'show=10' parameter)
    
    Returns:
    --------
    tuple : (final_weights, final_bias, error_history)
        final_weights : numpy array - learned weight vector
        final_bias : float - learned bias term
        error_history : list - number of errors per epoch
    """
    # Convert inputs to numpy arrays for easier manipulation
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    
    # Initialize parameters - start with zero weights and bias
    weights = np.zeros(n_features)  # Weight vector [w1, w2, ...]
    bias = 0.0                      # Bias term (threshold)
    error_history = []              # Track training progress
    
    if verbose:
        print(f"Training started: lr={learning_rate}, max_epochs={max_epochs}")
        print(f"Initial weights: {weights}")
        print(f"Initial bias: {bias}")
        print("-" * 50)
    
    # Main training loop - iterate through epochs
    for epoch in range(max_epochs):
        epoch_errors = 0  # Count errors in this epoch
        
        # Present all training samples to the network (one epoch)
        for i in range(n_samples):
            # Forward pass: compute prediction
            prediction = forward_pass(weights, bias, X[i])
            
            # Calculate prediction error
            error = y[i] - prediction
            
            # Delta rule: update weights only if there's an error
            if error != 0:
                # Weight update rule: w_new = w_old + learning_rate * error * input
                weights += learning_rate * error * X[i]
                
                # Bias update rule: b_new = b_old + learning_rate * error
                bias += learning_rate * error
                
                # Count this as an error for convergence tracking
                epoch_errors += 1
        
        # Store error count for this epoch
        error_history.append(epoch_errors)
        
        # Display progress every 10 epochs (like neurolab's show=10)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: {epoch_errors} errors - "
                  f"Weights: [{weights[0]:.3f}, {weights[1]:.3f}] - "
                  f"Bias: {bias:.3f}")
        
        # Check for convergence: no classification errors
        if epoch_errors == 0:
            if verbose:
                print(f"\n*** CONVERGENCE REACHED AT EPOCH {epoch + 1} ***")
                print(f"Final weights: [{weights[0]:.3f}, {weights[1]:.3f}]")
                print(f"Final bias: {bias:.3f}")
            break
    
    return weights, bias, error_history

def make_predictions(weights, bias, X):
    """
    Make predictions on new data using trained perceptron
    
    Parameters:
    -----------
    weights : numpy array
        Trained weight vector
    bias : float
        Trained bias term
    X : numpy array or list
        Input samples for prediction
    
    Returns:
    --------
    list or int : Predictions (0 or 1) for each sample
    """
    X = np.array(X)
    
    # Handle single sample vs multiple samples
    if X.ndim == 1:  # Single sample [x1, x2]
        return forward_pass(weights, bias, X)
    else:  # Multiple samples [[x1, x2], [x1, x2], ...]
        return [forward_pass(weights, bias, sample) for sample in X]

def evaluate_model(weights, bias, X_test, y_test):
    """
    Evaluate the trained perceptron and display results
    
    Parameters:
    -----------
    weights : numpy array
        Trained weight vector
    bias : float
        Trained bias term
    X_test : numpy array
        Test input data
    y_test : numpy array
        Test target labels
    
    Returns:
    --------
    float : Accuracy percentage
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Generate predictions
    predictions = make_predictions(weights, bias, X_test)
    
    # Display results table
    print("\nModel evaluation on test data:")
    print("Input    | Target | Prediction | Correct")
    print("---------+--------+------------+---------")
    
    correct_count = 0
    for i, (inp, target, pred) in enumerate(zip(X_test, y_test, predictions)):
        is_correct = (pred == target)
        correct_count += is_correct
        status_symbol = "✓" if is_correct else "✗"
        print(f"{list(inp)}  |   {target}    |     {pred}      |    {status_symbol}")
    
    # Calculate and display accuracy
    accuracy = (correct_count / len(X_test)) * 100
    print(f"\nModel Performance:")
    print(f"Correct predictions: {correct_count}/{len(X_test)}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Display learned decision boundary equation
    print(f"\nLearned decision rule:")
    print(f"output = step({weights[0]:.3f} × x1 + {weights[1]:.3f} × x2 + {bias:.3f})")
    print(f"where step(z) = 1 if z ≥ 0, else 0")
    
    return accuracy

def main():
    """
    Main function demonstrating perceptron training for AND logic gate
    Replicates the behavior of the original neurolab code
    """
    print("="*60)
    print("NUMPY PERCEPTRON IMPLEMENTATION FOR AND LOGIC GATE")
    print("Functional version - equivalent to neurolab code")
    print("="*60)
    
    # Define training data - same as original neurolab code
    # AND truth table: output is 1 only when both inputs are 1
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target_data = [0, 0, 0, 1]
    
    print("\nTraining Dataset - AND Logic Gate:")
    print("Input (x1, x2) | Target Output")
    print("---------------+--------------")
    for inp, target in zip(input_data, target_data):
        print(f"     {inp}      |      {target}")
    print()
    
    # Train the perceptron with same parameters as neurolab
    print("Starting perceptron training...")
    final_weights, final_bias, error_history = train_perceptron(
        X=input_data,
        y=target_data,
        learning_rate=0.1,    # Same as neurolab's lr=0.1
        max_epochs=100,       # Same as neurolab's epochs=100
        verbose=True          # Same as neurolab's show=10
    )
    
    # Evaluate the trained model
    accuracy = evaluate_model(final_weights, final_bias, input_data, target_data)
    
    # Display training statistics
    print(f"\nTraining Statistics:")
    print(f"Errors per epoch: {error_history}")
    print(f"Total training epochs: {len(error_history)}")
    print(f"Final model accuracy: {accuracy:.1f}%")
    
    # Return trained parameters for further use
    return final_weights, final_bias, error_history

def demonstrate_usage():
    """
    Additional demonstration of how to use the trained perceptron
    """
    print("\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60)
    
    # Train the perceptron
    weights, bias, _ = train_perceptron(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [0, 0, 0, 1],
        verbose=False  # Silent training
    )
    
    print("Trained perceptron can now make predictions:")
    
    # Test individual predictions
    test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for test_input in test_cases:
        prediction = make_predictions(weights, bias, test_input)
        print(f"AND{tuple(test_input)} = {prediction}")
    
    print(f"\nTrained parameters:")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")

# Execute the main demonstration
if __name__ == "__main__":
    # Run main training demonstration
    trained_weights, trained_bias, training_history = main()
    
    # Show additional usage examples
    demonstrate_usage()
    
    # Display comparison with original neurolab code
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL NEUROLAB CODE")
    print("="*60)
    print("""
ORIGINAL NEUROLAB IMPLEMENTATION:
---------------------------------
import neurolab as nl

# Logical AND data
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

# Create perceptron with 2 inputs and 1 neuron
net = nl.net.newp([[0, 1],[0, 1]], 1)

# Train with delta rule
error = net.train(input, target, epochs=100, show=10, lr=0.1)

EQUIVALENT NUMPY FUNCTIONAL VERSION:
------------------------------------
# Same training data
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_data = [0, 0, 0, 1]

# Train perceptron using delta rule
weights, bias, errors = train_perceptron(
    input_data, target_data,
    learning_rate=0.1,    # lr=0.1
    max_epochs=100,       # epochs=100
    verbose=True          # show=10
)

# Make predictions
predictions = make_predictions(weights, bias, input_data)

KEY EQUIVALENCES:
-----------------
• nl.net.newp()           ↔ train_perceptron() function
• net.train()             ↔ delta rule implementation
• HardLim activation      ↔ step_function()
• Weight initialization   ↔ np.zeros() for weights and bias
• Delta learning rule     ↔ w += lr * error * input
• Convergence detection   ↔ epoch_errors == 0
• Progress display        ↔ print every 10 epochs
    """)
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from multilayer_perceptron import (initialize_mlp_weights, mlp_forward_pass, 
                                  mlp_backward_pass)

# ============================================================================
# BINARY CLASSIFICATION WITH MLP - PRACTICAL EXAMPLE
# ============================================================================

def generate_classification_dataset(num_samples=200, noise_level=0.1, random_seed=42):
    """
    Generate a synthetic dataset for binary classification
    
    This function creates a realistic non-linearly separable dataset that simulates
    real-world classification problems. The dataset consists of two classes arranged
    in concentric patterns that cannot be separated by a straight line.
    
    Dataset structure:
    - Class 0: Points distributed in an inner circular region
    - Class 1: Points distributed in an outer annular (ring-shaped) region
    
    This pattern is common in real applications where:
    - Inner region represents one category (e.g., low-risk customers)
    - Outer region represents another category (e.g., high-risk customers)
    - The boundary between classes is circular, not linear
    
    Args:
        num_samples (int, optional): Total number of data points to generate. Default 200.
        noise_level (float, optional): Amount of random noise to add. Default 0.1.
                                     Higher values make classification more challenging.
        random_seed (int, optional): Seed for reproducible results. Default 42.
    
    Returns:
        tuple: (features, labels)
            features (numpy.ndarray): Input features, shape (num_samples, 2)
                Each row is one data point with [x, y] coordinates
            labels (numpy.ndarray): Binary class labels, shape (num_samples,)
                0 for inner class, 1 for outer class
    
    Note:
        The concentric pattern ensures the dataset is not linearly separable,
        making it a good test case for multi-layer perceptrons.
    """
    # Set random seed for reproducible results
    np.random.seed(random_seed)
    
    # Calculate samples per class (split evenly)
    samples_per_class = num_samples // 2
    
    # ========================================================================
    # GENERATE CLASS 0 (INNER CIRCULAR REGION)
    # ========================================================================
    
    # Generate random angles for polar coordinates
    # Full circle: 0 to 2π radians
    angles_class_0 = np.random.uniform(0, 2*np.pi, samples_per_class)
    
    # Generate random radii for inner circle
    # Radius between 0.5 and 1.5 creates inner circular region
    radii_class_0 = np.random.uniform(0.5, 1.5, samples_per_class)
    
    # Convert polar to Cartesian coordinates
    # x = r * cos(θ), y = r * sin(θ)
    x_class_0 = radii_class_0 * np.cos(angles_class_0)
    y_class_0 = radii_class_0 * np.sin(angles_class_0)
    
    # Add Gaussian noise to make the problem more realistic
    # Real-world data always has measurement noise
    x_class_0 += np.random.normal(0, noise_level, samples_per_class)
    y_class_0 += np.random.normal(0, noise_level, samples_per_class)
    
    # Combine x and y coordinates into feature matrix
    features_class_0 = np.column_stack([x_class_0, y_class_0])
    labels_class_0 = np.zeros(samples_per_class)  # All labeled as class 0
    
    # ========================================================================
    # GENERATE CLASS 1 (OUTER ANNULAR REGION)
    # ========================================================================
    
    # Generate random angles for outer ring
    angles_class_1 = np.random.uniform(0, 2*np.pi, samples_per_class)
    
    # Generate random radii for outer ring
    # Radius between 2.5 and 3.5 creates outer annular region
    # Gap between 1.5 and 2.5 separates the two classes
    radii_class_1 = np.random.uniform(2.5, 3.5, samples_per_class)
    
    # Convert polar to Cartesian coordinates
    x_class_1 = radii_class_1 * np.cos(angles_class_1)
    y_class_1 = radii_class_1 * np.sin(angles_class_1)
    
    # Add noise to outer class as well
    x_class_1 += np.random.normal(0, noise_level, samples_per_class)
    y_class_1 += np.random.normal(0, noise_level, samples_per_class)
    
    # Combine coordinates and create labels
    features_class_1 = np.column_stack([x_class_1, y_class_1])
    labels_class_1 = np.ones(samples_per_class)  # All labeled as class 1
    
    # ========================================================================
    # COMBINE CLASSES AND SHUFFLE
    # ========================================================================
    
    # Combine both classes into single dataset
    all_features = np.vstack([features_class_0, features_class_1])
    all_labels = np.hstack([labels_class_0, labels_class_1])
    
    # Shuffle the data to mix classes randomly
    # This is important for training - we don't want all class 0 samples first
    shuffle_indices = np.random.permutation(num_samples)
    shuffled_features = all_features[shuffle_indices]
    shuffled_labels = all_labels[shuffle_indices]
    
    return shuffled_features, shuffled_labels

def split_train_test(features, labels, test_ratio=0.3, random_seed=42):
    """
    Split dataset into training and testing sets
    
    Proper train/test splitting is crucial for evaluating machine learning models:
    - Training set: Used to learn model parameters
    - Test set: Used to evaluate final performance (never seen during training)
    
    This prevents overfitting and gives realistic performance estimates.
    
    Args:
        features (numpy.ndarray): Input features, shape (num_samples, num_features)
        labels (numpy.ndarray): Target labels, shape (num_samples,)
        test_ratio (float, optional): Fraction of data for testing. Default 0.3 (30%).
        random_seed (int, optional): Seed for reproducible splits. Default 42.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            X_train (numpy.ndarray): Training features
            X_test (numpy.ndarray): Test features  
            y_train (numpy.ndarray): Training labels
            y_test (numpy.ndarray): Test labels
    
    Note:
        We use stratified splitting to ensure both classes are represented
        proportionally in both training and test sets.
    """
    np.random.seed(random_seed)
    
    num_samples = len(features)
    num_test_samples = int(num_samples * test_ratio)
    
    # Create random permutation of indices
    shuffled_indices = np.random.permutation(num_samples)
    
    # Split indices into test and train
    test_indices = shuffled_indices[:num_test_samples]
    train_indices = shuffled_indices[num_test_samples:]
    
    # Extract corresponding data
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """
    Normalize features using training set statistics
    
    Feature normalization is essential for neural networks because:
    1. Different features may have vastly different scales
    2. Large-scale features can dominate the learning process
    3. Normalization accelerates convergence
    4. Improves numerical stability
    
    We use Z-score normalization: (x - mean) / std
    This transforms features to have mean=0 and std=1
    
    Args:
        X_train (numpy.ndarray): Training features, shape (num_train, num_features)
        X_test (numpy.ndarray): Test features, shape (num_test, num_features)
    
    Returns:
        tuple: (X_train_normalized, X_test_normalized, normalization_params)
            X_train_normalized: Normalized training features
            X_test_normalized: Normalized test features (using training stats)
            normalization_params: Dict with 'mean' and 'std' for denormalization
    
    Important:
        We compute normalization parameters ONLY from training data and apply
        them to both training and test sets. This prevents data leakage.
    """
    # Compute normalization parameters from training data only
    # This prevents information leakage from test set
    feature_means = np.mean(X_train, axis=0)  # Mean for each feature
    feature_stds = np.std(X_train, axis=0)    # Standard deviation for each feature
    
    # Avoid division by zero for constant features
    # Add small epsilon to standard deviation
    epsilon = 1e-8
    feature_stds = np.maximum(feature_stds, epsilon)
    
    # Apply normalization to both training and test sets
    X_train_normalized = (X_train - feature_means) / feature_stds
    X_test_normalized = (X_test - feature_means) / feature_stds
    
    # Store parameters for potential denormalization
    normalization_params = {
        'mean': feature_means,
        'std': feature_stds
    }
    
    return X_train_normalized, X_test_normalized, normalization_params

def train_mlp_classifier(X_train, y_train, hidden_layer_size=8, learning_rate=1.0, 
                        max_epochs=5000, convergence_tolerance=1e-4, verbose=True):
    """
    Train MLP for binary classification with comprehensive monitoring
    
    This function implements the complete training pipeline for binary classification:
    1. Initialize network with appropriate architecture
    2. Train using backpropagation with error and accuracy monitoring
    3. Track training progress for analysis and debugging
    4. Implement early stopping based on convergence criteria
    
    Args:
        X_train (numpy.ndarray): Training features, shape (num_samples, num_features)
        y_train (numpy.ndarray): Training labels, shape (num_samples,)
        hidden_layer_size (int, optional): Number of hidden neurons. Default 8.
        learning_rate (float, optional): Learning rate for weight updates. Default 1.0.
        max_epochs (int, optional): Maximum training iterations. Default 5000.
        convergence_tolerance (float, optional): Stop when error below this. Default 1e-4.
        verbose (bool, optional): Whether to print training progress. Default True.
    
    Returns:
        tuple: (weights_ih, bias_h, weights_ho, bias_o, training_history)
            weights_ih: Trained input-to-hidden weights
            bias_h: Trained hidden layer biases
            weights_ho: Trained hidden-to-output weights
            bias_o: Trained output layer biases
            training_history: Dict with 'errors' and 'accuracies' lists
    
    Note:
        We track both error (for convergence) and accuracy (for interpretability).
        Error decreases smoothly, while accuracy can plateau at discrete levels.
    """
    num_samples, num_features = X_train.shape
    
    # Initialize network architecture
    input_size = num_features
    output_size = 1  # Binary classification
    
    # Initialize weights and biases
    weights_ih, bias_h, weights_ho, bias_o = initialize_mlp_weights(
        input_size, hidden_layer_size, output_size)
    
    # Initialize training history tracking
    training_history = {
        'errors': [],      # Mean squared error per epoch
        'accuracies': []   # Classification accuracy per epoch
    }
    
    if verbose:
        print(f"Training MLP for Binary Classification:")
        print(f"  Training samples: {num_samples}")
        print(f"  Input features: {num_features}")
        print(f"  Hidden neurons: {hidden_layer_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max epochs: {max_epochs}")
        print("-" * 50)
    
    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    
    for epoch in range(max_epochs):
        epoch_total_error = 0
        epoch_correct_predictions = 0
        
        # Process each training sample
        for sample_idx in range(num_samples):
            current_input = X_train[sample_idx]
            current_target = y_train[sample_idx]
            
            # ================================================================
            # FORWARD PASS
            # ================================================================
            
            # Compute network output
            network_output, hidden_activations, hidden_net_input, output_net_input = \
                mlp_forward_pass(current_input, weights_ih, bias_h, weights_ho, bias_o)
            
            # Compute error for this sample
            sample_error = 0.5 * (current_target - network_output[0]) ** 2
            epoch_total_error += sample_error
            
            # Compute accuracy for this sample
            predicted_class = 1 if network_output[0] >= 0.5 else 0
            if predicted_class == current_target:
                epoch_correct_predictions += 1
            
            # ================================================================
            # BACKWARD PASS AND WEIGHT UPDATE
            # ================================================================
            
            # Compute gradients
            grad_w_ih, grad_b_h, grad_w_ho, grad_b_o = mlp_backward_pass(
                current_input, current_target, network_output, hidden_activations,
                hidden_net_input, output_net_input, weights_ih, weights_ho)
            
            # Update weights and biases
            weights_ih += learning_rate * grad_w_ih
            bias_h += learning_rate * grad_b_h
            weights_ho += learning_rate * grad_w_ho
            bias_o += learning_rate * grad_b_o
        
        # ====================================================================
        # EPOCH STATISTICS
        # ====================================================================
        
        # Compute epoch-level metrics
        average_error = epoch_total_error / num_samples
        accuracy = epoch_correct_predictions / num_samples
        
        # Store in training history
        training_history['errors'].append(average_error)
        training_history['accuracies'].append(accuracy)
        
        # Print progress periodically
        if verbose and (epoch % 500 == 0 or average_error < convergence_tolerance):
            print(f"Epoch {epoch+1}: Error = {average_error:.6f}, "
                  f"Accuracy = {accuracy:.3f}")
        
        # Check for convergence
        if average_error < convergence_tolerance:
            if verbose:
                print(f"Convergence achieved in {epoch+1} epochs!")
            break
    
    return weights_ih, bias_h, weights_ho, bias_o, training_history

def evaluate_mlp_classifier(X_test, y_test, weights_ih, bias_h, weights_ho, bias_o):
    """
    Comprehensive evaluation of trained MLP classifier
    
    This function computes detailed performance metrics including:
    1. Basic accuracy (fraction of correct predictions)
    2. Precision, recall, and F1-score
    3. Confusion matrix analysis
    4. Individual prediction details
    
    Args:
        X_test (numpy.ndarray): Test features, shape (num_samples, num_features)
        y_test (numpy.ndarray): True test labels, shape (num_samples,)
        weights_ih, bias_h, weights_ho, bias_o: Trained network parameters
    
    Returns:
        tuple: (accuracy, predictions, probabilities)
            accuracy (float): Overall classification accuracy
            predictions (list): Binary predictions for each test sample
            probabilities (list): Continuous output probabilities
    
    Note:
        We return both binary predictions and continuous probabilities
        because probabilities provide confidence information.
    """
    num_test_samples = len(X_test)
    predictions = []
    probabilities = []
    correct_count = 0
    
    # Evaluate each test sample
    for sample_idx in range(num_test_samples):
        current_input = X_test[sample_idx]
        true_label = y_test[sample_idx]
        
        # Get network output
        network_output, _, _, _ = mlp_forward_pass(
            current_input, weights_ih, bias_h, weights_ho, bias_o)
        
        # Extract probability and make binary prediction
        probability = network_output[0]
        binary_prediction = 1 if probability >= 0.5 else 0
        
        # Store results
        probabilities.append(probability)
        predictions.append(binary_prediction)
        
        # Count correct predictions
        if binary_prediction == true_label:
            correct_count += 1
    
    # Calculate overall accuracy
    accuracy = correct_count / num_test_samples
    
    return accuracy, predictions, probabilities

def compute_detailed_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive classification metrics
    
    This function calculates standard machine learning evaluation metrics:
    
    1. Confusion Matrix: Shows true vs predicted classifications
       - True Positives (TP): Correctly predicted positive class
       - True Negatives (TN): Correctly predicted negative class  
       - False Positives (FP): Incorrectly predicted positive class
       - False Negatives (FN): Incorrectly predicted negative class
    
    2. Derived Metrics:
       - Accuracy: (TP + TN) / (TP + TN + FP + FN)
       - Precision: TP / (TP + FP) - fraction of positive predictions that are correct
       - Recall: TP / (TP + FN) - fraction of actual positives that are detected
       - F1-Score: Harmonic mean of precision and recall
    
    Args:
        y_true (numpy.ndarray): True binary labels
        y_pred (numpy.ndarray): Predicted binary labels  
        y_prob (numpy.ndarray): Prediction probabilities (not used here but available)
    
    Returns:
        dict: Dictionary containing all computed metrics
            - 'accuracy': Overall accuracy
            - 'precision': Precision for positive class
            - 'recall': Recall for positive class
            - 'f1_score': F1-score
            - 'confusion_matrix': 2x2 numpy array
    """
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix components
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    # Compute derived metrics with safe division
    total_samples = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create confusion matrix
    confusion_matrix = np.array([[true_negatives, false_positives],
                                [false_negatives, true_positives]])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': confusion_matrix
    }

def visualize_dataset(features, labels, title="Dataset Visualization"):
    """
    Create scatter plot visualization of the dataset
    
    This visualization helps understand:
    1. The spatial distribution of classes
    2. Whether classes are linearly separable
    3. The complexity of the classification problem
    4. Potential outliers or mislabeled points
    
    Args:
        features (numpy.ndarray): 2D feature data, shape (num_samples, 2)
        labels (numpy.ndarray): Binary labels, shape (num_samples,)
        title (str, optional): Plot title. Default "Dataset Visualization".
    
    Returns:
        None: Creates and displays plot
    
    Note:
        This function assumes 2D features for visualization.
        For higher dimensions, dimensionality reduction would be needed.
    """
    plt.figure(figsize=(8, 6))
    
    # Define colors and labels for classes
    colors = ['red', 'blue']
    class_names = ['Class 0', 'Class 1']
    
    # Plot each class separately for proper legend
    for class_label in [0, 1]:
        # Create mask for current class
        class_mask = labels == class_label
        
        # Plot points for this class
        plt.scatter(features[class_mask, 0], features[class_mask, 1], 
                   c=colors[class_label], alpha=0.6, s=50, 
                   label=class_names[class_label])
    
    # Set plot properties
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio for circular patterns

def visualize_decision_surface(features, labels, weights_ih, bias_h, weights_ho, bias_o, 
                              normalization_params, title="MLP Decision Surface"):
    """
    Visualize the decision surface learned by the MLP
    
    This creates a comprehensive visualization showing:
    1. The continuous output surface as a color map
    2. The decision boundary (where output = 0.5)
    3. The actual data points colored by their true class
    4. How the MLP separates the classes
    
    Args:
        features (numpy.ndarray): Original feature data for plotting points
        labels (numpy.ndarray): True labels for coloring points
        weights_ih, bias_h, weights_ho, bias_o: Trained network parameters
        normalization_params (dict): Mean and std for feature normalization
        title (str, optional): Plot title
    
    Returns:
        None: Creates and displays plot, saves to file
    """
    plt.figure(figsize=(10, 8))
    
    # Create evaluation grid
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    grid_resolution = 100
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    
    # Evaluate MLP on grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Normalize grid points using training statistics
    grid_points_normalized = ((grid_points - normalization_params['mean']) / 
                             normalization_params['std'])
    
    # Compute MLP outputs for grid
    grid_outputs = []
    for point in grid_points_normalized:
        output, _, _, _ = mlp_forward_pass(point, weights_ih, bias_h, weights_ho, bias_o)
        grid_outputs.append(output[0])
    
    # Reshape for plotting
    output_surface = np.array(grid_outputs).reshape(xx.shape)
    
    # Create decision surface plot
    contour_plot = plt.contourf(xx, yy, output_surface, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(contour_plot, label='MLP Output Probability')
    
    # Draw decision boundary
    plt.contour(xx, yy, output_surface, levels=[0.5], colors='black', 
               linewidths=2, linestyles='--', label='Decision Boundary')
    
    # Plot data points
    colors = ['red', 'blue']
    class_names = ['Class 0', 'Class 1']
    
    for class_label in [0, 1]:
        class_mask = labels == class_label
        plt.scatter(features[class_mask, 0], features[class_mask, 1], 
                   c=colors[class_label], alpha=0.8, s=50, 
                   edgecolors='black', linewidth=0.5,
                   label=class_names[class_label])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def visualize_training_progress(training_history):
    """
    Plot training error and accuracy curves
    
    This visualization shows:
    1. How training error decreases over time
    2. How accuracy improves during training
    3. Whether the model has converged
    4. Potential overfitting or underfitting issues
    
    Args:
        training_history (dict): Dictionary with 'errors' and 'accuracies' lists
    
    Returns:
        None: Creates and displays plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(training_history['errors']) + 1)
    
    # Plot training error
    ax1.plot(epochs, training_history['errors'], 'b-', linewidth=2)
    ax1.set_title('Training Error Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_yscale('log')  # Log scale for better visualization
    ax1.grid(True, alpha=0.3)
    
    # Plot training accuracy
    ax2.plot(epochs, training_history['accuracies'], 'g-', linewidth=2)
    ax2.set_title('Training Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

def main():
    """
    Main function demonstrating binary classification with MLP
    
    This comprehensive example shows:
    1. Generating a realistic non-linearly separable dataset
    2. Proper data preprocessing (train/test split, normalization)
    3. Training an MLP with monitoring
    4. Comprehensive evaluation with multiple metrics
    5. Visualization of results and decision boundaries
    6. Analysis of training progress
    
    The goal is to demonstrate practical application of MLPs to
    real-world-like classification problems.
    """
    print("=" * 60)
    print("BINARY CLASSIFICATION WITH MLP - PRACTICAL EXAMPLE")
    print("=" * 60)
    
    # ========================================================================
    # DATASET GENERATION AND PREPROCESSING
    # ========================================================================
    
    print("1. Generating synthetic dataset...")
    features, labels = generate_classification_dataset(num_samples=300, noise_level=0.15)
    
    print("2. Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(features, labels, test_ratio=0.3)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    print("3. Normalizing features...")
    X_train_norm, X_test_norm, norm_params = normalize_features(X_train, X_test)
    
    # ========================================================================
    # DATASET VISUALIZATION
    # ========================================================================
    
    print("4. Visualizing training dataset...")
    visualize_dataset(X_train, y_train, "Training Dataset")
    plt.savefig('/home/ubuntu/training_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    
    print("5. Training MLP classifier...")
    weights_ih, bias_h, weights_ho, bias_o, training_history = train_mlp_classifier(
        X_train_norm, y_train, 
        hidden_layer_size=10,    # Sufficient capacity for this problem
        learning_rate=2.0,       # Aggressive learning rate for faster convergence
        max_epochs=3000,         # Enough epochs for convergence
        convergence_tolerance=1e-4)
    
    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================
    
    print("6. Evaluating on test set...")
    test_accuracy, test_predictions, test_probabilities = evaluate_mlp_classifier(
        X_test_norm, y_test, weights_ih, bias_h, weights_ho, bias_o)
    
    print("7. Computing detailed metrics...")
    detailed_metrics = compute_detailed_metrics(y_test, test_predictions, test_probabilities)
    
    # ========================================================================
    # RESULTS REPORTING
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print(f"Test Accuracy: {detailed_metrics['accuracy']:.3f}")
    print(f"Precision: {detailed_metrics['precision']:.3f}")
    print(f"Recall: {detailed_metrics['recall']:.3f}")
    print(f"F1-Score: {detailed_metrics['f1_score']:.3f}")
    
    print("\nConfusion Matrix:")
    print("           Predicted")
    print("         Class 0  Class 1")
    print(f"Actual 0    {detailed_metrics['confusion_matrix'][0,0]:3d}      {detailed_metrics['confusion_matrix'][0,1]:3d}")
    print(f"Actual 1    {detailed_metrics['confusion_matrix'][1,0]:3d}      {detailed_metrics['confusion_matrix'][1,1]:3d}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print("\n8. Generating visualizations...")
    
    # Decision surface visualization
    visualize_decision_surface(X_test, y_test, weights_ih, bias_h, weights_ho, bias_o, 
                              norm_params, "MLP Decision Surface (Test Set)")
    plt.savefig('/home/ubuntu/decision_surface_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training progress visualization
    visualize_training_progress(training_history)
    plt.savefig('/home/ubuntu/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # ANALYSIS AND CONCLUSIONS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("The MLP successfully learned to classify the non-linearly separable")
    print("concentric pattern, demonstrating the power of multi-layer architectures")
    print("for complex classification problems.")
    
    if detailed_metrics['accuracy'] > 0.9:
        print("\n✓ Excellent performance achieved!")
        print("  The model successfully learned the complex circular boundary.")
    elif detailed_metrics['accuracy'] > 0.8:
        print("\n✓ Good performance achieved!")
        print("  The model learned most of the pattern with some errors.")
    else:
        print("\n? Performance could be improved.")
        print("  Consider adjusting hyperparameters or network architecture.")
    
    print("=" * 60)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute the binary classification demonstration
    
    This script can be run independently to see the full demonstration
    of MLP binary classification, or imported as a module for other uses.
    """
    main()


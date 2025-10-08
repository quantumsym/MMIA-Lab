#!/usr/bin/env python

# ==========================================
# KERAS XOR EXAMPLE
# Demonstrates modern deep learning framework usage
# ==========================================

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not found. Please install with: pip install tensorflow")

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# XOR PROBLEM WITH KERAS
# ==========================================

def create_xor_dataset():
    """
    Create the XOR dataset
    Returns the classic XOR truth table
    """
    # XOR truth table
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    return X, y

def display_xor_problem():
    """
    Display the XOR problem clearly
    """
    X, y = create_xor_dataset()
    
    print("XOR PROBLEM - TRUTH TABLE")
    print("=" * 30)
    print("Input A | Input B | XOR Output")
    print("--------|---------|------------")
    
    for i in range(len(X)):
        a, b = int(X[i][0]), int(X[i][1])
        output = int(y[i][0])
        print(f"   {a}    |    {b}    |     {output}")
    
    print("\nWhy XOR is challenging:")
    print("• XOR is not linearly separable")
    print("• Single perceptron cannot solve it")
    print("• Requires non-linear decision boundary")
    print("• Perfect test case for multi-layer networks")
    
    return X, y

def create_keras_model_simple():
    """
    Create a simple Keras model for XOR
    Demonstrates the minimal code needed
    """
    if not KERAS_AVAILABLE:
        print("Keras not available - cannot create model")
        return None
    
    print("\n" + "="*40)
    print("CREATING SIMPLE KERAS MODEL")
    print("="*40)
    
    # Create sequential model (layers stacked one after another)
    model = keras.Sequential([
        # Hidden layer: 4 neurons with sigmoid activation
        layers.Dense(4, activation='sigmoid', input_shape=(2,), name='hidden_layer'),
        
        # Output layer: 1 neuron with sigmoid for binary classification
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # Compile the model (specify optimizer, loss function, metrics)
    model.compile(
        optimizer='adam',           # Adaptive learning rate optimizer
        loss='binary_crossentropy', # Perfect for binary classification
        metrics=['accuracy']        # Track accuracy during training
    )
    
    print("Model created successfully!")
    print("\nModel Architecture:")
    model.summary()
    
    return model

def create_keras_model_advanced():
    """
    Create a more advanced Keras model with additional features
    Shows more options and best practices
    """
    if not KERAS_AVAILABLE:
        print("Keras not available - cannot create model")
        return None
    
    print("\n" + "="*40)
    print("CREATING ADVANCED KERAS MODEL")
    print("="*40)
    
    # Alternative way: Functional API (more flexible)
    inputs = keras.Input(shape=(2,), name='xor_inputs')
    
    # First hidden layer with more neurons and ReLU
    x = layers.Dense(8, activation='relu', name='hidden_1')(inputs)
    
    # Second hidden layer
    x = layers.Dense(4, activation='relu', name='hidden_2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='XOR_Solver')
    
    # Compile with custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Advanced model created!")
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_keras_model(model, X, y, epochs=500, verbose=1):
    """
    Train the Keras model on XOR data
    
    Parameters:
    - model: Keras model to train
    - X, y: training data
    - epochs: number of training epochs
    - verbose: verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
    - history: training history object
    """
    if model is None:
        print("No model provided for training")
        return None
    
    print(f"\nTraining model for {epochs} epochs...")
    print("Watch the loss decrease and accuracy increase!")
    
    # Train the model
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=4,      # Use all samples in each batch (small dataset)
        verbose=verbose,   # Show training progress
        shuffle=True       # Shuffle data between epochs
    )
    
    print("Training completed!")
    
    return history

def evaluate_keras_model(model, X, y):
    """
    Evaluate the trained Keras model
    """
    if model is None:
        print("No model provided for evaluation")
        return
    
    print("\n" + "="*40)
    print("EVALUATING TRAINED MODEL")
    print("="*40)
    
    # Make predictions
    predictions = model.predict(X, verbose=0)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(binary_predictions.flatten() == y.flatten()) * 100
    
    # Display results
    print("Prediction Results:")
    print("Input -> Expected -> Predicted -> Probability -> Correct?")
    print("-" * 60)
    
    for i in range(len(X)):
        input_vals = X[i]
        expected = int(y[i][0])
        predicted = binary_predictions[i][0]
        probability = predictions[i][0]
        is_correct = predicted == expected
        
        print(f"[{int(input_vals[0])}, {int(input_vals[1])}] ->    {expected}     ->     {predicted}      ->   {probability:.4f}    -> {'✓' if is_correct else '✗'}")
    
    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("🎉 PERFECT! The model learned XOR completely!")
    elif accuracy >= 75:
        print("👍 Good! The model learned XOR well.")
    else:
        print("🤔 The model needs more training or different architecture.")
    
    return accuracy

def plot_training_history(history):
    """
    Plot training history (loss and accuracy over epochs)
    """
    if history is None:
        print("No training history to plot")
        return
    
    # Extract training history
    epochs = range(1, len(history.history['loss']) + 1)
    loss_values = history.history['loss']
    accuracy_values = history.history['accuracy']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, loss_values, 'b-o', markersize=3, linewidth=2, label='Training Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, accuracy_values, 'g-o', markersize=3, linewidth=2, label='Training Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_decision_boundary_keras(model, X, y):
    """
    Visualize the decision boundary learned by the Keras model
    """
    if model is None:
        print("No model provided for visualization")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_predictions = model.predict(mesh_points, verbose=0)
    mesh_predictions = mesh_predictions.reshape(xx.shape)
    
    # Plot decision boundary
    contour = plt.contourf(xx, yy, mesh_predictions, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(contour, label='Probability of XOR=1')
    
    # Plot the actual XOR points
    colors = ['red', 'blue']
    labels = ['XOR = 0', 'XOR = 1']
    
    for i in range(2):
        mask = (y.flatten() == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=200, 
                   label=labels[i], edgecolors='black', linewidth=2, alpha=0.9)
    
    # Add annotations for each point
    annotations = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    for i, (x_coord, y_coord) in enumerate(X):
        plt.annotate(annotations[i], (x_coord, y_coord), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input A', fontsize=14)
    plt.ylabel('Input B', fontsize=14)
    plt.title('Keras Model Decision Boundary for XOR Problem\n(Non-linear boundary learned automatically)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_with_numpy_implementation():
    """
    Show the difference in code complexity between Keras and NumPy
    """
    print("\n" + "="*60)
    print("KERAS vs NUMPY IMPLEMENTATION COMPARISON")
    print("="*60)
    
    print("\n1. KERAS APPROACH (High-level, simple):")
    print("-" * 40)
    keras_code = '''
    # Create model
    model = keras.Sequential([
        layers.Dense(4, activation='sigmoid', input_shape=(2,)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train
    model.fit(X, y, epochs=500)
    
    # Predict
    predictions = model.predict(X)
    '''
    print(keras_code)
    
    print("\n2. NUMPY APPROACH (Low-level, more code):")
    print("-" * 40)
    numpy_code = '''
    # Initialize weights manually
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))
    
    # Training loop
    for epoch in range(500):
        # Forward pass
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        
        # Backward pass (calculate gradients)
        dz2 = a2 - y
        dW2 = a1.T.dot(dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        da1 = dz2.dot(W2.T)
        dz1 = da1 * sigmoid_derivative(z1)
        dW1 = X.T.dot(dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    '''
    print(numpy_code)
    
    print("\n3. KEY DIFFERENCES:")
    print("-" * 40)
    print("✅ KERAS ADVANTAGES:")
    print("   • Much less code (90% reduction)")
    print("   • No manual gradient calculation")
    print("   • Built-in optimizers (Adam, RMSprop, etc.)")
    print("   • Automatic differentiation")
    print("   • GPU acceleration ready")
    print("   • Built-in regularization, dropout, etc.")
    
    print("\n📚 NUMPY ADVANTAGES:")
    print("   • Complete understanding of internals")
    print("   • Full control over every operation")
    print("   • Educational value")
    print("   • No dependencies (except NumPy)")
    print("   • Customizable algorithms")
    
    print("\n🎯 RECOMMENDATION FOR LEARNING:")
    print("   1. Start with NumPy to understand fundamentals")
    print("   2. Move to Keras for practical applications")
    print("   3. Use both: NumPy for learning, Keras for production")

def demonstrate_keras_features():
    """
    Demonstrate additional Keras features useful for beginners
    """
    if not KERAS_AVAILABLE:
        print("Keras not available - skipping feature demonstration")
        return
    
    print("\n" + "="*50)
    print("ADDITIONAL KERAS FEATURES FOR BEGINNERS")
    print("="*50)
    
    X, y = create_xor_dataset()
    
    # 1. Model with different optimizers
    print("\n1. TRYING DIFFERENT OPTIMIZERS:")
    print("-" * 35)
    
    optimizers = ['sgd', 'adam', 'rmsprop']
    results = {}
    
    for opt_name in optimizers:
        print(f"\nTesting {opt_name.upper()} optimizer...")
        
        # Create model
        model = keras.Sequential([
            layers.Dense(4, activation='sigmoid', input_shape=(2,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=opt_name, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train quietly
        history = model.fit(X, y, epochs=200, verbose=0)
        
        # Get final accuracy
        final_accuracy = history.history['accuracy'][-1]
        results[opt_name] = final_accuracy
        
        print(f"   Final accuracy: {final_accuracy:.3f}")
    
    best_optimizer = max(results, key=results.get)
    print(f"\n🏆 Best optimizer for this problem: {best_optimizer.upper()}")
    
    # 2. Model with callbacks
    print("\n2. USING CALLBACKS (Advanced features):")
    print("-" * 40)
    
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(2,)),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=50, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=25,
        verbose=1
    )
    
    print("Training with Early Stopping and Learning Rate Reduction...")
    history = model.fit(
        X, y, 
        epochs=1000, 
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    print(f"Training stopped early at epoch {len(history.history['loss'])}")
    
    # 3. Model saving and loading
    print("\n3. SAVING AND LOADING MODELS:")
    print("-" * 35)
    
    try:
        # Save the model
        model.save('xor_model.h5')
        print("✅ Model saved as 'xor_model.h5'")
        
        # Load the model
        loaded_model = keras.models.load_model('xor_model.h5')
        print("✅ Model loaded successfully")
        
        # Test loaded model
        test_predictions = loaded_model.predict(X, verbose=0)
        print("✅ Loaded model makes predictions correctly")
        
        # Clean up
        import os
        if os.path.exists('xor_model.h5'):
            os.remove('xor_model.h5')
            print("✅ Temporary file cleaned up")
            
    except Exception as e:
        print(f"⚠️  Model saving/loading failed: {e}")

# ==========================================
# MAIN DEMONSTRATION
# ==========================================

def main():
    """
    Complete demonstration of Keras for XOR problem
    """
    print("KERAS XOR SOLVER - MODERN DEEP LEARNING APPROACH")
    print("=" * 60)
    
    if not KERAS_AVAILABLE:
        print("\n❌ TensorFlow/Keras is not installed!")
        print("Please install it with: pip install tensorflow")
        print("This example requires Keras to demonstrate modern deep learning.")
        return
    
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ Keras version: {keras.__version__}")
    
    # 1. Show the problem
    print("\n" + "="*40)
    print("STEP 1: UNDERSTANDING THE PROBLEM")
    print("="*40)
    X, y = display_xor_problem()
    
    # 2. Create simple model
    print("\n" + "="*40)
    print("STEP 2: CREATE SIMPLE KERAS MODEL")
    print("="*40)
    model_simple = create_keras_model_simple()
    
    # 3. Train simple model
    print("\n" + "="*40)
    print("STEP 3: TRAIN THE MODEL")
    print("="*40)
    history_simple = train_keras_model(model_simple, X, y, epochs=500, verbose=2)
    
    # 4. Evaluate model
    print("\n" + "="*40)
    print("STEP 4: EVALUATE RESULTS")
    print("="*40)
    accuracy_simple = evaluate_keras_model(model_simple, X, y)
    
    # 5. Advanced model
    print("\n" + "="*40)
    print("STEP 5: ADVANCED MODEL COMPARISON")
    print("="*40)
    model_advanced = create_keras_model_advanced()
    history_advanced = train_keras_model(model_advanced, X, y, epochs=300, verbose=0)
    accuracy_advanced = evaluate_keras_model(model_advanced, X, y)
    
    # 6. Visualizations
    try:
        print("\n" + "="*40)
        print("STEP 6: VISUALIZATIONS")
        print("="*40)
        
        print("Plotting training history...")
        plot_training_history(history_simple)
        
        print("Plotting decision boundaries...")
        visualize_decision_boundary_keras(model_simple, X, y)
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Install matplotlib for visualizations: pip install matplotlib")
    
    # 7. Code comparison
    compare_with_numpy_implementation()
    
    # 8. Additional features
    demonstrate_keras_features()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✅ Simple Model Accuracy: {accuracy_simple:.1f}%")
    print(f"✅ Advanced Model Accuracy: {accuracy_advanced:.1f}%")
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("1. Keras makes neural networks incredibly simple")
    print("2. Just a few lines of code solve complex problems")
    print("3. Built-in optimizers and functions handle the math")
    print("4. Automatic differentiation eliminates manual gradients")
    print("5. Perfect for beginners and production use")
    
    print("\n🚀 NEXT STEPS:")
    print("- Try larger datasets (MNIST, CIFAR-10)")
    print("- Experiment with different architectures")
    print("- Learn about convolutional neural networks")
    print("- Explore transfer learning")
    print("- Build real-world applications")

if __name__ == "__main__":
    main()

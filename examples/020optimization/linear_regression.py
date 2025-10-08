#!/usr/bin/env python
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load values from a CSV file.
# The CSV file ('data.csv') must have two columns named 'x' and 'y'.
filename = 'data.csv'  # <--- Replace with your file path

# Read the data using pandas
data = pd.read_csv(filename)
X = data['x'].to_numpy()  # Feature values (input)
Y = data['y'].to_numpy()  # Target values (output)

# Initialize parameters for the regression line: y = m*x + b
m = 0.0  # Initial slope
b = 0.0  # Initial intercept

learning_rate = 0.0001   # Step size for parameter updates 
num_iterations = 10000   # Number of iterations

#---------------------------------------------------------
    
def linear_regression(X,Y,m,b):

    N = len(X)               # Number of data points
    
    for i in range(num_iterations):
        # Calculate current predictions
        y_pred = m * X + b
    
        # Compute gradients w.r.t. each parameter
        dm = (-2 / N) * np.sum(X * (Y - y_pred))  # Derivative w.r.t. slope
        db = (-2 / N) * np.sum(Y - y_pred)        # Derivative w.r.t. intercept
    
        # Take a gradient descent step
        m = m - learning_rate * dm
        b = b - learning_rate * db
    
    return m,b
    
#-----------------------------------------------------------
    
def plot_simulation(X,Y,m,b,y_pred_final): 
    # --- Plot results ---
    plt.figure(figsize=(8, 5))
    
    # Scatter plot of all input data points
    plt.scatter(X, Y, color='blue', label='Original data')
    
    # Draw regression line (learned by gradient descent)
    plt.plot(X, y_pred_final, color='red', label='Fitted line (gradient descent)')
    
    # Add chart details
    plt.xlabel('X (input)')
    plt.ylabel('Y (target)')
    plt.title('Linear Regression with Gradient Descent')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
    
    return 

#------------------------------------------------------



m,b = linear_regression(X,Y,m,b)

# Final predictions after training
y_pred_final = m * X + b

plot_simulation(X,Y,m,b,y_pred_final) 

# Print final results
print(f"Final slope (m): {m:.4f}")
print(f"Final intercept (b): {b:.4f}")
print(f"Final equation: y = {m:.4f}x + {b:.4f}")


#!/usr/bin/python
#


# Gradient Descent with JAX and Automatic Differentiation
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable double precision for better accuracy
jax.config.update("jax_enable_x64", True)

def gradient_descent_jax(f, x0, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """
    Gradient Descent using JAX's automatic differentiation.
    Args:
        f: Objective function (accepts a vector and returns a scalar)
        x0: Initial guess (array-like)
        learning_rate: Step size for updates
        max_iter: Maximum number of iterations
        tolerance: Stop criterion on parameter update
    Returns:
        x: Final parameters found
        path: Sequence of points visited during optimization
        costs: Sequence of cost function values
    """
    grad_f = jax.grad(f)       # Automatic gradient computation with JAX
    x = jnp.array(x0, dtype=jnp.float64)
    path = [np.array(x)]       # Store path for later visualization (as NumPy arrays)
    costs = [float(f(x))]

    for i in range(max_iter):
        grad = grad_f(x)                  # Compute gradient
        x_new = x - learning_rate * grad  # Gradient step
        # Check convergence: if step size is small enough, stop
        if jnp.linalg.norm(x_new - x) < tolerance:
            print(f"Convergence reached after {i+1} iterations")
            break
        x = x_new
        path.append(np.array(x))          # Convert to NumPy for plotting
        costs.append(float(f(x)))
    return x, np.array(path), np.array(costs)

def rosenbrock_jax(x):
    """
    The Rosenbrock (banana) function.
    Global minimum at (1, 1) with function value 0.
    Accepts a NumPy or JAX array of shape (2,).
    """
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2



def plot_rosenbrock_3d(path):
    """
    Plots the Rosenbrock function as a 3D surface and overlays the optimization path.
    Args:
        path: Numpy array of shape (steps, 2) with optimization trajectory
    """
    # Build mesh grid for the surface
    x_vals = np.linspace(-2, 2, 200)
    y_vals = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2


#    # Surface plot with function values
#    ax1 = fig.add_subplot(121, projection='3d')
#    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
#    ax1.set_title('Rosenbrock Function Surface')
#    ax1.set_xlabel('X')
#    ax1.set_ylabel('Y')
#    ax1.set_zlabel('f(X,Y)')
#    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # Plot optimization path on top of the surface
    fig = plt.figure(figsize=(15, 6))
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.65, linewidth=0, antialiased=True)

    # Extract path points for plotting
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_z = 100 * (path_y - path_x**2)**2 + (1 - path_x)**2

    # Plot the path as red dots connected by lines
    ax2.plot(path_x, path_y, path_z, 'r.-', markersize=6, linewidth=2, label='Gradient Descent Path')

    # Mark the start and end points
    ax2.plot([path_x[0]], [path_y[0]], [path_z[0]], 'go', label='Start', markersize=8)
    ax2.plot([path_x[-1]], [path_y[-1]], [path_z[-1]], 'b*', label='End', markersize=10)

    # Add true minimum
    ax2.plot([1], [1], [0], 'k*', label='True Minimum', markersize=12)
    ax2.set_title('Optimization Path on Rosenbrock Surface')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X,Y)')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    return

###----------------------------------------------

if __name__ == "__main__":

    
    # Run gradient descent optimization
    x_start = np.array([-1.2, 1.0])  # Use NumPy for plotting convenience
    x_min, path, costs = gradient_descent_jax( rosenbrock_jax, x_start, learning_rate=0.001, max_iter=3000)
    
    # Print results
    print(f"Minimum point found: {x_min}")
    print(f"Minimum value: {rosenbrock_jax(x_min):.8f}")
    
    # Call the visualization function
    plot_rosenbrock_3d(path)
    
    

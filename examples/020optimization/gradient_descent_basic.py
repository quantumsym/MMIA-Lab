#!/usr/bin/env python
#

#!/usr/bin/env python
"""
Basic Gradient Descent Implementation with NumPy
Demonstrates optimization of a simple quadratic function with 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# different starting points and learning rates
#   start_point: [5, 3],    learning_rate: 0.10  : Fast convergence
#   start_point: [8, -6],   learning_rate: 0.05  : Moderate learning rate
#   start_point: [10, 10],  learning_rate: 0.01  : Conservative learning rate
#
start_point = [9,-7]
learning_rate = 0.025

def quadratic(x):
    """
    Simple quadratic function f(x,y) = x² + y²
    This is a convex function with global minimum at (0,0)

    Args:
        x: numpy array of shape (2,) representing [x, y] coordinates
    Returns:
        float: function value at point x
    """
    return x[0]**2 + x[1]**2

def quadratic_grad(x):
    """
    Analytical gradient of the quadratic function
    ∇f(x,y) = [2x, 2y]

    Args:
        x: numpy array of shape (2,) representing [x, y] coordinates
    Returns:
        numpy array: gradient vector at point x
    """
    return np.array([2*x[0], 2*x[1]])

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """
    Basic gradient descent optimization algorithm

    Args:
        f: objective function to minimize
        grad_f: gradient function of the objective
        x0: starting point (list or array)
        learning_rate: step size for parameter updates
        max_iter: maximum number of iterations
        tolerance: convergence criterion (stop when step size < tolerance)

    Returns:
        tuple: (final_point, optimization_path, cost_history)
    """
    # Initialize starting point and convert to float array
    x = np.array(x0, dtype=float)

    # Store the optimization path for visualization
    path = [x.copy()]
    cost_history = [f(x)]  # Track function values over iterations

    print(f"Starting gradient descent from point: {x}")
    print(f"Initial function value: {f(x):.6f}")

    for i in range(max_iter):
        # Compute gradient at current point
        grad = grad_f(x)

        # Perform gradient descent step: x_new = x - α∇f(x)
        x_new = x - learning_rate * grad

        # Check convergence: if the step size is small enough, stop
        #step_size = np.linalg.norm(x_new - x)
        step_size = np.sqrt(np.mean( (x_new - x)**2))
        if step_size < tolerance:
            print(f"Convergence reached after {i+1} iterations")
            print(f"Final step size: {step_size:.2e}")
            break

        # Update current point
        x = x_new
        path.append(x.copy())
        cost_history.append(f(x))

        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: f({x[0]:.4f}, {x[1]:.4f}) = {f(x):.6f}")

    print(f"Final point: [{x[0]:.6f}, {x[1]:.6f}]")
    print(f"Final function value: {f(x):.8f}")

    return x, np.array(path), np.array(cost_history)



def plot_3d_surface(f, grad_f, x0, x_min, path, costs, learning_rate=0.01):
    """
    Plot 3D surface with gradient descent optimization path

    Args:
        f: objective function
        grad_f: gradient function
        x0: starting point
        x_min: final minimum point
        path: optimization path array
        costs: cost history array
        learning_rate: learning rate used
    """
    # Create mesh grid for function surface visualization
    x_range = np.linspace(-6, 6, 100)
    y_range = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute function values over the grid
    Z = X**2 + Y**2  # Direct computation for efficiency

    # Compute function values along the optimization path
    z_path = [f(point) for point in path]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the function surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                          linewidth=0, antialiased=True)

    # Plot optimization path as red line with markers
    ax.plot(path[:, 0], path[:, 1], z_path, 'r-', linewidth=3,
            label='Optimization Path', alpha=0.9)
    ax.scatter(path[:, 0], path[:, 1], z_path, c='red', s=30, alpha=0.8)

    # Mark start and end points
    ax.scatter([path[0, 0]], [path[0, 1]], [z_path[0]],
              c='green', s=100, marker='o', label='Start')
    ax.scatter([path[-1, 0]], [path[-1, 1]], [z_path[-1]],
              c='blue', s=150, marker='*', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.set_title('3D Surface with Gradient Descent Path\n(Press Q to close and continue)')
    ax.legend()

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Add statistics text box
    stats_text = (f'Starting point: [{x0[0]:.2f}, {x0[1]:.2f}]\n'
                 f'Final point: [{x_min[0]:.4f}, {x_min[1]:.4f}]\n'
                 f'Total iterations: {len(path)}\n'
                 f'Final value: {costs[-1]:.6f}\n'
                 f'Learning rate: {learning_rate}')

    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Set up key press event to close with 'q'
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print("3D Surface Plot - Press 'Q' to close and continue to next plot")
    plt.show()

def plot_2d_contour(f, grad_f, x0, x_min, path, costs, learning_rate=0.01):
    """
    Plot 2D contour map with gradient descent optimization path

    Args:
        f: objective function
        grad_f: gradient function
        x0: starting point
        x_min: final minimum point
        path: optimization path array
        costs: cost history array
        learning_rate: learning rate used
    """
    # Create mesh grid for contour visualization
    x_range = np.linspace(-6, 6, 100)
    y_range = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute function values over the grid
    Z = X**2 + Y**2

    # Create 2D contour plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot
    contours = ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6)
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8)

    # Plot optimization path
    ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=3,
            label='Optimization Path', alpha=0.9)
    ax.scatter(path[:, 0], path[:, 1], c='red', s=30, alpha=0.8)

    # Mark start and end points
    ax.scatter([path[0, 0]], [path[0, 1]], c='green', s=150,
              marker='o', label='Start', zorder=5, edgecolors='black')
    ax.scatter([path[-1, 0]], [path[-1, 1]], c='blue', s=200,
              marker='*', label='End', zorder=5, edgecolors='black')

    # Add step numbers along the path (every 10th step for clarity)
    step_interval = max(1, len(path) // 10)
    for i in range(0, len(path), step_interval):
        ax.annotate(f'{i}', (path[i, 0], path[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Contour Plot with Optimization Path\n(Press Q to close and continue)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Add statistics text box
    stats_text = (f'Starting point: [{x0[0]:.2f}, {x0[1]:.2f}]\n'
                 f'Final point: [{x_min[0]:.4f}, {x_min[1]:.4f}]\n'
                 f'Path length: {len(path)} steps\n'
                 f'Total distance: {np.sum([np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]):.3f}\n'
                 f'Learning rate: {learning_rate}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Set up key press event to close with 'q'
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print("2D Contour Plot - Press 'Q' to close and continue to next plot")
    plt.show()

def plot_convergence(f, grad_f, x0, x_min, path, costs, learning_rate=0.01):
    """
    Plot convergence analysis showing function value and gradient norm vs iterations

    Args:
        f: objective function
        grad_f: gradient function
        x0: starting point
        x_min: final minimum point
        path: optimization path array
        costs: cost history array
        learning_rate: learning rate used
    """
    # Create convergence analysis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    iterations = np.arange(len(costs))

    # Plot 1: Function value vs iterations (log scale)
    ax1.semilogy(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function Value (log scale)')
    ax1.set_title('Convergence Analysis: Function Value vs Iterations')
    ax1.grid(True, alpha=0.3)

    # Add convergence rate information
    if len(costs) > 10:
        # Calculate approximate convergence rate
        initial_cost = costs[0]
        final_cost = costs[-1]
        reduction_factor = final_cost / initial_cost
        ax1.axhline(y=final_cost, color='red', linestyle='--', alpha=0.5, label=f'Final value: {final_cost:.6f}')
        ax1.legend()

    # Plot 2: Gradient norm vs iterations
    gradient_norms = []
    for point in path:
        grad_norm = np.linalg.norm(grad_f(point))
        gradient_norms.append(grad_norm)

    ax2.semilogy(range(len(gradient_norms)), gradient_norms, 'g-', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm (log scale)')
    ax2.set_title('Gradient Norm vs Iterations (Convergence Indicator)')
    ax2.grid(True, alpha=0.3)

    # Add theoretical convergence line for quadratic functions
    if len(gradient_norms) > 1:
        theoretical_decay = gradient_norms[0] * ((1 - learning_rate)**np.arange(len(gradient_norms)))
        ax2.plot(range(len(theoretical_decay)), theoretical_decay, 'r--', alpha=0.5,
                label=f'Theoretical decay (rate={(1-learning_rate):.3f})')
        ax2.legend()

    # Add comprehensive statistics text box
    final_grad_norm = gradient_norms[-1] if gradient_norms else 0
    convergence_rate = -np.log(costs[-1]/costs[0]) / len(costs) if len(costs) > 1 else 0

    stats_text = (f'Optimization Summary:\n'
                 f'━━━━━━━━━━━━━━━━━━━\n'
                 f'Starting point: [{x0[0]:.2f}, {x0[1]:.2f}]\n'
                 f'Final point: [{x_min[0]:.6f}, {x_min[1]:.6f}]\n'
                 f'Initial cost: {costs[0]:.6f}\n'
                 f'Final cost: {costs[-1]:.8f}\n'
                 f'Cost reduction: {(costs[0]-costs[-1])/costs[0]*100:.2f}%\n'
                 f'Total iterations: {len(path)}\n'
                 f'Final gradient norm: {final_grad_norm:.2e}\n'
                 f'Avg. convergence rate: {convergence_rate:.4f}\n'
                 f'Learning rate: {learning_rate}')

    fig.text(0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='bottom')

    plt.suptitle('Convergence Analysis\n(Press Q to close)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for statistics

    # Set up key press event to close with 'q'
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print("Convergence Analysis Plot - Press 'Q' to close and finish")
    plt.show()

def visualize_gradient_descent(f, grad_f, x0, x_min, path, costs, learning_rate=0.01):
    """
    Interactive visualization of gradient descent with three separate plots
    Navigate through plots by pressing 'Q' to continue to the next one

    Args:
        f: objective function
        grad_f: gradient function
        x0: starting point
        x_min: final minimum point
        path: optimization path array
        costs: cost history array
        learning_rate: learning rate used
    """
    print("\n" + "="*60)
    print("INTERACTIVE GRADIENT DESCENT VISUALIZATION")
    print("="*60)
    print("Three plots will be shown sequentially:")
    print("1. 3D Surface with optimization path")
    print("2. 2D Contour map with optimization path")
    print("3. Convergence analysis")
    print("\nPress 'Q' in each plot window to continue to the next plot")
    print("="*60 + "\n")

    # Show plots sequentially
    print("Showing Plot 1/3...")
    plot_3d_surface(f, grad_f, x0, x_min, path, costs, learning_rate)

    print("Showing Plot 2/3...")
    plot_2d_contour(f, grad_f, x0, x_min, path, costs, learning_rate)

    print("Showing Plot 3/3...")
    plot_convergence(f, grad_f, x0, x_min, path, costs, learning_rate)

    print("\nVisualization complete!")

    return


###-------------------------------------------------------------------------------

# Main execution

if __name__ == "__main__":
    print("=== Gradient Descent Optimization Demo ===\n")
    f = quadratic
    grad_f = quadratic_grad

    print(f"Starting point: {start_point}, Learning rate: {learning_rate}")

    # Run gradient descent optimization
    x_min, path, costs = gradient_descent(f, grad_f, start_point, learning_rate=learning_rate)

    # Run optimization with visualization
    visualize_gradient_descent(f, grad_f, start_point, x_min, path, costs, learning_rate=0.01)

    print(f"Optimization completed in {len(path)} steps")
    print(f"Distance from true minimum (0,0): {np.linalg.norm(x_min):.6f}")
    print("-" * 50)



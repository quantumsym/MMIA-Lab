#!/usr/bin/env  python3
#
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
learning_rate=0.01
start_point=[-4.0,3.0]
n_steps = 35000

#---------------------------
# Rosenbrock function definition
def rosenbrock(params):
    x, y = params
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

#---------------------------

def plot_rosenbrock_2d_contour(traj_x, traj_y):
    # 2D contour plot of Rosenbrock function with optimization path
    x = jnp.linspace(-4.5, 4.5, 400)
    y = jnp.linspace(-3, 7, 400)
    X, Y = jnp.meshgrid(x, y)
    Z = rosenbrock(jnp.array([X, Y]))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contour lines with logarithmic levels for better visualization
    levels = jnp.logspace(-1, 3.5, 35)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6, linewidths=0.8)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Rosenbrock function value', fontsize=11)
    
    # Plot optimization trajectory
    ax.plot(traj_x, traj_y, 'r-', linewidth=2, alpha=0.7, label='Optimization path')
    ax.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(traj_x[-1], traj_y[-1], 'r*', markersize=15, label='End', zorder=5)
    ax.plot(1.0, 1.0, 'b*', markersize=15, label='Global minimum', zorder=5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Adam Optimizer Path on Rosenbrock Function (2D Contour)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("adam_jax_optax_rosenbrock_2d.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show()

#---------------------------
def plot_rosenbrock_path(traj_x, traj_y, traj_z):
    # 3D surface plot of Rosenbrock function with optimization path
    x = jnp.linspace(-4, 4, 400)
    y = jnp.linspace(-3, 7, 400)
    X, Y = jnp.meshgrid(x, y)
    Z = rosenbrock(jnp.array([X, Y]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0)
    # Plot optimization trajectory
    ax.plot(traj_x, traj_y, traj_z, color='r', marker='.', label='Adam optimization path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Rosenbrock')
    ax.set_title('Adam optimizer on Rosenbrock function')
    ax.legend()
    plt.savefig("adam_jax_optax_rosenbrock_3d.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show()

#---------------------------
def main():
    # Initial parameters (start point)
    params = jnp.array(start_point)
    # Adam optimizer setup
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Store parameter trajectory for plotting
    trajectory = [params.copy()]


    @jax.jit
    def update(params, opt_state):
        # Compute loss and gradient
        loss, grads = jax.value_and_grad(rosenbrock)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    for step in range(n_steps):
        params, opt_state, loss = update(params, opt_state)
        trajectory.append(params.copy())

    trajectory = jnp.stack(trajectory)
    traj_x, traj_y = trajectory[:, 0], trajectory[:, 1]
    traj_z = jnp.array([rosenbrock(jnp.array([x, y])) for x, y in zip(traj_x, traj_y)])

    # Print optimization summary
    print("="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Starting point:    ({start_point[0]:.6f}, {start_point[1]:.6f})")
    print(f"Final point:       ({params[0]:.6f}, {params[1]:.6f})")
    print(f"Number of steps:   {n_steps}")
    print(f"Initial loss:      {traj_z[0]:.6f}")
    print(f"Final loss:        {loss:.6f}")
    print(f"Optimal point:     (1.000000, 1.000000)")
    print(f"Distance from opt: {jnp.sqrt((params[0]-1.0)**2 + (params[1]-1.0)**2):.6f}")
    print("="*60)

    # Call the plot functions
    print("\nGenerating 2D contour plot...")
    plot_rosenbrock_2d_contour(traj_x, traj_y)
    
    print("Generating 3D surface plot...")
    plot_rosenbrock_path(traj_x, traj_y, traj_z)

if __name__ == "__main__":
    main()


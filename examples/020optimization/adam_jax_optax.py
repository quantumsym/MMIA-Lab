#!/usr/bin/env  python3
#
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
learning_rate=0.01
n_steps = 50000
# Initial parameters (start point)
start_point=[-4.0,3.0]

# Rosenbrock function definition
def rosenbrock(params):
    x, y = params
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

#---------------------------

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
    plt.savefig("adam_jax_optax_rosenbrock.svg")
    plt.show()

def main():
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

    # Call the plot function
    plot_rosenbrock_path(traj_x, traj_y, traj_z)

if __name__ == "__main__":
    main()


#!/usr/bin/env python
#

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
m = 1.0          # particle mass

# Initial conditions (position and momentum)
q_initial = np.array([0.1, 0.0])    # initial position [x, y]
p_initial = np.array([0.0, 0.5])    # initial momentum [px, py]

# Integration parameters
h = 0.01         # time step
n_steps = 1      # leapfrog steps per call
total_time = 50.0  # total simulation time
n_iterations = int(total_time / h)




###------------------------------------------------------------------------------------

def leapfrog(q, p, grad_U, h, L, m):
    """
    Leapfrog integration for Hamiltonian systems with scalar mass
    Args:
        q: current position array
        p: current momentum array
        grad_U: gradient of potential energy function (callable)
        h: integration step size
        L: number of integration steps
        m: scalar mass (float or int)
    Returns:
        q_new, p_new: updated position and momentum arrays
    """
    # Create copies to avoid modifying original arrays
    q, p = q.copy(), p.copy()

    # 1. Initial half-step update for momentum
    p = p - 0.5 * h * grad_U(q)

    # 2. L-1 full steps
    for i in range(L-1):
        # Full position update using scalar mass
        q = q + h * (p / m)
        # Full momentum update
        p = p - h * grad_U(q)

    # 3. Final full position update
    q = q + h * (p / m)

    # 4. Final half-step update for momentum
    p = p - 0.5 * h * grad_U(q)

    return q, p

def henon_heiles_potential(q):
    """
    Hénon-Heiles potential function: V(x,y) = 1/2(x² + y²) + x²y - 1/3y³

    Args:
        q: position array [x, y]
    Returns:
        potential energy value
    """
    x, y = q
    return 0.5 * (x**2 + y**2) + x**2 * y - (1/3) * y**3

def henon_heiles_gradient(q):
    """
    Gradient of Hénon-Heiles potential: ∇V = [∂V/∂x, ∂V/∂y]

    Derivatives:
    ∂V/∂x = x + 2xy
    ∂V/∂y = y + x² - y²

    Args:
        q: position array [x, y]
    Returns:
        gradient array [dV/dx, dV/dy]
    """
    x, y = q
    dV_dx = x + 2 * x * y
    dV_dy = y + x**2 - y**2
    return np.array([dV_dx, dV_dy])

def calculate_energy(q, p, m):
    """
    Calculate total energy: E = T + V = p²/(2m) + V(q)

    Args:
        q: position array [x, y]
        p: momentum array [px, py]
        m: scalar mass
    Returns:
        total energy
    """
    kinetic = np.sum(p**2) / (2 * m)  # T = p²/(2m)
    potential = henon_heiles_potential(q)  # V(x,y)
    return kinetic + potential

def simulate_henon_heiles():
    """
    Complete simulation of Hénon-Heiles system
    """

    # Storage arrays for trajectory
    positions = np.zeros((n_iterations + 1, 2))
    momenta = np.zeros((n_iterations + 1, 2))
    energies = np.zeros(n_iterations + 1)
    times = np.zeros(n_iterations + 1)

    # Set initial conditions
    positions[0] = q_initial
    momenta[0] = p_initial
    energies[0] = calculate_energy(q_initial, p_initial, m)
    times[0] = 0.0

    print(f"Hénon-Heiles System Simulation")
    print(f"Initial position: {q_initial}")
    print(f"Initial momentum: {p_initial}")
    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Integration time step: {h}")
    print(f"Total time: {total_time}")
    print("\nIntegrating...")

    # Main integration loop
    q_current = q_initial.copy()
    p_current = p_initial.copy()

    for i in range(n_iterations):
        # Integrate one step using leapfrog
        q_current, p_current = leapfrog(q_current, p_current,
                                       henon_heiles_gradient, h, n_steps, m)

        # Store results
        positions[i + 1] = q_current
        momenta[i + 1] = p_current
        energies[i + 1] = calculate_energy(q_current, p_current, m)
        times[i + 1] = (i + 1) * h

    print("Integration complete!")

    # Energy conservation check
    energy_drift = np.abs(energies[-1] - energies[0])
    print(f"Final energy: {energies[-1]:.6f}")
    print(f"Energy drift: {energy_drift:.2e}")

    return times, positions, momenta, energies

def create_visualizations(times, positions, momenta, energies):
    """
    Create comprehensive plots for Hénon-Heiles system
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Trajectory in configuration space (x-y plane)
    ax1 = axes[0, 0]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.8, alpha=0.8)
    ax1.plot(positions[0, 0], positions[0, 1], 'ro', markersize=8, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'gs', markersize=8, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Configuration Space Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Phase space (x vs px)
    ax2 = axes[0, 1]
    ax2.plot(positions[:, 0], momenta[:, 0], 'r-', linewidth=0.8, alpha=0.8)
    ax2.plot(positions[0, 0], momenta[0, 0], 'ro', markersize=8, label='Start')
    ax2.plot(positions[-1, 0], momenta[-1, 0], 'gs', markersize=8, label='End')
    ax2.set_xlabel('x')
    ax2.set_ylabel('px')
    ax2.set_title('Phase Space: x-px')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy conservation
    ax3 = axes[1, 0]
    ax3.plot(times, energies, 'g-', linewidth=2)
    ax3.axhline(y=energies[0], color='r', linestyle='--', alpha=0.7, label='Initial energy')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time evolution of coordinates
    ax4 = axes[1, 1]
    ax4.plot(times, positions[:, 0], 'b-', label='x(t)', linewidth=1.5)
    ax4.plot(times, positions[:, 1], 'r-', label='y(t)', linewidth=1.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Position')
    ax4.set_title('Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return
    

def print_basic_stats(positions,energies):
    # Display some trajectory statistics
    print(f"\nTrajectory Statistics:")
    print(f"Maximum x: {np.max(positions[:, 0]):.4f}")
    print(f"Minimum x: {np.min(positions[:, 0]):.4f}")
    print(f"Maximum y: {np.max(positions[:, 1]):.4f}")
    print(f"Minimum y: {np.min(positions[:, 1]):.4f}")
    print(f"Energy conservation error: {np.std(energies):.2e}")
    return

###--------------------------------------------------------------

# Main execution
if __name__ == "__main__":
    # Run the complete simulation
    times, positions, momenta, energies = simulate_henon_heiles()

    # Create visualizations
    create_visualizations(times, positions, momenta, energies)

    # Display some trajectory statistics
    print_basic_stats(positions,energies)




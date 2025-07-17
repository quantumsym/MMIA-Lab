#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt


def brownian_motion_1d(n_particles=100, n_steps=1000, dt=0.01, D=1.0):
    """
    Simula moto browniano: dx = √(2D*dt) * N(0,1)
    """
    # Parametri
    sigma = np.sqrt(2 * D * dt)

    # inizializza random number generator
    rng = np.random.default_rng()

    # Inizializza particelle
    positions = np.zeros((n_particles, n_steps))

    # Evoluzione
    for step in range(1, n_steps):
        random_increments = rng.normal(0, sigma, n_particles)
        positions[:, step] = positions[:, step-1] + random_increments
    return positions

# Simulazione
brownian_paths = brownian_motion_1d(50, 1000, 0.01, 0.5)
time = np.arange(brownian_paths.shape[1]) * 0.01


# Visualizza alcune traiettorie
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)


for i in range(min(10, brownian_paths.shape[0])):
    plt.plot(time, brownian_paths[i, :], alpha=0.7, linewidth=1)
plt.title('Traiettorie Browniane')
plt.xlabel('Tempo')
plt.ylabel('Posizione')
plt.grid(True, alpha=0.3)


# Distribuzione posizioni finali
plt.subplot(2, 2, 2)
final_positions = brownian_paths[:, -1]
plt.hist(final_positions, bins=20, density=True, alpha=0.7, edgecolor='black')
plt.title('Distribuzione Posizioni Finali')
plt.xlabel('Posizione')
plt.ylabel('Densità')


# Confronto con teoria gaussiana
x_theory = np.linspace(final_positions.min(), final_positions.max(), 100)
variance_theory = 2 * 0.5 * time[-1]  # 2*D*t
y_theory = np.exp(-x_theory**2 / (2*variance_theory)) / np.sqrt(2*np.pi*variance_theory)
# plt.plot(x_theory,a)

plt.show()




#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt

# Parametri simulazione
n_walks  = 5
n_steps  = 500
p_right  = 0.75      # probabilità di muoversi a destra (deriva verso +)
q_left   = 1 - p_right

# Funzione biased random walk 1d
def biased_random_walk_1d(n_steps=1000, p_right=0.6):
    """
    Random walk 1D con deriva:
    - p_right: probabilità di passo +1 (destra)
    - q = 1 - p_right: probabilità di passo -1 (sinistra)
    """
    steps = np.random.choice([-1, 1], size=n_steps, p=[1 - p_right, p_right])
    position = np.cumsum(steps)
    position = np.insert(position, 0, 0)  # posizione iniziale
    return position



plt.figure(figsize=(12, 8))

# Plot 1: Singole traiettorie con deriva
plt.subplot(2, 2, 1)
for i in range(n_walks):
    walk = biased_random_walk_1d(n_steps, p_right)
    plt.plot(walk, alpha=0.8, linewidth=1.5, label=f'Walk {i+1}')
plt.title(f'Random Walk 1D (p={p_right:.2f}) - Traiettorie')
plt.xlabel('Passi')
plt.ylabel('Posizione')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Distribuzione delle posizioni finali
plt.subplot(2, 2, 2)
n_hist = 200
final_positions = [biased_random_walk_1d(n_steps, p_right)[-1] for _ in range(n_hist)]
plt.hist(final_positions, bins=25, alpha=0.7, density=True, edgecolor='black')
plt.title(f'Distribuzione Posizioni Finali ({n_hist} camminate)')
plt.xlabel('Posizione finale')
plt.ylabel('Densità')
plt.grid(True, alpha=0.3)
# Curva gaussiana teorica con deriva
x_theory   = np.linspace(min(final_positions), max(final_positions), 200)
mu_theory  = (2 * p_right - 1) * n_steps
var_theory = 4 * p_right * q_left * n_steps  # 4pqN
y_theory   = np.exp(-(x_theory - mu_theory) ** 2 / (2 * var_theory)) \
             / np.sqrt(2 * np.pi * var_theory)
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label='Teoria N(μ,σ)')
plt.legend()

# Plot 3: Distanza quadratica media (MSD)
plt.subplot(2, 2, 3)
walk_long = biased_random_walk_1d(1000, p_right)
msd = walk_long ** 2
plt.plot(msd, 'b-', alpha=0.7, label='MSD empirica')
theory_line = 4 * p_right * q_left * np.arange(len(msd))
plt.plot(theory_line, 'r--', label='Teoria (4pq·t)')
plt.title('Distanza Quadratica Media con Deriva')
plt.xlabel('Passi')
plt.ylabel('Posizione²')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Media e deviazione standard su molte camminate
plt.subplot(2, 2, 4)
n_real   = 50
matrix   = np.array([biased_random_walk_1d(n_steps, p_right) for _ in range(n_real)])
mean_pos = np.mean(matrix, axis=0)
std_pos  = np.std(matrix, axis=0)
time     = np.arange(len(mean_pos))
plt.plot(time, mean_pos, 'b-', linewidth=2, label='Media empirica')
plt.fill_between(time, mean_pos - std_pos, mean_pos + std_pos, alpha=0.3, label='±1 std')
# Curva media teorica
plt.plot(time, (2 * p_right - 1) * time, 'r--', label='Teoria μ(t)')
plt.title(f'Media e Deviazione Standard ({n_real} camminate)')
plt.xlabel('Passi')
plt.ylabel('Posizione')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistiche finali
mean_final = np.mean([biased_random_walk_1d(n_steps, p_right)[-1] for _ in range(1000)])
std_final  = np.std([biased_random_walk_1d(n_steps, p_right)[-1] for _ in range(1000)])
print(f"p = {p_right:.2f}, q = {q_left:.2f}")
print(f"Media teorica finale: {(2*p_right-1)*n_steps:.2f}")
print(f"Varianza teorica finale: {4*p_right*q_left*n_steps:.2f}")
print(f"Media simulata finale: {mean_final:.2f}")
print(f"Dev.std simulata finale:  {std_final:.2f}")



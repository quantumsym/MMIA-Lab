#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Parametri di simulazione
# --------------------------
L        = 64          # lato del reticolo
T        = 1.5         # temperatura ridotta (J/kB)
n_sweep  = 1_000       # sweep Monte Carlo totali
rng      = np.random.default_rng()

# --------------------------
# 2. Inizializzazione
# --------------------------
spins = rng.choice([-1, 1], size=(L, L))

def delta_E(lattice, i, j):
    """Variazione di energia per un flip dello spin (i,j)."""
    nn_sum = (
        lattice[(i-1) % L, j] + lattice[(i+1) % L, j] +
        lattice[i, (j-1) % L] + lattice[i, (j+1) % L]
    )
    return 2 * lattice[i, j] * nn_sum

def glauber_step(lattice):
    """Un singolo aggiornamento Glauber (heat-bath) su uno spin casuale."""
    i = rng.integers(L)
    j = rng.integers(L)
    dE = delta_E(lattice, i, j)
    if rng.random() < 1.0 / (1.0 + np.exp(dE / T)):
        lattice[i, j] *= -1

# --------------------------
# 3. Loop Monte Carlo
# --------------------------
magnetization = np.empty(n_sweep)

for sweep in range(n_sweep):
    # L×L passi ≈ 1 sweep
    for _ in range(L * L):
        glauber_step(spins)
    magnetization[sweep] = spins.mean()      # M(t)

# --------------------------
# 4. Plot dei risultati
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# (1) Magnetizzazione vs tempo
ax1.plot(magnetization, color="tab:blue")
ax1.set_xlabel("Sweep Monte Carlo")
ax1.set_ylabel("Magnetizzazione M(t)")
ax1.set_title(f"Andamento di M(t) a T={T}")
ax1.grid(True, alpha=0.3)

# (2) Configurazione finale
im = ax2.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1)
ax2.set_title("Configurazione finale")
ax2.set_axis_off()
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_ticks([-1, 1])
cbar.set_ticklabels(["-1", "+1"])

plt.tight_layout()
plt.show()


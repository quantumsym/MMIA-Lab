#!/usr/bin/env python3
#
import numpy as np
def gibbs_discrete_example(num_samples=1000):
    """
    Esempio semplice di Gibbs con distribuzione discreta.
    Supponiamo P(X=1, Y=1) = 0.5, P(X=1, Y=0) = 0.2
    P(X=0, Y=1) = 0.2, P(X=0, Y=0) = 0.1
    """
    samples = []
    x, y = 0, 0  # inizializzazione
    for i in range(num_samples):
        # Campiona X dato Y
        if y == 1:
            # P(X=1|Y=1) = 0.5/(0.5+0.2) = 0.714
            x = 1 if np.random.rand() < 0.714 else 0
        else:
            # P(X=1|Y=0) = 0.2/(0.2+0.1) = 0.667
            x = 1 if np.random.rand() < 0.667 else 0
        # Campiona Y dato X
        if x == 1:
            # P(Y=1|X=1) = 0.5/(0.5+0.2) = 0.714
            y = 1 if np.random.rand() < 0.714 else 0
        else:
            # P(Y=1|X=0) = 0.2/(0.2+0.1) = 0.667
            y = 1 if np.random.rand() < 0.667 else 0
        samples.append([x, y])
    return np.array(samples)
# Testa l'esempio
discrete_samples = gibbs_discrete_example(5000)
print("Frequenze osservate:")
print(f"(0,0): {np.mean((discrete_samples[:, 0] == 0) & (discrete_samples[:, 1] == 0)):.3f}")
print(f"(0,1): {np.mean((discrete_samples[:, 0] == 0) & (discrete_samples[:, 1] == 1)):.3f}")
print(f"(1,0): {np.mean((discrete_samples[:, 0] == 1) & (discrete_samples[:, 1] == 0)):.3f}")
print(f"(1,1): {np.mean((discrete_samples[:, 0] == 1) & (discrete_samples[:, 1] == 1)):.3f}")


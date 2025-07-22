
import numpy as np
import matplotlib.pyplot as plt

# Funzione random walk 1d

# Genera una lista num_steps casuali: +1 o -1 con probabilità 50/50
# con la funzione choice di numpy.random
random_steps = np.random.choice([-1, 1], size=num_steps)

# Calcola posizione cumulativa
# con la funzione cumsum di numpy
position = np.cumsum(random_steps)

# Aggiunge posizione iniziale (0) all'inizio
# con la funzione insert di numpy
position = np.insert(position, 0, 0)

# *** NUOVO: Calcolo della posizione media e scarto quadratico medio ***
mean_position = np.mean(position)
std_position = np.std(position)

# Stampa le statistiche
print(f"Posizione media: {mean_position:.4f}")
print(f"Scarto quadratico medio: {std_position:.4f}")
print(f"Posizione finale: {position[-1]}")
print(f"Posizione minima: {np.min(position)}")
print(f"Posizione massima: {np.max(position)}")

# Visualizzazione

# Creo un plot con i valori nella lista position
plt.plot(range(num_steps + 1), position, marker='o', markersize=2)

# Aggiungo il titolo
plt.title('Random Walk 1D con Statistiche')

# Aggiungo le etichette sugli assi ed una griglia
plt.xlabel('Passo')
plt.ylabel('Posizione')
plt.grid(True)

# *** NUOVO: Aggiungo le statistiche come annotazioni sul grafico ***
plt.text(0.02, 0.98, f'Media: {mean_position:.2f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(0.02, 0.90, f'Std: {std_position:.2f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Mostro la figura
plt.show()


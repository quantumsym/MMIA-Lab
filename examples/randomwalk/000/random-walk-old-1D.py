
]
import numpy as np
import matplotlib.pyplot as plt

num_steps = 100  # numero di passi

# funzione random walk  1d

# Genera una lista num_steps casuali: +1 o -1 con probabilità 50/50 
# con la funzione choiche di numpy.random 
random_steps = np.random.choice([-1, 1], size=num_steps)

# Calcola posizione cumulativa
# con la funzione cumsum di numpy
position = np.cumsum(random_steps)

# Aggiunge posizione iniziale (0) all'inizio
# con la funzione insert di numpy
position = np.insert(position, 0, 0)

# visualizzazione

# creo un plot con i valori nella lista position
plt.plot(range(num_steps + 1), position, marker='o')

# aggiungo il titolo
plt.title('Random Walk 1D')

# aggiungo le etichette sugli assi ed una griglia
plt.xlabel('Passo')
plt.ylabel('Posizione')
plt.grid(True)

# mostro la figura
plt.show()


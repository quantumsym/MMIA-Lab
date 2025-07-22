#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
N = 150  # numero di passi


# random walk  1d
def random_walk_1d(N=100):
    # generatore di numeri casuali
    rng = np.random.default_rng()

    # genera N spostamenti casuali equiprobabili, -1 o +1
    random_steps = rng.choice([-1, 1], size=N)

    # genera il vettore posizioni con la funzione cumsum (somma cumulativa di un'array)
    position = np.cumsum(random_steps)

    # inserisce la posizione iniziale in testa alla lista
    position = np.insert(position, 0, 0)  # aggiungi la posizione iniziale

    return position
 
# visualizzazione
def plot_random_walk(position, N=100):
    # grafico della posizione dopo ogni passo
    plt.plot(range(N + 1), position, marker='o')

    # titoli, etichette assi e griglia
    plt.title('Random Walk 1D')
    plt.xlabel('Passo')
    plt.ylabel('Posizione')
    plt.grid(True)

    # mostra il grafico
    plt.show()


#
# Esecuzione della simulazione
#
position = random_walk_1d(N)

#  Calcolo posizione media e scarto quadratico medio
mean_position = np.mean(position)
std_position = np.std(position)

# Stampa le statistiche
print(f"Posizione media: {mean_position:.4f}")
print(f"Scarto quadratico medio: {std_position:.4f}")
print(f"Posizione finale: {position[-1]}")
print(f"Posizione minima: {np.min(position)}")
print(f"Posizione massima: {np.max(position)}")

# Visualizzo il grafico
plot_random_walk(position,N)





#!/usr/bin/python
#
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HopfieldNetwork:
    def __init__(self, n_neurons):
        """
        Inizializza la rete di Hopfield con un numero specificato di neuroni.
        
        Args:
            n_neurons (int): Numero di neuroni nella rete
        """
        self.n_neurons = n_neurons
        # Matrice dei pesi (simmetrica, diagonale a zero)
        self.weights = np.zeros((n_neurons, n_neurons))
    
    def train(self, patterns):
        """
        Addestra la rete utilizzando la regola di Hebb.
        
        Args:
            patterns (list): Lista di pattern di addestramento (array numpy)
        """
        for pattern in patterns:
            # Normalizza i pattern a -1 e 1
            p = np.where(pattern >= 0, 1, -1).flatten()
            # Aggiorna i pesi con la regola di Hebb
            self.weights += np.outer(p, p)
        
        # Imposta la diagonale a zero (nessuna auto-connessione)
        np.fill_diagonal(self.weights, 0)
        # Normalizza i pesi
        self.weights /= self.n_neurons
    
    def energy(self, state):
        """
        Calcola l'energia della rete per un dato stato.
        
        Args:
            state (np.array): Stato corrente della rete
            
        Returns:
            float: Energia della configurazione
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def predict(self, input_pattern, max_steps=100, mode='async'):
        """
        Recupera un pattern dalla memoria associativa.
        
        Args:
            input_pattern (np.array): Pattern di input (eventualmente rumoroso)
            max_steps (int): Numero massimo di iterazioni
            mode (str): 'async' per aggiornamento asincrono, 'sync' per sincrono
            
        Returns:
            tuple: (Stato finale, Storia degli stati, Energia)
        """
        # Inizializza lo stato
        state = np.where(input_pattern.flatten() >= 0, 1, -1)
        states_history = [state.copy()]
        energy_history = [self.energy(state)]
        
        # Funzione di aggiornamento
        def update_async():
            nonlocal state
            for _ in range(self.n_neurons):
                i = np.random.randint(self.n_neurons)
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
        
        def update_sync():
            nonlocal state
            activation = np.dot(self.weights, state)
            state = np.where(activation >= 0, 1, -1)
        
        # Ciclo principale
        for _ in range(max_steps):
            if mode == 'async':
                update_async()
            else:
                update_sync()
            
            states_history.append(state.copy())
            energy_history.append(self.energy(state))
            
            # Verifica convergenza
            if np.array_equal(states_history[-1], states_history[-2]):
                break
        
        return state, states_history, energy_history
    
    def plot_weights(self):
        """Visualizza la matrice dei pesi"""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.weights, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Matrice dei Pesi")
        plt.show()



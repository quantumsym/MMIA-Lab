#!/usr/bin/python
#
#

if __name__ == "__main__":
    # Parametri della simulazione
    n_neurons = 100  # 10x10 grid
    patterns = [
        np.array([1 if (i//10 + i%10) % 2 == 0 else -1 for i in range(100)]),  # Scacchiera
        np.array([1 if i%10 < 5 else -1 for i in range(100)]),  # Metà superiore
        np.array([1 if i//10 < 5 else -1 for i in range(100)]),  # Metà sinistra
    ]
    
    # Crea e addestra la rete
    hopfield = HopfieldNetwork(n_neurons)
    hopfield.train(patterns)
    
    # Crea un pattern rumoroso (20% di rumore)
    original = patterns[^0].copy()
    noisy = original.copy()
    flip_indices = np.random.choice(n_neurons, size=int(n_neurons*0.2), replace=False)
    noisy[flip_indices] *= -1
    
    # Ricostruzione
    reconstructed, states_history, energy_history = hopfield.predict(
        noisy, max_steps=100, mode='async'
    )
    
    # Visualizzazione risultati
    def plot_pattern(pattern, title):
        plt.imshow(pattern.reshape(10, 10), cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_pattern(original, "Pattern Originale")
    plt.subplot(1, 3, 2)
    plot_pattern(noisy, "Pattern Rumoroso")
    plt.subplot(1, 3, 3)
    plot_pattern(reconstructed, "Pattern Ricostruito")
    plt.tight_layout()
    plt.show()
    
    # Grafico dell'energia
    plt.figure(figsize=(10, 5))
    plt.plot(energy_history)
    plt.xlabel("Iterazioni")
    plt.ylabel("Energia")
    plt.title("Dinamica dell'Energia")
    plt.grid(True)
    plt.show()
    
    # Matrice dei pesi
    hopfield.plot_weights()


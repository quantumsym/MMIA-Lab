#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import HopfieldMonteCarlo

if __name__ == "__main__":
    # Parametri simulazione
    n_neurons = 100  # 10x10 grid
    n_patterns = 3
    
    # Genera pattern di test
    patterns = []
    patterns.append(np.array([1 if (i//10 + i%10) % 2 == 0 else -1 
                             for i in range(n_neurons)]))  # Scacchiera
    patterns.append(np.array([1 if i%10 < 5 else -1 
                             for i in range(n_neurons)]))  # Metà verticale
    patterns.append(np.array([1 if i//10 < 5 else -1 
                             for i in range(n_neurons)]))  # Metà orizzontale
    
    print("🧠 Inizializzazione rete di Hopfield con Monte Carlo...")
    
    # Crea e addestra la rete
    hopfield_mc = HopfieldMonteCarlo(
        n_neurons=n_neurons, 
        temperature=2.0, 
        cooling_rate=0.99
    )
    hopfield_mc.train(patterns)
    
    print(f"✅ Addestramento completato con {len(patterns)} pattern")
    
    # Crea pattern rumoroso (30% rumore)
    target = patterns[0].copy()
    noisy = target.copy()
    noise_indices = np.random.choice(n_neurons, size=int(n_neurons*0.3), replace=False)
    noisy[noise_indices] *= -1
    
    print(f"🎯 Pattern rumoroso creato (30% rumore)")
    
    # Simulazione con Simulated Annealing
    print("\n🔥 Avvio Simulated Annealing...")
    sa_result = hopfield_mc.simulate_annealing(
        noisy, 
        max_iterations=5000,
        min_temperature=0.01
    )
    
    print(f"✅ Simulazione completata!")
    print(f"   • Energia finale: {sa_result['best_energy']:.4f}")
    print(f"   • Tasso accettazione: {sa_result['acceptance_rate']:.3f}")
    
    # Calcola overlap con pattern originale
    overlap = np.dot(sa_result['best_state'], target) / n_neurons
    print(f"   • Overlap con target: {overlap:.3f}")
    
    # Visualizzazione pattern
    def plot_patterns(original, noisy, recovered, title="Recupero Pattern"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        axes[0].imshow(original.reshape(10, 10), cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title("Pattern Originale")
        axes[0].axis('off')
        
        axes[1].imshow(noisy.reshape(10, 10), cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title("Pattern Rumoroso")
        axes[1].axis('off')
        
        axes[2].imshow(recovered.reshape(10, 10), cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title(f"Recuperato (overlap: {overlap:.3f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    plot_patterns(target, noisy, sa_result['best_state'])
    hopfield_mc.plot_results(sa_result, "Simulazione Hopfield Monte Carlo")
    
    # Test Parallel Tempering
    print("\n🌡️ Test Parallel Tempering...")
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    pt_result = hopfield_mc.parallel_tempering(
        noisy, 
        temperatures=temperatures,
        max_iterations=2000
    )
    
    print(f"✅ Parallel Tempering completato!")
    print(f"   • Tasso scambio: {pt_result['exchange_rate']:.3f}")
    
    # Trova migliore replica (temperatura più bassa)
    best_idx = np.argmin(pt_result['energies'])
    best_temp = temperatures[best_idx]
    best_overlap = np.dot(pt_result['states'][best_idx], target) / n_neurons
    
    print(f"   • Migliore replica: T={best_temp}, overlap={best_overlap:.3f}")
    
    # Analisi bacini di attrazione
    print("\n🎯 Analisi bacini di attrazione...")
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    basin_analysis = hopfield_mc.analyze_basin_attraction(
        target, 
        noise_levels=noise_levels,
        n_trials=20,
        max_iterations=1000
    )
    
    # Plot analisi bacini
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(noise_levels, basin_analysis['success_rates'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Livello Rumore')
    plt.ylabel('Tasso Successo')
    plt.title('Bacini di Attrazione')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(noise_levels, basin_analysis['avg_energies'], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Livello Rumore')
    plt.ylabel('Energia Media Finale')
    plt.title('Energia vs Rumore')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("🎉 Simulazione completata con successo!")


